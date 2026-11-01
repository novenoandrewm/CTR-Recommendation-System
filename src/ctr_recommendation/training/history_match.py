"""Controlled B1-versus B2A historical item-matching experiments."""

# Import standard libraries
import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Import necessary classes
from ctr_recommendation.data.splits import (
    ValidationWindow,
    load_validation_windows,
    validate_window,
)
from ctr_recommendation.data.validate import load_input_contract
from ctr_recommendation.evaluation.metrics import score_by_rowid
from ctr_recommendation.training.baseline import (
    SMOOTHING,
    read_candidates,
)
from ctr_recommendation.training.user_baseline import USER_WEIGHT

HISTORY_BONUS = 0.05


@dataclass(frozen=True)
class HistoryMatchResult:
    """Results of a controlled B1/B2A comparison."""

    fitting_rows: int
    validation_events: int
    validation_rows: int
    clicked_events: int
    matched_candidates: int
    matched_clicks: int
    b0_auc: float
    b1_auc: float
    b2_auc: float
    b1_event_auc: float
    b2_event_auc: float


def within_event_auc(
    events: dict[str, list[tuple[int, float, float]]],
    score_index: int,
) -> tuple[float, int]:
    """Rank clicked candidates against negatives in their own events.

    Events without clicks are excluded from this diagnostic and
    are reported separately through the returned clicked-event count."""

    concordant = 0.0
    total_pairs = 0
    clicked_events = 0

    for rows in events.values():
        positives = [row[score_index] for row in rows if row[0] == 1]

        if len(positives) > 1:
            raise ValueError("Event contains multiple clicks")

        if not positives:
            continue

        clicked_events += 1
        positive_score = positives[0]

        for row in rows:
            if row[0] != 0:
                continue

            negative_score = row[score_index]

            if positive_score > negative_score:
                concordant += 1.0
            elif positive_score == negative_score:
                concordant += 0.5

            total_pairs += 1

    if total_pairs == 0:
        raise ValueError("No clicked validation events")

    return concordant / total_pairs, clicked_events


def evaluate_history_match(
    csv_path: Path,
    contract_path: Path,
    window: ValidationWindow,
) -> HistoryMatchResult:
    """Compare B0, B1 and B2A on identical temporal validation rows."""

    validate_window(window)

    contract = load_input_contract(contract_path)

    item_counts: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])

    user_counts: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])

    selected_by_user: dict[str, tuple[int, str]] = {}

    fitting_rows = 0
    fitting_clicks = 0
    previous_timestamp: int | None = None

    print("[B2A] Pass 1: fitting historical statistics...", flush=True)

    for record in read_candidates(csv_path, contract):
        if previous_timestamp is not None and record.timestamp < previous_timestamp:
            raise ValueError("Training CSV is not chronological")

        previous_timestamp = record.timestamp

        if record.timestamp >= window.validate_before:
            break

        if record.timestamp < window.fit_before:
            if record.clicked not in (0, 1):
                raise ValueError("Invalid fitting label")

            item = item_counts[record.item_id]
            item[0] += 1
            item[1] += record.clicked

            user = user_counts[record.user_id]
            user[0] += 1
            user[1] += record.clicked

            fitting_rows += 1
            fitting_clicks += record.clicked

        elif record.timestamp >= window.validate_from:
            candidate_event = (
                record.timestamp,
                record.event_id,
            )

            previous_event = selected_by_user.get(record.user_id)

            if previous_event is None or candidate_event < previous_event:
                selected_by_user[record.user_id] = candidate_event

    if fitting_rows == 0:
        raise ValueError("No fitting observations")

    if not selected_by_user:
        raise ValueError("No validation events")

    global_prior = fitting_clicks / fitting_rows

    selected_event_ids = {event_id for _, event_id in selected_by_user.values()}

    print(
        f"[B2A] Fitting rows: {fitting_rows:,}",
        flush=True,
    )

    print(
        f"[B2A] Selected events: {len(selected_event_ids):,}",
        flush=True,
    )

    print("[B2A] Pass 2: scoring validation rows...", flush=True)

    truth: list[tuple[str, int]] = []

    b0_predictions: list[tuple[str, float]] = []
    b1_predictions: list[tuple[str, float]] = []
    b2_predictions: list[tuple[str, float]] = []

    event_rows: dict[str, list[tuple[int, float, float]]] = defaultdict(list)

    matched_candidates = 0
    matched_clicks = 0

    for record in read_candidates(csv_path, contract):
        if record.timestamp >= window.validate_before:
            break

        if record.event_id not in selected_event_ids:
            continue

        if record.clicked not in (0, 1):
            raise ValueError("Invalid validation label")

        item_impressions, item_clicks = item_counts.get(
            record.item_id,
            [0, 0],
        )

        user_impressions, user_clicks = user_counts.get(
            record.user_id,
            [0, 0],
        )

        # B0: historical item popularity.
        b0_score = (item_clicks + SMOOTHING * global_prior) / (item_impressions + SMOOTHING)

        # B1: historical item popularity + user propensity.
        user_probability = (user_clicks + SMOOTHING * global_prior) / (user_impressions + SMOOTHING)

        b1_score = b0_score + USER_WEIGHT * (user_probability - global_prior)

        b1_score = min(1.0, max(0.0, b1_score))

        # B2A: add candidate-specific historical item matching.
        history_match = int(record.item_id in record.history_item_ids)

        b2_score = min(
            1.0,
            max(
                0.0,
                b1_score + HISTORY_BONUS * history_match,
            ),
        )

        if history_match:
            matched_candidates += 1
            matched_clicks += record.clicked

        truth.append((record.rowid, record.clicked))

        b0_predictions.append((record.rowid, b0_score))
        b1_predictions.append((record.rowid, b1_score))
        b2_predictions.append((record.rowid, b2_score))

        event_rows[record.event_id].append((record.clicked, b1_score, b2_score))

    expected_candidates = contract["events"]["expected_candidate_count"]

    for event_id in selected_event_ids:
        actual = len(event_rows.get(event_id, []))

        if actual != expected_candidates:
            raise ValueError(
                f"Validation event {event_id} has {actual} "
                f"candidates; expected {expected_candidates}"
            )

    b0_auc = score_by_rowid(truth, b0_predictions)
    b1_auc = score_by_rowid(truth, b1_predictions)
    b2_auc = score_by_rowid(truth, b2_predictions)

    b1_event_auc, clicked_events = within_event_auc(
        event_rows,
        score_index=1,
    )

    b2_event_auc, b2_clicked_events = within_event_auc(
        event_rows,
        score_index=2,
    )

    if clicked_events != b2_clicked_events:
        raise ValueError("Diagnostic event populations differ")

    return HistoryMatchResult(
        fitting_rows=fitting_rows,
        validation_events=len(selected_event_ids),
        validation_rows=len(truth),
        clicked_events=clicked_events,
        matched_candidates=matched_candidates,
        matched_clicks=matched_clicks,
        b0_auc=b0_auc,
        b1_auc=b1_auc,
        b2_auc=b2_auc,
        b1_event_auc=b1_event_auc,
        b2_event_auc=b2_event_auc,
    )


def main() -> None:
    """Run the preregistered B2A experiment."""

    parser = argparse.ArgumentParser(description="Evaluate historical item matching")

    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/raw/train.csv"),
    )

    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("contracts/input_schema.json"),
    )

    parser.add_argument(
        "--validation-config",
        type=Path,
        default=Path("configs/validation.yaml"),
    )

    parser.add_argument(
        "--window",
        default="development",
    )

    args = parser.parse_args()

    windows = load_validation_windows(args.validation_config)

    if args.window not in windows:
        raise ValueError(f"Unknown window: {args.window}")

    result = evaluate_history_match(
        args.train,
        args.contract,
        windows[args.window],
    )

    print()
    print("=" * 56)
    print("EXPERIMENT B2A — HISTORICAL ITEM MATCHING")
    print("=" * 56)

    print(f"Window: {args.window}")
    print(f"Fitting rows: {result.fitting_rows:,}")
    print(f"Validation events: {result.validation_events:,}")
    print(f"Validation rows: {result.validation_rows:,}")

    print()
    print(f"Clicked events: {result.clicked_events:,}")
    print(f"Events without clicks: {result.validation_events - result.clicked_events:,}")

    print(f"History-matched candidates: {result.matched_candidates:,}")
    print(f"History-matched clicks: {result.matched_clicks:,}")

    print()
    print(f"B0 Global AUC: {result.b0_auc:.6f}")
    print(f"B1 Global AUC: {result.b1_auc:.6f}")
    print(f"B2 Global AUC: {result.b2_auc:.6f}")

    print(f"B2 - B1: {result.b2_auc - result.b1_auc:+.6f}")

    print()
    print(f"B1 Within-event AUC: {result.b1_event_auc:.6f}")
    print(f"B2 Within-event AUC: {result.b2_event_auc:.6f}")

    print(f"Within-event difference: {result.b2_event_auc - result.b1_event_auc:+.6f}")

    print("=" * 56)


if __name__ == "__main__":
    main()
