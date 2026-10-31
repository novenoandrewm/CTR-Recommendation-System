"""Compare item CTR against item CTR plus user click propensity."""

# Import standard libraries
import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Import necesarry classses
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

USER_WEIGHT = 0.5


@dataclass(frozen=True)
class UserBaselineResult:
    """Results from a controlled B0-versus-B1 experiment."""

    fitting_rows: int
    validation_events: int
    validation_rows: int
    known_user_events: int
    item_auc: float
    user_auc: float


def evaluate_user_baseline(
    csv_path: Path,
    contract_path: Path,
    window: ValidationWindow,
    *,
    user_weight: float = USER_WEIGHT,
) -> UserBaselineResult:
    """Compare models using identical temporal validation records."""

    validate_window(window)

    if not 0.0 <= user_weight <= 1.0:
        raise ValueError("User weight must lie between 0 and 1")

    contract = load_input_contract(contract_path)

    item_counts: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])

    user_counts: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])

    selected_by_user: dict[str, tuple[int, str]] = {}

    fitting_rows = 0
    fitting_clicks = 0
    previous_timestamp: int | None = None

    print("[B1] Pass 1: fitting historical statistics...", flush=True)

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

    known_user_events = sum(user_id in user_counts for user_id in selected_by_user)

    print(
        f"[B1] Fitting rows: {fitting_rows:,}",
        flush=True,
    )

    print(
        f"[B1] Validation events: {len(selected_event_ids):,}",
        flush=True,
    )

    print(
        "[B1] Pass 2: scoring identical validation rows...",
        flush=True,
    )

    truth: list[tuple[str, int]] = []
    item_predictions: list[tuple[str, float]] = []
    user_predictions: list[tuple[str, float]] = []

    event_counts: defaultdict[str, int] = defaultdict(int)

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

        item_probability = (item_clicks + SMOOTHING * global_prior) / (item_impressions + SMOOTHING)

        user_probability = (user_clicks + SMOOTHING * global_prior) / (user_impressions + SMOOTHING)

        combined_probability = item_probability + user_weight * (user_probability - global_prior)

        combined_probability = min(
            1.0,
            max(0.0, combined_probability),
        )

        truth.append((record.rowid, record.clicked))

        item_predictions.append((record.rowid, item_probability))

        user_predictions.append((record.rowid, combined_probability))

        event_counts[record.event_id] += 1

    expected_candidates = contract["events"]["expected_candidate_count"]

    for event_id in selected_event_ids:
        if event_counts[event_id] != expected_candidates:
            raise ValueError(f"Unexpected candidate count for {event_id}")

    item_auc = score_by_rowid(
        truth,
        item_predictions,
    )

    user_auc = score_by_rowid(
        truth,
        user_predictions,
    )

    return UserBaselineResult(
        fitting_rows=fitting_rows,
        validation_events=len(selected_event_ids),
        validation_rows=len(truth),
        known_user_events=known_user_events,
        item_auc=item_auc,
        user_auc=user_auc,
    )


def main() -> None:
    """Run the controlled B0/B1 comparison."""

    parser = argparse.ArgumentParser(description="Compare item CTR and user-propensity CTR")

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
        raise ValueError(f"Unknown validation window: {args.window}")

    result = evaluate_user_baseline(
        args.train,
        args.contract,
        windows[args.window],
    )

    coverage = result.known_user_events / result.validation_events

    print()
    print("=" * 55)
    print("CONTROLLED BASELINE COMPARISON")
    print("=" * 55)

    print(f"Window: {args.window}")
    print(f"Fitting rows: {result.fitting_rows:,}")
    print(f"Validation events: {result.validation_events:,}")
    print(f"Validation rows: {result.validation_rows:,}")
    print(f"Known-user coverage: {coverage:.2%}")

    print()
    print(f"B0 Item CTR AUC:      {result.item_auc:.6f}")
    print(f"B1 User + Item AUC:   {result.user_auc:.6f}")
    print(f"AUC difference:       {result.user_auc - result.item_auc:+.6f}")

    print("=" * 55)


if __name__ == "__main__":
    main()
