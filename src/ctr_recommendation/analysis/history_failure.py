"""Failure diagnostics for the fixed B2A history-matching model."""

# Import standard libraries
import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Import necessary libraries
from ctr_recommendation.data.validate import load_input_contract
from ctr_recommendation.evaluation.metrics import score_by_rowid
from ctr_recommendation.evaluation.window_policy import (
    load_development_window,
)
from ctr_recommendation.training.baseline import (
    SMOOTHING,
    read_candidates,
)
from ctr_recommendation.training.history_match import HISTORY_BONUS
from ctr_recommendation.training.user_baseline import USER_WEIGHT


@dataclass(frozen=True)
class ScoredCandidate:
    """A candidate with its observed label and model predictions."""

    rowid: str
    clicked: int
    b1_score: float
    b2_score: float
    matched: bool


@dataclass
class ScoredEvent:
    """All candidates belonging to one recommendation event."""

    user_id: str
    timestamp: int
    history_ids: tuple[str, ...]
    candidates: list[ScoredCandidate] = field(default_factory=list)

    @property
    def history_length(self) -> int:
        return len(self.history_ids)

    @property
    def has_match(self) -> bool:
        return any(candidate.matched for candidate in self.candidates)

    @property
    def has_click(self) -> bool:
        return any(candidate.clicked == 1 for candidate in self.candidates)


def pair_credit(positive: float, negative: float) -> float:
    """Award one point for a win and half a point for a tie."""

    if positive > negative:
        return 1.0

    if positive == negative:
        return 0.5

    return 0.0


def event_ranking(
    event: ScoredEvent,
) -> tuple[float, float] | None:
    """Return B1/B2 within-event AUC for a clicked event."""

    positives = [row for row in event.candidates if row.clicked == 1]

    if len(positives) > 1:
        raise ValueError("Multiple clicks in one event")

    if not positives:
        return None

    positive = positives[0]

    negatives = [row for row in event.candidates if row.clicked == 0]

    if not negatives:
        raise ValueError("Clicked event has no negative candidates")

    b1_credit = sum(pair_credit(positive.b1_score, negative.b1_score) for negative in negatives)

    b2_credit = sum(pair_credit(positive.b2_score, negative.b2_score) for negative in negatives)

    return (
        b1_credit / len(negatives),
        b2_credit / len(negatives),
    )


def history_group(history_length: int) -> str:
    """Assign an event to a predetermined history-length group."""

    if history_length == 0:
        return "0"

    if history_length <= 5:
        return "1-5"

    if history_length <= 10:
        return "6-10"

    if history_length <= 20:
        return "11-20"

    return "21+"


def score_validation_events(
    train_path: Path,
    contract_path: Path,
    window_name: str,
    config_path: Path,
) -> dict[str, ScoredEvent]:
    """Reconstruct B1/B2 scores using a permitted development window."""

    # This must happen before reading any training data.
    window = load_development_window(
        config_path,
        window_name,
    )

    contract = load_input_contract(contract_path)

    item_counts: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])

    user_counts: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])

    selected_by_user: dict[str, tuple[int, str]] = {}

    fitting_rows = 0
    fitting_clicks = 0
    previous_timestamp: int | None = None

    print("[R2] Pass 1: fitting B1 statistics...", flush=True)

    for record in read_candidates(train_path, contract):
        if previous_timestamp is not None and record.timestamp < previous_timestamp:
            raise ValueError("Training CSV is not chronological")

        previous_timestamp = record.timestamp

        if record.timestamp >= window.validate_before:
            break

        if record.timestamp < window.fit_before:
            if record.clicked not in (0, 1):
                raise ValueError("Invalid fitting label")

            item_counts[record.item_id][0] += 1
            item_counts[record.item_id][1] += record.clicked

            user_counts[record.user_id][0] += 1
            user_counts[record.user_id][1] += record.clicked

            fitting_rows += 1
            fitting_clicks += record.clicked

        elif record.timestamp >= window.validate_from:
            candidate_event = (
                record.timestamp,
                record.event_id,
            )

            previous = selected_by_user.get(record.user_id)

            if previous is None or candidate_event < previous:
                selected_by_user[record.user_id] = candidate_event

    if fitting_rows == 0:
        raise ValueError("No fitting rows found")

    if not selected_by_user:
        raise ValueError("No validation events found")

    global_prior = fitting_clicks / fitting_rows

    selected_event_ids = {event_id for _, event_id in selected_by_user.values()}

    print(
        f"[R2] Selected {len(selected_event_ids):,} events",
        flush=True,
    )

    print("[R2] Pass 2: collecting predictions...", flush=True)

    events: dict[str, ScoredEvent] = {}

    for record in read_candidates(train_path, contract):
        if record.timestamp >= window.validate_before:
            break

        if record.event_id not in selected_event_ids:
            continue

        if record.timestamp < window.validate_from:
            raise ValueError("Selected event outside validation window")

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

        b1_score = item_probability + USER_WEIGHT * (user_probability - global_prior)

        b1_score = min(1.0, max(0.0, b1_score))

        matched = record.item_id in record.history_item_ids

        b2_score = min(
            1.0,
            max(
                0.0,
                b1_score + HISTORY_BONUS * int(matched),
            ),
        )

        history_ids = tuple(record.history_item_ids)

        if record.event_id not in events:
            events[record.event_id] = ScoredEvent(
                user_id=record.user_id,
                timestamp=record.timestamp,
                history_ids=history_ids,
            )

        event = events[record.event_id]

        # Protect against silently inconsistent event metadata.
        if (
            event.user_id != record.user_id
            or event.timestamp != record.timestamp
            or event.history_ids != history_ids
        ):
            raise ValueError(f"Inconsistent event metadata: {record.event_id}")

        event.candidates.append(
            ScoredCandidate(
                rowid=record.rowid,
                clicked=record.clicked,
                b1_score=b1_score,
                b2_score=b2_score,
                matched=matched,
            )
        )

    expected_candidates = contract["events"]["expected_candidate_count"]

    for event_id in selected_event_ids:
        if event_id not in events:
            raise ValueError(f"Missing validation event: {event_id}")

        event = events[event_id]

        if len(event.candidates) != expected_candidates:
            raise ValueError(f"Incorrect candidate count for {event_id}")

        click_count = sum(candidate.clicked for candidate in event.candidates)

        if click_count > 1:
            raise ValueError(f"Multiple clicks in event {event_id}")

    print(
        f"[R2] Fitting rows: {fitting_rows:,}",
        flush=True,
    )

    return events


def calculate_auc(
    events: list[ScoredEvent],
) -> tuple[float | None, float | None]:
    """Compute global AUC on a complete set of events."""

    candidates = [candidate for event in events for candidate in event.candidates]

    positives = sum(row.clicked for row in candidates)

    if positives == 0 or positives == len(candidates):
        return None, None

    truth = [(row.rowid, row.clicked) for row in candidates]

    b1_predictions = [(row.rowid, row.b1_score) for row in candidates]

    b2_predictions = [(row.rowid, row.b2_score) for row in candidates]

    return (
        score_by_rowid(truth, b1_predictions),
        score_by_rowid(truth, b2_predictions),
    )


def describe_group(
    name: str,
    events: list[ScoredEvent],
) -> None:
    """Report failures and benefits within a complete-event group."""

    if not events:
        print(f"\n{name}: no events")
        return

    candidates = [row for event in events for row in event.candidates]

    matched = [row for row in candidates if row.matched]

    clicked_events = sum(event.has_click for event in events)

    matched_clicks = sum(row.clicked for row in matched)

    false_positive_matches = len(matched) - matched_clicks

    improved = 0
    worsened = 0
    unchanged = 0

    b1_event_scores: list[float] = []
    b2_event_scores: list[float] = []

    for event in events:
        ranking = event_ranking(event)

        if ranking is None:
            continue

        b1_rank, b2_rank = ranking

        b1_event_scores.append(b1_rank)
        b2_event_scores.append(b2_rank)

        difference = b2_rank - b1_rank

        if difference > 1e-12:
            improved += 1
        elif difference < -1e-12:
            worsened += 1
        else:
            unchanged += 1

    b1_auc, b2_auc = calculate_auc(events)

    print()
    print("-" * 60)
    print(name)
    print("-" * 60)

    print(f"Events: {len(events):,}")
    print(f"Clicked events: {clicked_events:,}")
    print(f"No-click events: {len(events) - clicked_events:,}")

    print(f"Candidates: {len(candidates):,}")
    print(f"Matched candidates: {len(matched):,}")
    print(f"Matched clicks: {matched_clicks:,}")
    print(f"Matched nonclicks: {false_positive_matches:,}")

    print(f"Events with improved ranking: {improved:,}")
    print(f"Events with worsened ranking: {worsened:,}")
    print(f"Events with unchanged ranking: {unchanged:,}")

    if b1_auc is not None and b2_auc is not None:
        print(f"B1 global AUC: {b1_auc:.6f}")
        print(f"B2 global AUC: {b2_auc:.6f}")
        print(f"Global AUC difference: {b2_auc - b1_auc:+.6f}")
    else:
        print("Global AUC: undefined (only one label class)")

    if b1_event_scores:
        b1_event_auc = sum(b1_event_scores) / len(b1_event_scores)
        b2_event_auc = sum(b2_event_scores) / len(b2_event_scores)

        print(f"B1 within-event AUC: {b1_event_auc:.6f}")
        print(f"B2 within-event AUC: {b2_event_auc:.6f}")
        print(f"Within-event difference: {b2_event_auc - b1_event_auc:+.6f}")
    else:
        print("Within-event AUC: undefined (no clicked events)")


def main() -> None:
    """Run event-level B2A failure diagnostics."""

    parser = argparse.ArgumentParser(description="Audit historical item-matching failures")

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

    events_by_id = score_validation_events(
        args.train,
        args.contract,
        args.window,
        args.validation_config,
    )

    events = list(events_by_id.values())

    print()
    print("=" * 60)
    print("B2A-R2: HISTORICAL FEATURE FAILURE ANALYSIS")
    print("=" * 60)

    print(f"Window: {args.window}")

    describe_group("ALL VALIDATION EVENTS", events)

    print()
    print("=" * 60)
    print("HISTORICAL MATCH COVERAGE")
    print("=" * 60)

    describe_group(
        "EVENTS WITH NO HISTORICAL MATCH",
        [event for event in events if not event.has_match],
    )

    describe_group(
        "EVENTS WITH AT LEAST ONE HISTORICAL MATCH",
        [event for event in events if event.has_match],
    )

    print()
    print("=" * 60)
    print("HISTORY LENGTH SUBGROUPS")
    print("=" * 60)

    for group in ("0", "1-5", "6-10", "11-20", "21+"):
        group_events = [event for event in events if history_group(event.history_length) == group]

        describe_group(
            f"HISTORY LENGTH: {group}",
            group_events,
        )


if __name__ == "__main__":
    main()
