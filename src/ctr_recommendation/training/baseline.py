"""Streaming item-popularity baseline for temporal CTR validation."""

# Import standard libraries
import argparse
import csv
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Import necassary data classes
from ctr_recommendation.data.splits import (
    ValidationWindow,
    load_validation_windows,
    validate_window,
)
from ctr_recommendation.data.validate import (
    CandidateRecord,
    load_input_contract,
    validate_candidate,
    validate_header,
)
from ctr_recommendation.evaluation.metrics import score_by_rowid

SMOOTHING = 50.0


@dataclass(frozen=True)
class BaselineResult:
    """Results from one temporal validation experiment."""

    fitting_rows: int
    validation_events: int
    validation_rows: int
    global_auc: float


def read_candidates(
    path: Path,
    contract: dict[str, Any],
) -> Iterator[CandidateRecord]:
    """Read and validate candidate records sequentially."""

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle, strict=True)

        if reader.fieldnames is None:
            raise ValueError("Training CSV has no header")

        validate_header(
            reader.fieldnames,
            contract,
            partition="train",
        )

        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed CSV near line {reader.line_num}")

            yield validate_candidate(
                row,
                contract,
                partition="train",
            )


def evaluate_item_baseline(
    csv_path: Path,
    contract_path: Path,
    window: ValidationWindow,
) -> BaselineResult:
    """Evaluate smoothed item CTR using a temporal holdout.

    Pass 1:
        Fit item counts using pre-cutoff observations.
        Select the first validation event per user.

    Pass 2:
        Score only the selected validation events.

    No validation labels contribute to the fitted statistics.
    """

    validate_window(window)

    contract = load_input_contract(contract_path)

    # Each value contains [impressions, clicks].
    item_counts: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])

    # Map each validation user to (timestamp, event_id).
    selected_by_user: dict[str, tuple[int, str]] = {}

    fitting_rows = 0
    fitting_clicks = 0
    previous_timestamp: int | None = None

    print("[BASELINE] Pass 1: fitting item statistics...", flush=True)

    for record in read_candidates(csv_path, contract):
        if previous_timestamp is not None and record.timestamp < previous_timestamp:
            raise ValueError("Training CSV is not chronological")

        previous_timestamp = record.timestamp

        if record.timestamp >= window.validate_before:
            break

        if record.timestamp < window.fit_before:
            if record.clicked is None:
                raise ValueError("Missing training label")

            counts = item_counts[record.item_id]

            counts[0] += 1
            counts[1] += record.clicked

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
        raise ValueError("No fitting observations found")

    if not selected_by_user:
        raise ValueError("No validation events found")

    global_prior = fitting_clicks / fitting_rows

    selected_event_ids = {event_id for _, event_id in selected_by_user.values()}

    print(
        f"[BASELINE] Fitting rows: {fitting_rows:,}",
        flush=True,
    )

    print(
        f"[BASELINE] Selected validation events: {len(selected_event_ids):,}",
        flush=True,
    )

    print(
        "[BASELINE] Pass 2: generating validation predictions...",
        flush=True,
    )

    truth: list[tuple[str, int]] = []
    predictions: list[tuple[str, float]] = []

    event_counts: defaultdict[str, int] = defaultdict(int)

    for record in read_candidates(csv_path, contract):
        if record.timestamp >= window.validate_before:
            break

        if record.event_id not in selected_event_ids:
            continue

        if record.clicked is None:
            raise ValueError("Missing validation label")

        impressions, clicks = item_counts.get(
            record.item_id,
            [0, 0],
        )

        probability = (clicks + SMOOTHING * global_prior) / (impressions + SMOOTHING)

        truth.append((record.rowid, record.clicked))

        predictions.append((record.rowid, probability))

        event_counts[record.event_id] += 1

    expected_candidates = contract["events"]["expected_candidate_count"]

    for event_id in selected_event_ids:
        count = event_counts[event_id]

        if count != expected_candidates:
            raise ValueError(
                f"Validation event {event_id} has "
                f"{count} candidates; expected "
                f"{expected_candidates}"
            )

    auc = score_by_rowid(
        truth,
        predictions,
    )

    return BaselineResult(
        fitting_rows=fitting_rows,
        validation_events=len(selected_event_ids),
        validation_rows=len(predictions),
        global_auc=auc,
    )


def main() -> None:
    """Execute the item-popularity reference experiment."""

    parser = argparse.ArgumentParser(description="Run the CTR item-popularity baseline")

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

    result = evaluate_item_baseline(
        args.train,
        args.contract,
        windows[args.window],
    )

    print()
    print("=" * 50)
    print("CTR BASELINE RESULTS")
    print("=" * 50)

    print(f"Validation window: {args.window}")
    print(f"Fitting rows: {result.fitting_rows:,}")
    print(f"Validation events: {result.validation_events:,}")
    print(f"Validation rows: {result.validation_rows:,}")
    print(f"Global ROC AUC: {result.global_auc:.6f}")

    print("=" * 50)


if __name__ == "__main__":
    main()
