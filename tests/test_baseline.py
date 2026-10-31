"""Integration tests for the streaming item-popularity baseline."""

# Import standard libraries
import csv
from datetime import UTC, datetime
from pathlib import Path

import pytest

# Import data classes
from ctr_recommendation.data.splits import ValidationWindow
from ctr_recommendation.training.baseline import (
    evaluate_item_baseline,
)

ROOT = Path(__file__).resolve().parents[1]

CONTRACT = ROOT / "contracts" / "input_schema.json"

CUTOFF = int(
    datetime(
        2025,
        1,
        23,
        tzinfo=UTC,
    ).timestamp()
)


def make_event(
    event_id: str,
    user_id: str,
    timestamp: int,
    clicked_index: int,
) -> list[dict[str, str]]:
    """Construct one valid synthetic recommendation event."""

    # Derive temporal metadata from the actual UTC timestamp.
    # Never hardcode time_bucket or day_type in test fixtures.
    event_time = datetime.fromtimestamp(timestamp, tz=UTC)

    time_bucket = event_time.hour // 6
    day_type = int(event_time.weekday() >= 5)

    rows = []

    for index in range(10):
        rows.append(
            {
                "rowid": f"{event_id}_row_{index}",
                "event_id": event_id,
                "user_id": user_id,
                "item_id": f"item_{index}",
                "timestamp": str(timestamp),
                "time_bucket": str(time_bucket),
                "day_type": str(day_type),
                "history_item_ids": "[]",
                "history_timestamps": "[]",
                "clicked": str(int(index == clicked_index)),
            }
        )

    return rows


def write_training_file(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    """Write synthetic candidate observations."""

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
        )

        writer.writeheader()
        writer.writerows(rows)


def test_baseline_selects_first_validation_event(
    tmp_path: Path,
) -> None:
    """Verify event selection and exact baseline ROC AUC."""

    rows = (
        make_event(
            "train_a",
            "user_a",
            CUTOFF - 60,
            0,
        )
        + make_event(
            "validation_first",
            "user_a",
            CUTOFF + 60,
            0,
        )
        + make_event(
            "validation_later",
            "user_a",
            CUTOFF + 120,
            1,
        )
        + make_event(
            "validation_other",
            "user_b",
            CUTOFF + 180,
            1,
        )
    )

    path = tmp_path / "train.csv"

    write_training_file(path, rows)

    window = ValidationWindow(
        fit_before=CUTOFF,
        validate_from=CUTOFF,
        validate_before=CUTOFF + 3600,
    )

    result = evaluate_item_baseline(
        path,
        CONTRACT,
        window,
    )

    # Only the fitting event contributes to item statistics.
    assert result.fitting_rows == 10

    # Only the first validation event for each user is selected.
    assert result.validation_events == 2

    # Two selected events with 10 candidates each.
    assert result.validation_rows == 20

    # Verify the exact expected global ROC AUC.
    #
    # The fitting event clicks item_0.
    # Therefore, item_0 receives the highest prediction.
    #
    # Validation positives:
    #   - item_0: high score
    #   - item_1: low score
    #
    # Validation negatives:
    #   - 1 candidate with a high score
    #   - 17 candidates with low scores
    #
    # Pairwise ROC AUC:
    #   - High-score positive: 17 wins + 0.5 tie
    #   - Low-score positive: 8.5 ties
    #
    # Total: 26 / 36 = 13 / 18
    assert result.global_auc == pytest.approx(13 / 18)


def test_baseline_rejects_unsorted_input(
    tmp_path: Path,
) -> None:
    """Chronological ordering must be enforced."""

    rows = make_event(
        "validation_first",
        "user_a",
        CUTOFF + 60,
        0,
    ) + make_event(
        "train_a",
        "user_a",
        CUTOFF - 60,
        0,
    )

    path = tmp_path / "train.csv"

    write_training_file(path, rows)

    window = ValidationWindow(
        fit_before=CUTOFF,
        validate_from=CUTOFF,
        validate_before=CUTOFF + 3600,
    )

    with pytest.raises(
        ValueError,
        match="not chronological",
    ):
        evaluate_item_baseline(
            path,
            CONTRACT,
            window,
        )
