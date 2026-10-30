"""Tests for candidate and event data validation."""

# Import standard libraries
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ctr_recommendation.data.validate import (
    load_input_contract,
    validate_candidate,
    validate_event,
    validate_header,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = load_input_contract(ROOT / "contracts" / "input_schema.json")

TIMESTAMP = int(datetime(2025, 1, 23, tzinfo=UTC).timestamp())


def make_event(*, training: bool = True) -> list[dict[str, str]]:
    """Create a valid synthetic event with ten candidates."""

    rows = []

    for index in range(10):
        row = {
            "rowid": f"row_{index}",
            "event_id": "event_001",
            "user_id": "user_001",
            "item_id": f"item_{index}",
            "timestamp": str(TIMESTAMP),
            "time_bucket": "0",
            "day_type": "0",
            "history_item_ids": json.dumps(["item_a", "item_a"]),
            "history_timestamps": json.dumps([TIMESTAMP - 120, TIMESTAMP - 120]),
        }

        if training:
            row["clicked"] = "1" if index == 0 else "0"

        rows.append(row)

    return rows


def test_valid_training_event() -> None:
    """A valid training event must pass."""

    result = validate_event(
        make_event(),
        CONTRACT,
        partition="train",
    )

    assert len(result) == 10
    assert sum(candidate.clicked == 1 for candidate in result) == 1


def test_valid_evaluation_event() -> None:
    """Evaluation events must not require labels."""

    result = validate_event(
        make_event(training=False),
        CONTRACT,
        partition="evaluation",
    )

    assert len(result) == 10
    assert all(candidate.clicked is None for candidate in result)


def test_empty_history_is_allowed() -> None:
    """Users without previous clicks are valid."""

    row = make_event()[0]

    row["history_item_ids"] = "[]"
    row["history_timestamps"] = "[]"

    result = validate_candidate(row, CONTRACT, partition="train")

    assert result.history_item_ids == ()


def test_repeated_history_items_are_allowed() -> None:
    """Repeated clicks and equal historical timestamps are valid."""

    result = validate_candidate(
        make_event()[0],
        CONTRACT,
        partition="train",
    )

    assert result.history_item_ids == ("item_a", "item_a")


def test_history_length_mismatch_rejected() -> None:
    row = make_event()[0]
    row["history_timestamps"] = "[]"

    with pytest.raises(ValueError, match="length mismatch"):
        validate_candidate(row, CONTRACT, partition="train")


def test_future_history_rejected() -> None:
    row = make_event()[0]

    row["history_timestamps"] = json.dumps([TIMESTAMP - 120, TIMESTAMP])

    with pytest.raises(ValueError, match="non-prior"):
        validate_candidate(row, CONTRACT, partition="train")


def test_history_order_rejected() -> None:
    row = make_event()[0]

    row["history_timestamps"] = json.dumps([TIMESTAMP - 60, TIMESTAMP - 120])

    with pytest.raises(ValueError, match="chronological"):
        validate_candidate(row, CONTRACT, partition="train")


def test_multiple_clicks_rejected() -> None:
    rows = make_event()

    rows[1]["clicked"] = "1"

    with pytest.raises(ValueError, match="maximum click"):
        validate_event(rows, CONTRACT, partition="train")


def test_inconsistent_event_user_rejected() -> None:
    rows = make_event()

    rows[1]["user_id"] = "different_user"

    with pytest.raises(ValueError, match="event user"):
        validate_event(rows, CONTRACT, partition="train")


def test_duplicate_rowid_rejected() -> None:
    rows = make_event()

    rows[1]["rowid"] = rows[0]["rowid"]

    with pytest.raises(ValueError, match="Duplicate rowid"):
        validate_event(rows, CONTRACT, partition="train")


def test_incomplete_event_rejected() -> None:
    rows = make_event()[:9]

    with pytest.raises(ValueError, match="Expected 10"):
        validate_event(rows, CONTRACT, partition="train")


def test_invalid_label_rejected() -> None:
    row = make_event()[0]

    row["clicked"] = "2"

    with pytest.raises(ValueError, match="Invalid click label"):
        validate_candidate(row, CONTRACT, partition="train")


def test_evaluation_label_column_rejected() -> None:
    row = make_event()[0]

    with pytest.raises(ValueError, match="Invalid columns"):
        validate_candidate(row, CONTRACT, partition="evaluation")


def test_duplicate_header_rejected() -> None:
    columns = list(make_event()[0])
    columns.append("rowid")

    with pytest.raises(ValueError, match="Duplicate column"):
        validate_header(columns, CONTRACT, partition="train")
