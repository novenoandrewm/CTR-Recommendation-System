"""Tests for complete CSV integrity auditing."""

# Import standard libraries
import csv
import json
from pathlib import Path

import pytest

from ctr_recommendation.data.audit_csv import audit_csv

CONTRACT = Path(__file__).resolve().parents[1] / "contracts" / "input_schema.json"


def make_event(
    event_id: str,
    user_id: str,
    *,
    training: bool = True,
) -> list[dict[str, str]]:
    """Build a synthetic event containing ten candidates."""

    rows = []

    for index in range(10):
        row = {
            "rowid": f"{event_id}_row_{index}",
            "event_id": event_id,
            "user_id": user_id,
            "item_id": f"item_{index}",
            "timestamp": "1737590400",
            "time_bucket": "0",
            "day_type": "0",
            "history_item_ids": "[]",
            "history_timestamps": "[]",
        }

        if training:
            row["clicked"] = "1" if index == 0 else "0"

        rows.append(row)

    return rows


def write_csv(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    """Write synthetic candidate records."""

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
        )

        writer.writeheader()
        writer.writerows(rows)


def test_valid_interleaved_events(tmp_path: Path) -> None:
    """Events need not occupy consecutive CSV rows."""

    first = make_event("event_a", "user_a")
    second = make_event("event_b", "user_b")

    interleaved = []

    for row_a, row_b in zip(first, second, strict=True):
        interleaved.extend([row_a, row_b])

    path = tmp_path / "train.csv"
    write_csv(path, interleaved)

    result = audit_csv(path, CONTRACT, partition="train")

    assert result.candidate_rows == 20
    assert result.events == 2
    assert result.clicks == 2


def test_duplicate_global_rowid_rejected(tmp_path: Path) -> None:
    """A row ID may not appear in different events."""

    first = make_event("event_a", "user_a")
    second = make_event("event_b", "user_b")

    second[0]["rowid"] = first[0]["rowid"]

    path = tmp_path / "train.csv"
    write_csv(path, first + second)

    with pytest.raises(ValueError, match="Duplicate rowid"):
        audit_csv(path, CONTRACT, partition="train")


def test_incomplete_event_rejected(tmp_path: Path) -> None:
    """Every event must contain exactly ten candidates."""

    path = tmp_path / "train.csv"
    write_csv(path, make_event("event_a", "user_a")[:9])

    with pytest.raises(ValueError, match="Invalid event"):
        audit_csv(path, CONTRACT, partition="train")


def test_inconsistent_event_history_rejected(tmp_path: Path) -> None:
    """All candidates must share the same historical context."""

    rows = make_event("event_a", "user_a")

    rows[5]["history_item_ids"] = json.dumps(["previous_item"])
    rows[5]["history_timestamps"] = json.dumps([1737590000])

    path = tmp_path / "train.csv"
    write_csv(path, rows)

    with pytest.raises(ValueError, match="Inconsistent context"):
        audit_csv(path, CONTRACT, partition="train")


def test_evaluation_repeated_user_rejected(tmp_path: Path) -> None:
    """Official evaluation contains only one event per user."""

    first = make_event("event_a", "user_a", training=False)
    second = make_event("event_b", "user_a", training=False)

    path = tmp_path / "evaluation.csv"
    write_csv(path, first + second)

    with pytest.raises(ValueError, match="multiple events"):
        audit_csv(path, CONTRACT, partition="evaluation")


def test_valid_evaluation_file(tmp_path: Path) -> None:
    """Evaluation should contain no click labels."""

    rows = make_event("event_a", "user_a", training=False)

    path = tmp_path / "evaluation.csv"
    write_csv(path, rows)

    result = audit_csv(path, CONTRACT, partition="evaluation")

    assert result.candidate_rows == 10
    assert result.events == 1
    assert result.clicks is None
