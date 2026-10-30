"""Input record and event validation for Huawei TechArena CTR data."""

# Import standard libraries
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CandidateRecord:
    """A validated candidate impression."""

    rowid: str
    event_id: str
    user_id: str
    item_id: str
    timestamp: int
    time_bucket: int
    day_type: int
    history_item_ids: tuple[str, ...]
    history_timestamps: tuple[int, ...]
    clicked: int | None


def load_input_contract(path: Path) -> dict[str, Any]:
    """Load the project's input contract."""
    with path.open("r", encoding="utf-8") as handle:
        contract = json.load(handle)

    # Validation
    if not isinstance(contract, dict):
        raise TypeError("Input contract must be a JSON object.")

    return contract


def validate_header(
    columns: Sequence[str],
    contract: dict[str, Any],
    *,
    partition: str,
) -> None:
    """Validate candidate input column names."""

    if partition not in contract["files"]:
        raise ValueError(f"Unknown partition: {partition}")

    required = set(contract["files"][partition]["required_columns"])

    if len(columns) != len(set(columns)):
        raise ValueError("Duplicate column names.")

    actual = set(columns)

    if actual != required:
        missing = sorted(required - actual)
        unexpected = sorted(actual - required)

        raise ValueError(f"Invalid columns: missing = {missing}, unexpected = {unexpected}")


def _parse_history(
    raw_items: str,
    raw_timestamps: str,
    event_timestamp: int,
    maximum_length: int,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Parse and validate historical click sequences."""

    try:
        items = json.loads(raw_items)
        timestamps = json.loads(raw_timestamps)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Malformed history JSON") from exc

    # Validation
    if not isinstance(items, list) or not isinstance(timestamps, list):
        raise TypeError("Histories must be JSON arrays")

    if len(items) != len(timestamps):
        raise ValueError("History item/timestamp length mismatch")

    if len(items) > maximum_length:
        raise ValueError("History exceeds maximum length")

    if any(not isinstance(item, str) or not item for item in items):
        raise ValueError("History item IDs must be non-empty strings")

    if any(
        isinstance(timestamp, bool) or not isinstance(timestamp, int) for timestamp in timestamps
    ):
        raise ValueError("History timestamps must be integers")

    if any(timestamp >= event_timestamp for timestamp in timestamps):
        raise ValueError("History contains non-prior timestamps")

    if timestamps != sorted(timestamps):
        raise ValueError("History timestamps are not chronological")

    return tuple(items), tuple(timestamps)


def validate_candidate(
    row: Mapping[str, str],
    contract: dict[str, Any],
    *,
    partition: str,
) -> CandidateRecord:
    """Validate and parse one candidate impression."""

    validate_header(list(row), contract, partition=partition)

    for field in contract["identifiers"]["columns"]:
        if not isinstance(row[field], str) or not row[field]:
            raise ValueError(f"Invalid identifier: {field}")

    try:
        timestamp = int(row["timestamp"])
        time_bucket = int(row["time_bucket"])
        day_type = int(row["day_type"])
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid numeric event field") from exc

    if timestamp < 0:
        raise ValueError("Timestamp must be non-negative")

    try:
        event_time = datetime.fromtimestamp(timestamp, tz=UTC)
    except (OverflowError, OSError, ValueError) as exc:
        raise ValueError("Timestamp is outside the supported range") from exc

    if time_bucket not in contract["time_bucket"]["allowed_values"]:
        raise ValueError("Invalid time bucket")

    if day_type not in contract["day_type"]["allowed_values"]:
        raise ValueError("Invalid day type")

    if time_bucket != event_time.hour // 6:
        raise ValueError("Time bucket disagrees with UTC timestamp")

    expected_day_type = int(event_time.weekday() >= 5)

    if day_type != expected_day_type:
        raise ValueError("Day type disagrees with UTC timestamp")

    items, timestamps = _parse_history(
        row["history_item_ids"],
        row["history_timestamps"],
        timestamp,
        contract["history"]["maximum_length"],
    )

    clicked = None

    if contract["files"][partition]["has_labels"]:
        if row["clicked"] not in {"0", "1"}:
            raise ValueError("Invalid click label")

        clicked = int(row["clicked"])

    return CandidateRecord(
        rowid=row["rowid"],
        event_id=row["event_id"],
        user_id=row["user_id"],
        item_id=row["item_id"],
        timestamp=timestamp,
        time_bucket=time_bucket,
        day_type=day_type,
        history_item_ids=items,
        history_timestamps=timestamps,
        clicked=clicked,
    )


def validate_event(
    rows: Sequence[Mapping[str, str]],
    contract: dict[str, Any],
    *,
    partition: str,
) -> tuple[CandidateRecord, ...]:
    """Validate one complete recommendation event."""

    expected_count = contract["events"]["expected_candidate_count"]

    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} candidates, received {len(rows)}")

    candidates = tuple(validate_candidate(row, contract, partition=partition) for row in rows)

    reference = candidates[0]

    rowids: set[str] = set()

    for candidate in candidates:
        if candidate.rowid in rowids:
            raise ValueError("Duplicate rowid within event")

        rowids.add(candidate.rowid)

        if candidate.event_id != reference.event_id:
            raise ValueError("Inconsistent event IDs")

        if candidate.user_id != reference.user_id:
            raise ValueError("Inconsistent event user")

        if candidate.timestamp != reference.timestamp:
            raise ValueError("Inconsistent event timestamp")

        if candidate.time_bucket != reference.time_bucket:
            raise ValueError("Inconsistent event time bucket")

        if candidate.day_type != reference.day_type:
            raise ValueError("Inconsistent event day type")

        if candidate.history_item_ids != reference.history_item_ids:
            raise ValueError("Inconsistent event history items")

        if candidate.history_timestamps != reference.history_timestamps:
            raise ValueError("Inconsistent event history timestamps")

    if contract["files"][partition]["has_labels"]:
        click_count = sum(candidate.clicked == 1 for candidate in candidates)

        maximum_clicks = contract["events"]["maximum_clicks_per_event"]

        if click_count > maximum_clicks:
            raise ValueError("Event exceeds maximum click count")

    return candidates
