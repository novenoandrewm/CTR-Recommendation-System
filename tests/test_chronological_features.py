"""Tests for leakage-safe chronological item features."""

from dataclasses import replace

import pytest

# Import necessary classes
from ctr_recommendation.data.validate import CandidateRecord
from ctr_recommendation.features.chronological import (
    chronological_item_counts,
)
from ctr_recommendation.training.leakage import (
    FitBoundary,
    ItemCounts,
    LeakageError,
)


def make_boundary() -> FitBoundary:
    """Create a synthetic fitting boundary."""

    return FitBoundary(
        fit_before=100,
        fit_event_ids=frozenset(
            {
                "train_a",
                "train_b",
                "train_c",
                "train_d",
            }
        ),
        held_out_event_ids=frozenset({"validation_a"}),
    )


def make_record(
    event_id: str,
    timestamp: int,
    clicked: int | None,
    *,
    item_id: str = "item_x",
) -> CandidateRecord:
    """Create a synthetic candidate record."""

    return CandidateRecord(
        rowid=f"row_{event_id}",
        event_id=event_id,
        user_id=f"user_{event_id}",
        item_id=item_id,
        timestamp=timestamp,
        time_bucket=0,
        day_type=0,
        history_item_ids=(),
        history_timestamps=(),
        clicked=clicked,
    )


def encode(
    records: list[CandidateRecord],
) -> list[tuple[str, ItemCounts]]:
    """Execute the chronological encoder completely."""

    return list(
        chronological_item_counts(
            records,
            boundary=make_boundary(),
        )
    )


def test_earlier_observations_contribute_to_later_features() -> None:
    """Counts must include earlier observations only."""

    records = [
        make_record("train_a", 10, 1),
        make_record("train_b", 20, 0),
        make_record("train_c", 30, 1),
    ]

    result = encode(records)

    assert result == [
        ("row_train_a", ItemCounts(impressions=0, clicks=0)),
        ("row_train_b", ItemCounts(impressions=1, clicks=1)),
        ("row_train_c", ItemCounts(impressions=2, clicks=1)),
    ]


def test_same_timestamp_events_cannot_influence_each_other() -> None:
    """Equal-time events must not exchange target information."""

    records = [
        make_record("train_a", 10, 1),
        make_record("train_b", 10, 0),
        make_record("train_c", 20, 0),
    ]

    result = encode(records)

    assert result == [
        ("row_train_a", ItemCounts(impressions=0, clicks=0)),
        ("row_train_b", ItemCounts(impressions=0, clicks=0)),
        ("row_train_c", ItemCounts(impressions=2, clicks=1)),
    ]


def test_current_label_cannot_change_its_own_features() -> None:
    """Changing a target must not change its own training features."""

    records = [
        make_record("train_a", 10, 0),
        make_record("train_b", 10, 1),
        make_record("train_c", 20, 0),
    ]

    original = encode(records)

    changed_records = [
        replace(records[0], clicked=1),
        records[1],
        records[2],
    ]

    changed = encode(changed_records)

    # Current and same-time features must remain unchanged.
    assert original[:2] == changed[:2]

    # Later observations may legitimately see the earlier click.
    assert original[2] != changed[2]


def test_item_statistics_remain_separate() -> None:
    """One item's clicks must not enter another item's counts."""

    records = [
        make_record("train_a", 10, 1, item_id="item_x"),
        make_record("train_b", 20, 0, item_id="item_y"),
        make_record("train_c", 30, 0, item_id="item_x"),
    ]

    result = encode(records)

    assert result == [
        ("row_train_a", ItemCounts(impressions=0, clicks=0)),
        ("row_train_b", ItemCounts(impressions=0, clicks=0)),
        ("row_train_c", ItemCounts(impressions=1, clicks=1)),
    ]


def test_unsorted_records_are_rejected() -> None:
    """The encoder must not silently accept decreasing timestamps."""

    records = [
        make_record("train_a", 20, 1),
        make_record("train_b", 10, 0),
    ]

    with pytest.raises(LeakageError, match="nondecreasing"):
        encode(records)


def test_held_out_labels_are_rejected() -> None:
    """Validation labels cannot contribute to training features."""

    records = [
        make_record("train_a", 10, 1),
        make_record("validation_a", 20, 1),
    ]

    with pytest.raises(LeakageError, match="Held-out event"):
        encode(records)


def test_future_observations_are_rejected() -> None:
    """Records at the fitting cutoff are not eligible."""

    records = [
        make_record("train_a", 100, 1),
    ]

    with pytest.raises(LeakageError, match="cutoff"):
        encode(records)


def test_missing_labels_are_rejected() -> None:
    """Unlabelled observations cannot update training statistics."""

    records = [
        make_record("train_a", 10, None),
    ]

    with pytest.raises(LeakageError, match="Missing or invalid"):
        encode(records)


def test_unknown_events_are_rejected() -> None:
    """Unexpected event IDs must not enter the feature population."""

    records = [
        make_record("unexpected_event", 10, 1),
    ]

    with pytest.raises(LeakageError, match="not in the fitting split"):
        encode(records)
