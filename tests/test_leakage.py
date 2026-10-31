"""Regression tests for temporal fitting-boundary safeguards."""

# Import standard libraries
from dataclasses import replace

import pytest

from ctr_recommendation.data.splits import (
    EventRef,
    ValidationWindow,
    split_events,
)
from ctr_recommendation.data.validate import CandidateRecord
from ctr_recommendation.training.leakage import (
    FitBoundary,
    ItemCounts,
    LeakageError,
    fit_item_counts,
)


def make_record(
    event_id: str,
    timestamp: int,
    *,
    clicked: int | None = 0,
    item_id: str = "item_a",
) -> CandidateRecord:
    """Construct a synthetic candidate for testing fitting boundaries."""

    return CandidateRecord(
        rowid=f"row_{event_id}",
        event_id=event_id,
        user_id="user_a",
        item_id=item_id,
        timestamp=timestamp,
        time_bucket=0,
        day_type=0,
        history_item_ids=(),
        history_timestamps=(),
        clicked=clicked,
    )


def make_boundary() -> FitBoundary:
    """Create a fitting boundary with two training and one held-out event."""

    window = ValidationWindow(
        fit_before=100,
        validate_from=100,
        validate_before=200,
    )

    events = [
        EventRef("train_a", "user_a", 90),
        EventRef("train_b", "user_a", 99),
        EventRef("validation_a", "user_a", 120),
    ]

    split = split_events(events, window)

    return FitBoundary.from_split(window, split)


def test_permitted_training_labels() -> None:
    """Only authorised pre-cutoff labels contribute to item statistics."""

    boundary = make_boundary()

    records = [
        make_record("train_a", 90, clicked=1),
        make_record("train_b", 99, clicked=0),
    ]

    counts = fit_item_counts(records, boundary=boundary)

    assert counts["item_a"] == ItemCounts(
        impressions=2,
        clicks=1,
    )


def test_held_out_event_rejected() -> None:
    """Validation labels must never enter fitted statistics."""

    boundary = make_boundary()

    record = make_record(
        "validation_a",
        120,
        clicked=1,
    )

    with pytest.raises(LeakageError, match="Held-out event"):
        fit_item_counts([record], boundary=boundary)


def test_future_timestamp_rejected() -> None:
    """A fitting event ID cannot override the temporal cutoff."""

    boundary = make_boundary()

    record = make_record(
        "train_a",
        100,
        clicked=1,
    )

    with pytest.raises(LeakageError, match="timestamp"):
        fit_item_counts([record], boundary=boundary)


def test_unknown_event_rejected() -> None:
    """Unknown event IDs cannot enter the fitting population."""

    boundary = make_boundary()

    record = make_record(
        "unknown_event",
        90,
        clicked=1,
    )

    with pytest.raises(LeakageError, match="not in the fitting split"):
        fit_item_counts([record], boundary=boundary)


def test_missing_label_rejected() -> None:
    """Unlabelled records must never contribute target statistics."""

    boundary = make_boundary()

    record = make_record(
        "train_a",
        90,
        clicked=None,
    )

    with pytest.raises(LeakageError, match="Missing or invalid"):
        fit_item_counts([record], boundary=boundary)


def test_zero_click_label_allowed() -> None:
    """Negative impressions must be counted as valid observations."""

    boundary = make_boundary()

    record = make_record(
        "train_a",
        90,
        clicked=0,
    )

    counts = fit_item_counts([record], boundary=boundary)

    assert counts["item_a"] == ItemCounts(
        impressions=1,
        clicks=0,
    )


def test_boundary_overlap_rejected() -> None:
    """Training and held-out event identities must not overlap."""

    boundary = make_boundary()

    invalid = replace(
        boundary,
        held_out_event_ids=frozenset({"train_a", "validation_a"}),
    )

    assert invalid.fit_event_ids & invalid.held_out_event_ids

    # Construct a new boundary through the public validation pathway.
    from ctr_recommendation.data.splits import SplitResult

    split = SplitResult(
        train_event_ids=invalid.fit_event_ids,
        validation_event_ids=invalid.held_out_event_ids,
        tied_users=frozenset(),
    )

    window = ValidationWindow(100, 100, 200)

    with pytest.raises(LeakageError, match="overlap"):
        FitBoundary.from_split(window, split)
