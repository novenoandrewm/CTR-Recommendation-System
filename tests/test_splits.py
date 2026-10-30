"""Regression tests for temporal validation splitting."""

# Import standard libraries
from pathlib import Path

import pytest

# Import data splitting utilities
from ctr_recommendation.data.splits import (
    EventRef,
    ValidationWindow,
    load_validation_windows,
    split_events,
)


def test_validation_configuration_loads() -> None:
    """All configured validation windows must have valid boundaries."""
    root = Path(__file__).resolve().parents[1]

    windows = load_validation_windows(root / "configs" / "validation.yaml")

    assert "development" in windows
    assert "confirmation" in windows
    assert "weekend_robustness" in windows
    assert "longer_horizon" in windows


def test_first_event_selection_and_boundaries() -> None:
    """Select the earliest eligible event and respect cutoff boundaries."""
    window = ValidationWindow(
        fit_before=100,
        validate_from=100,
        validate_before=200,
    )

    events = [
        EventRef("train_a", "user_a", 90),
        EventRef("later_a", "user_a", 150),
        EventRef("first_a", "user_a", 120),
        EventRef("first_b", "user_b", 100),
        EventRef("excluded_c", "user_c", 200),
    ]

    result = split_events(events, window)

    assert result.train_event_ids == frozenset({"train_a"})

    assert result.validation_event_ids == frozenset({"first_a", "first_b"})

    assert result.tied_users == frozenset()


def test_timestamp_ties_are_deterministic() -> None:
    """Equal-time events must produce a stable selection."""
    window = ValidationWindow(100, 100, 200)

    events = [
        EventRef("event_z", "user_a", 120),
        EventRef("event_a", "user_a", 120),
    ]

    first = split_events(events, window)
    second = split_events(reversed(events), window)

    assert first.validation_event_ids == frozenset({"event_a"})
    assert first.validation_event_ids == second.validation_event_ids

    assert first.tied_users == frozenset({"user_a"})


def test_duplicate_events_are_rejected() -> None:
    """The event-level input must not contain duplicate event IDs."""
    window = ValidationWindow(100, 100, 200)

    events = [
        EventRef("event_1", "user_a", 120),
        EventRef("event_1", "user_a", 120),
    ]

    with pytest.raises(ValueError, match="Duplicate event ID"):
        split_events(events, window)


def test_invalid_window_is_rejected() -> None:
    """Validation must not begin before the fitting cutoff."""
    window = ValidationWindow(
        fit_before=150,
        validate_from=100,
        validate_before=200,
    )

    with pytest.raises(ValueError, match="fit_before"):
        split_events([], window)
