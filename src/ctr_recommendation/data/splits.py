"""Deterministic, event-level temporal validation splitting."""

# Import standard libraries
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import yaml


@dataclass(frozen=True)
class EventRef:
    """Metadata for one unique recommendation event."""

    event_id: str
    user_id: str
    timestamp: int


@dataclass(frozen=True)
class ValidationWindow:
    """Temporal boundaries expressed as UTC Unix seconds."""

    fit_before: int
    validate_from: int
    validate_before: int


@dataclass(frozen=True)
class SplitResult:
    """Event identities and diagnostics for one temporal split."""

    train_event_ids: frozenset[str]
    validation_event_ids: frozenset[str]
    tied_users: frozenset[str]


def parse_utc_timestamp(value: str) -> int:
    """Parse an explicit UTC timestamp into Unix seconds."""
    parsed = datetime.fromisoformat(value)

    if parsed.utcoffset() != timedelta(0):
        raise ValueError(f"Timestamp must specify UTC: {value}")

    return int(parsed.timestamp())


def load_validation_windows(path: Path) -> dict[str, ValidationWindow]:
    """Load and validate temporal windows from the YAML configuration."""
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if config["task"]["timezone"] != "UTC":
        raise ValueError("Validation timezone must be UTC")

    if config["validation"]["selection"] != "first_event_per_user":
        raise ValueError("Unsupported validation selection policy")

    windows = {}

    for name, spec in config["windows"].items():
        window = ValidationWindow(
            fit_before=parse_utc_timestamp(spec["fit_before"]),
            validate_from=parse_utc_timestamp(spec["validate_from"]),
            validate_before=parse_utc_timestamp(spec["validate_before"]),
        )

        validate_window(window)
        windows[name] = window

    return windows


def validate_window(window: ValidationWindow) -> None:
    """Reject invalid temporal boundaries."""
    if not (
        window.fit_before == window.validate_from and window.validate_from < window.validate_before
    ):
        raise ValueError("Expected fit_before == validate_from < validate_before")


def split_events(
    events: Iterable[EventRef],
    window: ValidationWindow,
) -> SplitResult:
    """Create a temporal split using the first eligible event per user.

    The input must contain exactly one EventRef per distinct event.
    Candidate-row grouping and integrity checks belong to ingestion.
    """
    validate_window(window)

    seen_events: set[str] = set()
    train_ids: set[str] = set()
    selected: dict[str, EventRef] = {}
    tied_users: set[str] = set()

    for event in events:
        if not event.event_id or not event.user_id:
            raise ValueError("Event and user IDs must be non-empty")

        if isinstance(event.timestamp, bool) or not isinstance(event.timestamp, int):
            raise TypeError("Event timestamp must be an integer")

        if event.event_id in seen_events:
            raise ValueError(f"Duplicate event ID: {event.event_id}")

        seen_events.add(event.event_id)

        if event.timestamp < window.fit_before:
            train_ids.add(event.event_id)
            continue

        if not (window.validate_from <= event.timestamp < window.validate_before):
            continue

        previous = selected.get(event.user_id)

        if previous is None or event.timestamp < previous.timestamp:
            selected[event.user_id] = event
            tied_users.discard(event.user_id)

        elif event.timestamp == previous.timestamp:
            tied_users.add(event.user_id)

            # Deterministic tie-breaker, not inferred chronology.
            if event.event_id < previous.event_id:
                selected[event.user_id] = event

    return SplitResult(
        train_event_ids=frozenset(train_ids),
        validation_event_ids=frozenset(event.event_id for event in selected.values()),
        tied_users=frozenset(tied_users),
    )
