"""Strict fitting boundaries for leaking-safe CTR experiments."""

# Import standard libraries
from collections.abc import Iterable
from dataclasses import dataclass

# Import necessary module from data
from ctr_recommendation.data.splits import SplitResult, ValidationWindow
from ctr_recommendation.data.validate import CandidateRecord


# Error message class
class LeakageError(ValueError):
    """Raised when an observation violates the fitting boundary."""


@dataclass(frozen=True)
class ItemCounts:
    """Observed impressions and clicks for one item."""

    impressions: int
    clicks: int


@dataclass(frozen=True)
class FitBoundary:
    """Define which events and labels may enter model fitting."""

    fit_before: int
    fit_event_ids: frozenset[str]
    held_out_event_ids: frozenset[str]

    @classmethod
    def from_split(
        cls,
        window: ValidationWindow,
        split: SplitResult,
    ) -> "FitBoundary":
        """Construct a fitting boundary from a validated temporal split."""

        # Validation
        if window.fit_before != window.validate_from:
            raise LeakageError("Invalid temporal fitting boundary.")

        overlap = split.train_event_ids & split.validation_event_ids

        if overlap:
            raise LeakageError("Training and validation events overlap.")

        return cls(
            fit_before=window.fit_before,
            fit_event_ids=split.train_event_ids,
            held_out_event_ids=split.validation_event_ids,
        )

    def require_fit_label(self, record: CandidateRecord) -> int:
        """Return a label only when its observation is permitted for fitting."""

        if record.event_id in self.held_out_event_ids:
            raise LeakageError(f"Held-out event entered fitting: {record.event_id}")

        if record.event_id not in self.fit_event_ids:
            raise LeakageError(f"Event is not in the fitting split: {record.event_id}")

        if record.timestamp >= self.fit_before:
            raise LeakageError(f"Event timestamp violates fitting cutoff: {record.event_id}")

        if type(record.clicked) is not int or record.clicked not in (0, 1):
            raise LeakageError(f"Missing or invalid fitting label: {record.rowid}")

        return record.clicked


def fit_item_counts(
    records: Iterable[CandidateRecord],
    *,
    boundary: FitBoundary,
) -> dict[str, ItemCounts]:
    """Count item impressions and clicks using permitted fitting rows only.

    The resulting statistics are suitable for held-out inference.
    They must not be used directly as in-sample training features.
    """

    counts: dict[str, ItemCounts] = {}

    for record in records:
        label = boundary.require_fit_label(record)

        previous = counts.get(
            record.item_id,
            ItemCounts(impressions=0, clicks=0),
        )

        counts[record.item_id] = ItemCounts(
            impressions=previous.impressions + 1,
            clicks=previous.clicks + label,
        )

    return counts
