"""Leakage-safe chronological item statistics for training features."""

from _collections_abc import Iterable, Iterator

# Import necessary classes
from ctr_recommendation.data.validate import CandidateRecord
from ctr_recommendation.training.leakage import (
    FitBoundary,
    ItemCounts,
    LeakageError,
)

ZERO_COUNTS = ItemCounts(impressions=0, clicks=0)


def chronological_item_counts(
    records: Iterable[CandidateRecord],
    *,
    boundary: FitBoundary,
) -> Iterator[tuple[str, ItemCounts]]:
    """Generate item-count features using strictly earlier timestamps.

    Input requirements:
        - Records must be ordered by nondecreasing timestamp.
        - Every record must satisfy the supplied fitting boundary.
        - Input records must have unique row IDs, verified upstream.

    Each output contains:
        (rowid, counts_from_strictly_earlier_timestamps)

    Records sharing a timestamp cannot influence one another.

    The function processes records incrementally without retaining
    the complete candidate dataset in memory."""

    committed: dict[str, ItemCounts] = {}
    pending: dict[str, ItemCounts] = {}
    previous_timestamp: int | None = None

    for record in records:
        # Verify membership, timestamp cutoff, and label validity.
        label = boundary.require_fit_label(record)

        if previous_timestamp is not None and record.timestamp < previous_timestamp:
            raise LeakageError("Records must be sorted by nondecreasing timestamp")

        # A new timestamp means the previous timestamp group
        # is complete and may now contribute to future features.
        if previous_timestamp is not None and record.timestamp > previous_timestamp:
            for item_id, new_counts in pending.items():
                previous = committed.get(item_id, ZERO_COUNTS)

                committed[item_id] = ItemCounts(
                    impressions=(previous.impressions + new_counts.impressions),
                    clicks=previous.clicks + new_counts.clicks,
                )

            pending.clear()

        previous_timestamp = record.timestamp

        # Only committed, earlier-timestamp observations
        # are visible to the current candidate.
        historical_counts = committed.get(
            record.item_id,
            ZERO_COUNTS,
        )

        # Buffer the current observation.
        # Its label must not influence any event at this timestamp.
        current_pending = pending.get(
            record.item_id,
            ZERO_COUNTS,
        )

        pending[record.item_id] = ItemCounts(
            impressions=current_pending.impressions + 1,
            clicks=current_pending.clicks + label,
        )

        # Preserve the candidate's original identity.
        yield record.rowid, historical_counts
