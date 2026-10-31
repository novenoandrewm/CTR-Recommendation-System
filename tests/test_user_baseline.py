"""Integration tests for user-propensity CTR experiments."""

# Import standard libraries
import csv
from datetime import UTC, datetime
from pathlib import Path

import pytest

# Import necessary classes
from ctr_recommendation.data.splits import ValidationWindow
from ctr_recommendation.training.user_baseline import (
    evaluate_user_baseline,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts" / "input_schema.json"

CUTOFF = int(datetime(2025, 1, 23, tzinfo=UTC).timestamp())


def make_event(
    event_id: str,
    user_id: str,
    timestamp: int,
    clicked_index: int,
) -> list[dict[str, str]]:
    """Create an event; use clicked_index=-1 for no click."""

    event_time = datetime.fromtimestamp(timestamp, tz=UTC)

    return [
        {
            "rowid": f"{event_id}_row_{index}",
            "event_id": event_id,
            "user_id": user_id,
            "item_id": f"item_{index}",
            "timestamp": str(timestamp),
            "time_bucket": str(event_time.hour // 6),
            "day_type": str(int(event_time.weekday() >= 5)),
            "history_item_ids": "[]",
            "history_timestamps": "[]",
            "clicked": str(int(index == clicked_index)),
        }
        for index in range(10)
    ]


def test_user_propensity_changes_global_ranking(
    tmp_path: Path,
) -> None:
    """Check exact AUCs for known users with different click rates."""

    rows = (
        make_event(
            "fit_a",
            "user_a",
            CUTOFF - 120,
            0,
        )
        + make_event(
            "fit_b",
            "user_b",
            CUTOFF - 60,
            -1,
        )
        + make_event(
            "validation_a",
            "user_a",
            CUTOFF + 60,
            1,
        )
        + make_event(
            "validation_b",
            "user_b",
            CUTOFF + 120,
            -1,
        )
    )

    path = tmp_path / "train.csv"

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

    window = ValidationWindow(
        fit_before=CUTOFF,
        validate_from=CUTOFF,
        validate_before=CUTOFF + 3600,
    )

    result = evaluate_user_baseline(
        path,
        CONTRACT,
        window,
    )

    assert result.fitting_rows == 20
    assert result.validation_events == 2
    assert result.validation_rows == 20
    assert result.known_user_events == 2

    # Both baselines are evaluated on the same 20 rows.
    # There is one clicked candidate and 19 negatives.
    assert result.item_auc == pytest.approx(8.5 / 19)
    assert result.user_auc == pytest.approx(13 / 19)

    # The added user signal improves this controlled example.
    assert result.user_auc > result.item_auc

    # Removing the user adjustment must reproduce B0 exactly.
    zero_weight = evaluate_user_baseline(
        path,
        CONTRACT,
        window,
        user_weight=0.0,
    )

    assert zero_weight.user_auc == pytest.approx(zero_weight.item_auc)
