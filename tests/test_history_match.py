"""Integration tests for historical item-matching experiments."""

# Import standard libraries
import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

# Import necessary classes
from ctr_recommendation.data.splits import ValidationWindow
from ctr_recommendation.training.history_match import (
    evaluate_history_match,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts" / "input_schema.json"

CUTOFF = int(datetime(2025, 1, 23, tzinfo=UTC).timestamp())


def make_event(
    event_id: str,
    user_id: str,
    timestamp: int,
    clicked_index: int,
    *,
    history: tuple[str, ...] = (),
) -> list[dict[str, str]]:
    """Construct a valid event with optional previous-click history."""

    event_time = datetime.fromtimestamp(timestamp, tz=UTC)

    history_times = [timestamp - 3600 for _ in history]

    return [
        {
            "rowid": f"{event_id}_row_{index}",
            "event_id": event_id,
            "user_id": user_id,
            "item_id": f"item_{index}",
            "timestamp": str(timestamp),
            "time_bucket": str(event_time.hour // 6),
            "day_type": str(int(event_time.weekday() >= 5)),
            "history_item_ids": json.dumps(history),
            "history_timestamps": json.dumps(history_times),
            "clicked": str(int(index == clicked_index)),
        }
        for index in range(10)
    ]


def write_csv(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    """Write synthetic training observations."""

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


def make_window() -> ValidationWindow:
    """Return the synthetic development window."""

    return ValidationWindow(
        fit_before=CUTOFF,
        validate_from=CUTOFF,
        validate_before=CUTOFF + 3600,
    )


def make_rows(
    *,
    include_history_match: bool,
) -> list[dict[str, str]]:
    """Create known-user events with one clicked validation candidate."""

    history = ("item_1",) if include_history_match else ()

    return (
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
            history=history,
        )
        + make_event(
            "validation_b",
            "user_b",
            CUTOFF + 120,
            -1,
        )
    )


def test_history_matching_improves_synthetic_ranking(
    tmp_path: Path,
) -> None:
    """Verify exact controls and candidate-specific history effects."""

    path = tmp_path / "train.csv"

    write_csv(
        path,
        make_rows(include_history_match=True),
    )

    result = evaluate_history_match(
        path,
        CONTRACT,
        make_window(),
    )

    assert result.fitting_rows == 20
    assert result.validation_events == 2
    assert result.validation_rows == 20

    assert result.clicked_events == 1

    assert result.matched_candidates == 1
    assert result.matched_clicks == 1

    # The reconstructed controls must match the known fixture.
    assert result.b0_auc == pytest.approx(8.5 / 19)
    assert result.b1_auc == pytest.approx(13 / 19)

    # Historical matching raises the clicked candidate above
    # every negative candidate in this deliberately simple fixture.
    assert result.b2_auc == pytest.approx(1.0)

    # B2 must also improve ranking within the clicked event.
    assert result.b1_event_auc == pytest.approx(4 / 9)
    assert result.b2_event_auc == pytest.approx(1.0)


def test_no_history_match_reproduces_b1(
    tmp_path: Path,
) -> None:
    """Without history matches, B2 must equal B1 exactly."""

    path = tmp_path / "train.csv"

    write_csv(
        path,
        make_rows(include_history_match=False),
    )

    result = evaluate_history_match(
        path,
        CONTRACT,
        make_window(),
    )

    assert result.matched_candidates == 0
    assert result.matched_clicks == 0

    assert result.b2_auc == pytest.approx(result.b1_auc)

    assert result.b2_event_auc == pytest.approx(result.b1_event_auc)


def test_future_history_is_rejected(
    tmp_path: Path,
) -> None:
    """Interaction histories must not contain future observations."""

    rows = make_rows(include_history_match=True)

    # Corrupt a historical timestamp so it equals the target event.
    for row in rows:
        if row["event_id"] == "validation_a":
            row["history_timestamps"] = json.dumps([CUTOFF + 60])

    path = tmp_path / "train.csv"

    write_csv(path, rows)

    with pytest.raises(ValueError):
        evaluate_history_match(
            path,
            CONTRACT,
            make_window(),
        )
