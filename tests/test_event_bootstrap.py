"""Tests for weighted global AUC and paired event bootstrapping."""

import numpy as np
import pytest

from ctr_recommendation.analysis.event_bootstrap import (
    WeightedGlobalAUC,
    build_bootstrap_inputs,
    paired_event_bootstrap,
)
from ctr_recommendation.analysis.history_failure import (
    ScoredCandidate,
    ScoredEvent,
)
from ctr_recommendation.evaluation.metrics import score_by_rowid


def make_candidate(
    rowid: str,
    clicked: int,
    b1: float,
    b2: float,
) -> ScoredCandidate:
    return ScoredCandidate(
        rowid=rowid,
        clicked=clicked,
        b1_score=b1,
        b2_score=b2,
        matched=False,
    )


def make_events() -> dict[str, ScoredEvent]:
    """Construct two clicked events and one no-click event."""

    return {
        "event_0": ScoredEvent(
            user_id="user_0",
            timestamp=100,
            history_ids=(),
            candidates=[
                make_candidate("row_0", 1, 0.10, 0.15),
                make_candidate("row_1", 0, 0.12, 0.12),
            ],
        ),
        "event_1": ScoredEvent(
            user_id="user_1",
            timestamp=200,
            history_ids=(),
            candidates=[
                make_candidate("row_2", 1, 0.70, 0.70),
                make_candidate("row_3", 0, 0.30, 0.30),
            ],
        ),
        "event_2": ScoredEvent(
            user_id="user_2",
            timestamp=300,
            history_ids=(),
            candidates=[
                make_candidate("row_4", 0, 0.50, 0.50),
                make_candidate("row_5", 0, 0.60, 0.60),
            ],
        ),
    }


def test_weighted_auc_matches_unweighted_reference() -> None:
    """Unit event weights must reproduce the established metric."""

    events = make_events()
    inputs = build_bootstrap_inputs(events)

    weights = np.ones(3, dtype=np.int64)

    actual = inputs.b1_auc.score(weights)

    truth = []
    predictions = []

    for event in events.values():
        for row in event.candidates:
            truth.append((row.rowid, row.clicked))
            predictions.append((row.rowid, row.b1_score))

    expected = score_by_rowid(
        truth,
        predictions,
    )

    assert actual == pytest.approx(
        expected,
        abs=1e-12,
    )


def test_weighted_auc_matches_explicit_event_duplication() -> None:
    """Event weights must equal physically duplicated event rows."""

    events = make_events()
    inputs = build_bootstrap_inputs(events)

    multiplicities = np.array(
        [3, 2, 1],
        dtype=np.int64,
    )

    actual = inputs.b1_auc.score(multiplicities)

    truth = []
    predictions = []

    unique_id = 0

    for event_index, event_id in enumerate(sorted(events)):
        event = events[event_id]

        for _ in range(int(multiplicities[event_index])):
            for row in event.candidates:
                rowid = f"duplicate_{unique_id}"
                unique_id += 1

                truth.append((rowid, row.clicked))
                predictions.append((rowid, row.b1_score))

    expected = score_by_rowid(
        truth,
        predictions,
    )

    assert actual == pytest.approx(
        expected,
        abs=1e-12,
    )


def test_weighted_auc_handles_ties() -> None:
    """Equal-score positive/negative pairs receive half credit."""

    auc = WeightedGlobalAUC(
        event_indices=[0, 0, 1, 1],
        labels=[1, 0, 1, 0],
        scores=[0.2, 0.2, 0.5, 0.5],
    )

    result = auc.score(np.array([1, 1], dtype=np.int64))

    assert result == pytest.approx(0.5)


def test_zero_weight_events_are_excluded() -> None:
    """An event with multiplicity zero contributes no observations."""

    auc = WeightedGlobalAUC(
        event_indices=[0, 0, 1, 1],
        labels=[1, 0, 1, 0],
        scores=[0.9, 0.1, 0.2, 0.8],
    )

    result = auc.score(np.array([1, 0], dtype=np.int64))

    assert result == pytest.approx(1.0)


def test_invalid_multiplicities_rejected() -> None:
    """Negative bootstrap multiplicities are invalid."""

    auc = WeightedGlobalAUC(
        event_indices=[0, 0],
        labels=[1, 0],
        scores=[0.8, 0.2],
    )

    with pytest.raises(
        ValueError,
        match="Negative event multiplicity",
    ):
        auc.score(np.array([-1], dtype=np.int64))


def test_one_class_sample_rejected() -> None:
    """Global AUC is undefined if no positive candidates remain."""

    auc = WeightedGlobalAUC(
        event_indices=[0, 1],
        labels=[1, 0],
        scores=[0.8, 0.2],
    )

    with pytest.raises(
        ValueError,
        match="positive and negative",
    ):
        auc.score(np.array([0, 1], dtype=np.int64))


def test_bootstrap_is_deterministic() -> None:
    """The fixed random seed must reproduce the same intervals."""

    inputs = build_bootstrap_inputs(make_events())

    first = paired_event_bootstrap(
        inputs,
        replicates=50,
        seed=2026,
    )

    second = paired_event_bootstrap(
        inputs,
        replicates=50,
        seed=2026,
    )

    assert first == second


def test_bootstrap_preserves_observed_auc() -> None:
    """Bootstrapping must not alter the original point estimate."""

    inputs = build_bootstrap_inputs(make_events())

    result = paired_event_bootstrap(
        inputs,
        replicates=50,
        seed=2026,
    )

    weights = np.ones(
        inputs.event_count,
        dtype=np.int64,
    )

    expected_b1 = inputs.b1_auc.score(weights)
    expected_b2 = inputs.b2_auc.score(weights)

    assert result.b1_global == pytest.approx(expected_b1)

    assert result.b2_global == pytest.approx(expected_b2)

    assert result.global_difference == pytest.approx(expected_b2 - expected_b1)

    assert result.clicked_events == 2


def test_empty_events_rejected() -> None:
    """A bootstrap requires actual recommendation events."""

    with pytest.raises(
        ValueError,
        match="No events",
    ):
        build_bootstrap_inputs({})
