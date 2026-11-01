"""Regression tests for B2A failure-analysis diagnostics."""

import pytest

from ctr_recommendation.analysis.history_failure import (
    ScoredCandidate,
    ScoredEvent,
    event_ranking,
    history_group,
    pair_credit,
)


def make_candidate(
    rowid: str,
    clicked: int,
    b1: float,
    b2: float,
    matched: bool = False,
) -> ScoredCandidate:
    return ScoredCandidate(
        rowid=rowid,
        clicked=clicked,
        b1_score=b1,
        b2_score=b2,
        matched=matched,
    )


def test_pair_credit() -> None:
    assert pair_credit(0.8, 0.2) == 1.0
    assert pair_credit(0.5, 0.5) == 0.5
    assert pair_credit(0.2, 0.8) == 0.0


def test_history_groups() -> None:
    assert history_group(0) == "0"
    assert history_group(1) == "1-5"
    assert history_group(5) == "1-5"
    assert history_group(6) == "6-10"
    assert history_group(10) == "6-10"
    assert history_group(11) == "11-20"
    assert history_group(20) == "11-20"
    assert history_group(21) == "21+"


def test_history_match_can_improve_ranking() -> None:
    event = ScoredEvent(
        user_id="user_a",
        timestamp=1000,
        history_ids=("item_1",),
        candidates=[
            make_candidate("clicked", 1, 0.10, 0.15, True),
            make_candidate("negative", 0, 0.12, 0.12),
        ],
    )

    assert event_ranking(event) == (0.0, 1.0)


def test_history_match_can_worsen_ranking() -> None:
    event = ScoredEvent(
        user_id="user_a",
        timestamp=1000,
        history_ids=("item_1",),
        candidates=[
            make_candidate("clicked", 1, 0.12, 0.12),
            make_candidate("negative", 0, 0.10, 0.15, True),
        ],
    )

    assert event_ranking(event) == (1.0, 0.0)


def test_zero_match_event_has_identical_ranking() -> None:
    event = ScoredEvent(
        user_id="user_a",
        timestamp=1000,
        history_ids=(),
        candidates=[
            make_candidate("clicked", 1, 0.10, 0.10),
            make_candidate("negative", 0, 0.20, 0.20),
        ],
    )

    b1_auc, b2_auc = event_ranking(event)

    assert b1_auc == b2_auc


def test_no_click_event_has_no_within_event_auc() -> None:
    event = ScoredEvent(
        user_id="user_a",
        timestamp=1000,
        history_ids=(),
        candidates=[
            make_candidate("negative_1", 0, 0.10, 0.10),
            make_candidate("negative_2", 0, 0.20, 0.20),
        ],
    )

    assert event_ranking(event) is None


def test_multiple_clicks_rejected() -> None:
    event = ScoredEvent(
        user_id="user_a",
        timestamp=1000,
        history_ids=(),
        candidates=[
            make_candidate("clicked_1", 1, 0.10, 0.10),
            make_candidate("clicked_2", 1, 0.20, 0.20),
        ],
    )

    with pytest.raises(ValueError, match="Multiple clicks"):
        event_ranking(event)
