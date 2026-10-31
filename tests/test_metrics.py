"""Regression tests for global ROC AUC and row-ID alignment."""

# Import standard libraries
from math import inf, nan
from random import Random

import pytest

# Import necessary classes
from ctr_recommendation.evaluation.metrics import (
    global_roc_auc,
    score_by_rowid,
)


def test_perfect_auc() -> None:
    """Correct ranking must produce AUC 1.0."""

    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]

    assert global_roc_auc(labels, scores) == pytest.approx(1.0)


def test_reversed_auc() -> None:
    """Completely reversed ranking must produce AUC 0.0."""

    labels = [0, 0, 1, 1]
    scores = [0.9, 0.8, 0.2, 0.1]

    assert global_roc_auc(labels, scores) == pytest.approx(0.0)


def test_constant_predictions() -> None:
    """Equal predictions must receive half credit."""

    labels = [0, 1, 0, 1]
    scores = [0.5, 0.5, 0.5, 0.5]

    assert global_roc_auc(labels, scores) == pytest.approx(0.5)


def test_global_auc_includes_cross_event_comparisons() -> None:
    """Global AUC includes comparisons across different events."""

    labels = [1, 0, 1, 0]
    scores = [0.9, 0.8, 0.4, 0.3]

    assert global_roc_auc(labels, scores) == pytest.approx(0.75)


def test_prediction_order_does_not_matter() -> None:
    """Predictions must align by rowid rather than input order."""

    truth = [
        ("row_a", 1),
        ("row_b", 0),
        ("row_c", 0),
        ("row_d", 1),
    ]

    predictions = [
        ("row_c", 0.1),
        ("row_a", 0.9),
        ("row_d", 0.8),
        ("row_b", 0.2),
    ]

    assert score_by_rowid(truth, predictions) == pytest.approx(1.0)


def test_missing_prediction_rejected() -> None:
    """Missing prediction IDs must be rejected."""

    truth = [
        ("row_a", 1),
        ("row_b", 0),
    ]

    predictions = [
        ("row_a", 0.9),
    ]

    with pytest.raises(ValueError, match="Missing predictions"):
        score_by_rowid(truth, predictions)


def test_unknown_prediction_rejected() -> None:
    """Predictions for unknown rows must be rejected."""

    truth = [
        ("row_a", 1),
        ("row_b", 0),
    ]

    predictions = [
        ("row_a", 0.9),
        ("row_b", 0.1),
        ("row_unknown", 0.5),
    ]

    with pytest.raises(ValueError, match="Unknown prediction rowid"):
        score_by_rowid(truth, predictions)


def test_duplicate_prediction_rejected() -> None:
    """Repeated prediction IDs must be rejected."""

    truth = [
        ("row_a", 1),
        ("row_b", 0),
    ]

    predictions = [
        ("row_a", 0.9),
        ("row_a", 0.8),
        ("row_b", 0.1),
    ]

    with pytest.raises(ValueError, match="Duplicate prediction rowid"):
        score_by_rowid(truth, predictions)


def test_duplicate_truth_rejected() -> None:
    """Repeated truth IDs must be rejected."""

    truth = [
        ("row_a", 1),
        ("row_a", 0),
    ]

    predictions = [
        ("row_a", 0.9),
    ]

    with pytest.raises(ValueError, match="Duplicate truth rowid"):
        score_by_rowid(truth, predictions)


def test_single_class_rejected() -> None:
    """AUC is undefined when only one class is present."""

    with pytest.raises(ValueError, match="both positive and negative"):
        global_roc_auc([0, 0, 0], [0.1, 0.2, 0.3])


@pytest.mark.parametrize("invalid_score", [nan, inf, -inf])
def test_nonfinite_scores_rejected(invalid_score: float) -> None:
    """NaN and infinite predictions must be rejected."""

    with pytest.raises(ValueError, match="finite"):
        global_roc_auc([0, 1], [0.1, invalid_score])


def test_boolean_labels_rejected() -> None:
    """Python booleans must not silently become click labels."""

    with pytest.raises(ValueError, match="binary integers"):
        global_roc_auc([False, True], [0.1, 0.9])


def test_auc_matches_pairwise_reference() -> None:
    """Compare global AUC against an independent exhaustive calculation."""

    rng = Random(1729)

    # Test 100 reproducible datasets.
    for _ in range(100):
        # Guarantee that both classes are represented.
        labels = [0, 1] + [rng.randrange(2) for _ in range(38)]

        # Generate scores from a small discrete range.
        # This intentionally creates many tied scores.
        # Negative scores are also included.
        scores = [rng.randrange(-4, 5) / 4 for _ in labels]

        # Extract positive and negative prediction scores.
        positives = [score for label, score in zip(labels, scores, strict=True) if label == 1]

        negatives = [score for label, score in zip(labels, scores, strict=True) if label == 0]

        # Calculate ROC AUC using exhaustive pairwise comparisons.
        #
        # Positive score > negative score: 1 point
        # Positive score = negative score: 0.5 points
        # Positive score < negative score: 0 points

        concordant = sum(
            (1.0 if positive > negative else 0.5 if positive == negative else 0.0)
            for positive in positives
            for negative in negatives
        )

        # Reference ROC AUC.
        reference_auc = concordant / (len(positives) * len(negatives))

        # ROC AUC produced by our implementation.
        actual_auc = global_roc_auc(labels, scores)

        # Both independent calculations must agree.
        assert actual_auc == pytest.approx(reference_auc)
