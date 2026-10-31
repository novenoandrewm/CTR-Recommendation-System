"""Global ROC AUC and row-ID-aligned evaluation utilities."""

# Import standard libraries
from collections.abc import Iterable
from math import isfinite


# Global ROC AUC func
def global_roc_auc(
    labels: Iterable[int],
    scores: Iterable[float],
) -> float:
    """Compute global binary ROC AUC with correct tie handling.

    Scores may be any finite real-valued model scores.
    They do not need to be calibrated probabilities."""

    y = list(labels)
    predictions = list(scores)

    # Validation
    if not y:
        raise ValueError("Cannot calculate AUC on an empty dataset.")

    if len(y) != len(predictions):
        raise ValueError("Labels and predictions have different lengths.")

    for label in y:
        # Validation
        if type(label) is not int or label not in (0, 1):
            raise ValueError("Labels must be binary integers")

    for score in predictions:
        # Validation
        if type(score) not in (int, float):
            raise TypeError("Prediction scores must be numeric")

        if not isfinite(score):
            raise ValueError("Prediction scores must be finite")

    positives = sum(y)
    negatives = len(y) - positives

    if positives == 0 or negatives == 0:
        # Validation
        raise ValueError("ROC AUC requires both positive and negative labels")

    # Sort by score, ascending.
    pairs = sorted(zip(predictions, y, strict=True))

    negatives_below = 0
    concordant_pairs = 0.0

    index = 0

    while index < len(pairs):
        current_score = pairs[index][0]

        group_positives = 0
        group_negatives = 0

        # Collect all observations with the same prediction score.
        while index < len(pairs) and pairs[index][0] == current_score:
            label = pairs[index][1]

            if label == 1:
                group_positives += 1
            else:
                group_negatives += 1

            index += 1

        # Positives beat negatives with lower scores.
        # Equal-score pairs receive half credit.
        concordant_pairs += group_positives * (negatives_below + 0.5 * group_negatives)

        negatives_below += group_negatives

    return concordant_pairs / (positives * negatives)


# Score by rowid func
def score_by_rowid(
    truth: Iterable[tuple[str, int]],
    predictions: Iterable[tuple[str, float]],
) -> float:
    """Calculate AUC after aligning predictions to labels by row ID.

    Reject missing, unknown, or duplicated identifiers."""

    truth_by_id: dict[str, int] = {}

    for rowid, label in truth:
        # Validation
        if not isinstance(rowid, str) or not rowid:
            raise ValueError("Invalid truth rowid")

        if rowid in truth_by_id:
            raise ValueError(f"Duplicate truth rowid: {rowid}")

        truth_by_id[rowid] = label

    predictions_by_id: dict[str, float] = {}

    for rowid, score in predictions:
        # Validation
        if not isinstance(rowid, str) or not rowid:
            raise ValueError("Invalid prediction rowid")

        if rowid in predictions_by_id:
            raise ValueError(f"Duplicate prediction rowid: {rowid}")

        if rowid not in truth_by_id:
            raise ValueError(f"Unknown prediction rowid: {rowid}")

        predictions_by_id[rowid] = score

    missing = truth_by_id.keys() - predictions_by_id.keys()

    # Validation
    if missing:
        example = next(iter(missing))
        raise ValueError(f"Missing predictions for {len(missing)} rows; example={example}")

    aligned_labels = list(truth_by_id.values())

    aligned_scores = [predictions_by_id[rowid] for rowid in truth_by_id]

    return global_roc_auc(aligned_labels, aligned_scores)
