"""Paired event-cluster bootstrap for fixed B1 and B2A predictions.

The bootstrap preserves complete recommendation events.

Global ROC AUC is calculated using event multiplicities rather
than physically duplicating candidate rows.

The same bootstrap sample is used for both models."""

# Import standard libraries
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

# Import third-party libraries
import numpy as np

# Import local libraries
from ctr_recommendation.analysis.history_failure import (
    ScoredEvent,
    event_ranking,
    score_validation_events,
)

BOOTSTRAP_REPLICATES = 500
BOOTSTRAP_SEED = 2026
CONFIDENCE_LEVEL = 0.95


class WeightedGlobalAUC:
    """Calculate exact weighted ROC AUC with correct tie handling."""

    def __init__(
        self,
        event_indices: list[int],
        labels: list[int],
        scores: list[float],
    ) -> None:
        event_array = np.asarray(
            event_indices,
            dtype=np.int64,
        )

        label_array = np.asarray(
            labels,
            dtype=np.int64,
        )

        score_array = np.asarray(
            scores,
            dtype=np.float64,
        )

        if event_array.ndim != 1 or label_array.ndim != 1 or score_array.ndim != 1:
            raise ValueError("Inputs must be one-dimensional")

        if not (len(event_array) == len(label_array) == len(score_array)):
            raise ValueError("Input lengths must match")

        if len(event_array) == 0:
            raise ValueError("No candidate observations")

        if np.any(event_array < 0):
            raise ValueError("Negative event index")

        if not np.all((label_array == 0) | (label_array == 1)):
            raise ValueError("Labels must be binary")

        if not np.all(np.isfinite(score_array)):
            raise ValueError("Scores must be finite")

        # Scores are sorted only once, not once per bootstrap.
        order = np.argsort(
            score_array,
            kind="stable",
        )

        sorted_scores = score_array[order]

        self.event_indices = event_array[order]
        self.labels = label_array[order]

        self.minimum_events = int(event_array.max()) + 1

        # Identify the beginning of each equal-score group.
        # Every group contains candidates with identical scores.
        self.group_starts = np.concatenate(
            (
                np.array([0], dtype=np.int64),
                np.flatnonzero(np.diff(sorted_scores) != 0) + 1,
            )
        )

    def score(
        self,
        event_multiplicities: np.ndarray,
    ) -> float:
        """Return exact global AUC for weighted candidate observations.

        AUC is the probability that a weighted positive candidate
        outranks a weighted negative candidate, with half credit
        for equal scores.
        """

        multiplicities = np.asarray(event_multiplicities)

        if multiplicities.ndim != 1:
            raise ValueError("Event multiplicities must be one-dimensional")

        if len(multiplicities) < self.minimum_events:
            raise ValueError("Insufficient event multiplicities")

        if not np.issubdtype(
            multiplicities.dtype,
            np.integer,
        ):
            raise TypeError("Event multiplicities must be integers")

        if np.any(multiplicities < 0):
            raise ValueError("Negative event multiplicity")

        # Each candidate inherits its event's multiplicity.
        weights = multiplicities[self.event_indices]

        positive_weights = weights * self.labels
        negative_weights = weights - positive_weights

        # Sum candidate weights within each equal-score group.
        group_positive = np.add.reduceat(
            positive_weights,
            self.group_starts,
        )

        group_negative = np.add.reduceat(
            negative_weights,
            self.group_starts,
        )

        total_positive = int(group_positive.sum())
        total_negative = int(group_negative.sum())

        if total_positive == 0 or total_negative == 0:
            raise ValueError("ROC AUC requires positive and negative labels")

        # Negative weight strictly below each score group.
        negative_below = (
            np.cumsum(
                group_negative,
                dtype=np.float64,
            )
            - group_negative
        )

        # Every positive beats all lower-scored negatives.
        # Equal-scored positive/negative pairs receive half credit.
        comparisons = np.dot(
            group_positive.astype(np.float64),
            negative_below + 0.5 * group_negative,
        )

        denominator = float(total_positive) * float(total_negative)

        return float(comparisons / denominator)


@dataclass(frozen=True)
class BootstrapInputs:
    """Precomputed data needed for event-cluster resampling."""

    event_count: int
    b1_auc: WeightedGlobalAUC
    b2_auc: WeightedGlobalAUC
    clicked_mask: np.ndarray
    within_b1: np.ndarray
    within_b2: np.ndarray


@dataclass(frozen=True)
class BootstrapResult:
    """Observed metrics and paired bootstrap intervals."""

    event_count: int
    clicked_events: int
    replicates: int
    seed: int

    b1_global: float
    b2_global: float
    global_difference: float
    global_ci_lower: float
    global_ci_upper: float

    b1_within: float
    b2_within: float
    within_difference: float
    within_ci_lower: float
    within_ci_upper: float


def build_bootstrap_inputs(
    events: dict[str, ScoredEvent],
) -> BootstrapInputs:
    """Flatten events while preserving their cluster identities."""

    if not events:
        raise ValueError("No events supplied")

    event_indices: list[int] = []
    labels: list[int] = []
    b1_scores: list[float] = []
    b2_scores: list[float] = []

    event_count = len(events)

    clicked_mask = np.zeros(
        event_count,
        dtype=np.int64,
    )

    within_b1 = np.zeros(
        event_count,
        dtype=np.float64,
    )

    within_b2 = np.zeros(
        event_count,
        dtype=np.float64,
    )

    seen_rowids: set[str] = set()

    # Sorting event IDs makes array construction deterministic.
    for event_index, event_id in enumerate(sorted(events)):
        event = events[event_id]

        if not event.candidates:
            raise ValueError(f"Event {event_id} contains no candidates")

        ranking = event_ranking(event)

        if ranking is not None:
            clicked_mask[event_index] = 1

            within_b1[event_index] = ranking[0]
            within_b2[event_index] = ranking[1]

        for candidate in event.candidates:
            if candidate.rowid in seen_rowids:
                raise ValueError(f"Duplicate row ID: {candidate.rowid}")

            seen_rowids.add(candidate.rowid)

            event_indices.append(event_index)
            labels.append(candidate.clicked)

            b1_scores.append(candidate.b1_score)
            b2_scores.append(candidate.b2_score)

    if clicked_mask.sum() == 0:
        raise ValueError("No clicked events available")

    return BootstrapInputs(
        event_count=event_count,
        b1_auc=WeightedGlobalAUC(
            event_indices,
            labels,
            b1_scores,
        ),
        b2_auc=WeightedGlobalAUC(
            event_indices,
            labels,
            b2_scores,
        ),
        clicked_mask=clicked_mask,
        within_b1=within_b1,
        within_b2=within_b2,
    )


def paired_event_bootstrap(
    inputs: BootstrapInputs,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> BootstrapResult:
    """Estimate paired uncertainty by resampling complete events."""

    if replicates < 1:
        raise ValueError("Replicates must be positive")

    n_events = inputs.event_count

    # All events receive weight one in the original population.
    original_weights = np.ones(
        n_events,
        dtype=np.int64,
    )

    observed_b1_global = inputs.b1_auc.score(original_weights)

    observed_b2_global = inputs.b2_auc.score(original_weights)

    observed_global_difference = observed_b2_global - observed_b1_global

    clicked_events = int(inputs.clicked_mask.sum())

    observed_b1_within = float(
        np.dot(
            original_weights,
            inputs.within_b1,
        )
        / clicked_events
    )

    observed_b2_within = float(
        np.dot(
            original_weights,
            inputs.within_b2,
        )
        / clicked_events
    )

    observed_within_difference = observed_b2_within - observed_b1_within

    rng = np.random.default_rng(seed)

    global_differences = np.empty(
        replicates,
        dtype=np.float64,
    )

    within_differences = np.empty(
        replicates,
        dtype=np.float64,
    )

    # Predetermined number of bootstrap replicates.
    rejected_resamples = 0

    for iteration in range(replicates):
        # Resample complete events until the sample contains
        # at least one clicked event.
        for _ in range(100):
            sampled_events = rng.integers(
                0,
                n_events,
                size=n_events,
            )

            multiplicities = np.bincount(
                sampled_events,
                minlength=n_events,
            ).astype(np.int64)

            selected_clicked_events = int(
                np.dot(
                    multiplicities,
                    inputs.clicked_mask,
                )
            )

            if selected_clicked_events > 0:
                break

            rejected_resamples += 1

        else:
            raise RuntimeError("Unable to obtain a valid bootstrap sample after 100 attempts")

        if rejected_resamples:
            print(
                f"[BOOTSTRAP] Rejected {rejected_resamples} resamples without clicked events.",
                flush=True,
            )

        # Exactly the same event sample is used for both models.
        b1_global = inputs.b1_auc.score(multiplicities)

        b2_global = inputs.b2_auc.score(multiplicities)

        global_differences[iteration] = b2_global - b1_global

        # Within-event AUC is defined only on clicked events.
        selected_clicked_events = int(
            np.dot(
                multiplicities,
                inputs.clicked_mask,
            )
        )

        if selected_clicked_events == 0:
            raise ValueError("Bootstrap sample has no clicked events")

        b1_within = float(
            np.dot(
                multiplicities,
                inputs.within_b1,
            )
            / selected_clicked_events
        )

        b2_within = float(
            np.dot(
                multiplicities,
                inputs.within_b2,
            )
            / selected_clicked_events
        )

        within_differences[iteration] = b2_within - b1_within

    alpha = 1.0 - CONFIDENCE_LEVEL

    global_interval = np.quantile(
        global_differences,
        [alpha / 2, 1 - alpha / 2],
    )

    within_interval = np.quantile(
        within_differences,
        [alpha / 2, 1 - alpha / 2],
    )

    return BootstrapResult(
        event_count=n_events,
        clicked_events=clicked_events,
        replicates=replicates,
        seed=seed,
        b1_global=observed_b1_global,
        b2_global=observed_b2_global,
        global_difference=observed_global_difference,
        global_ci_lower=float(global_interval[0]),
        global_ci_upper=float(global_interval[1]),
        b1_within=observed_b1_within,
        b2_within=observed_b2_within,
        within_difference=observed_within_difference,
        within_ci_lower=float(within_interval[0]),
        within_ci_upper=float(within_interval[1]),
    )


def main() -> None:
    """Run the preregistered paired event-cluster bootstrap."""

    parser = argparse.ArgumentParser(description="B2A paired event-cluster bootstrap")

    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/raw/train.csv"),
    )

    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("contracts/input_schema.json"),
    )

    parser.add_argument(
        "--validation-config",
        type=Path,
        default=Path("configs/validation.yaml"),
    )

    parser.add_argument(
        "--window",
        default="development",
    )

    args = parser.parse_args()

    # This existing function enforces the development-window policy
    # before opening the training dataset.
    events = score_validation_events(
        args.train,
        args.contract,
        args.window,
        args.validation_config,
    )

    inputs = build_bootstrap_inputs(events)

    # First validate the observed, unweighted statistics.
    unit_weights = np.ones(
        inputs.event_count,
        dtype=np.int64,
    )

    observed_b1 = inputs.b1_auc.score(unit_weights)
    observed_b2 = inputs.b2_auc.score(unit_weights)

    # Reference values from the frozen January 23 experiment.
    if args.window == "development" and (
        abs(observed_b1 - 0.533983) > 0.00000051 or abs(observed_b2 - 0.571505) > 0.00000051
    ):
        raise ValueError(
            "Unweighted global AUC does not reproduce the original B1/B2A development measurements"
        )

    print(
        "[BOOTSTRAP] Unweighted AUC verification passed.",
        flush=True,
    )

    print(
        "[BOOTSTRAP] Running 500 paired event-cluster replicates...",
        flush=True,
    )

    result = paired_event_bootstrap(inputs)

    print()
    print("=" * 64)
    print("B2A-R2b: PAIRED EVENT-CLUSTER BOOTSTRAP")
    print("=" * 64)

    print(f"Window: {args.window}")
    print(f"Events: {result.event_count:,}")
    print(f"Clicked events: {result.clicked_events:,}")
    print(f"Bootstrap replicates: {result.replicates}")
    print(f"Random seed: {result.seed}")

    print()
    print("GLOBAL ROC AUC")
    print("-" * 64)

    print(f"B1: {result.b1_global:.6f}")
    print(f"B2A: {result.b2_global:.6f}")

    print(f"Observed difference: {result.global_difference:+.6f}")

    print(f"95% bootstrap interval: [{result.global_ci_lower:+.6f}, {result.global_ci_upper:+.6f}]")

    print()
    print("WITHIN-EVENT ROC AUC")
    print("-" * 64)

    print(f"B1: {result.b1_within:.6f}")
    print(f"B2A: {result.b2_within:.6f}")

    print(f"Observed difference: {result.within_difference:+.6f}")

    print(f"95% bootstrap interval: [{result.within_ci_lower:+.6f}, {result.within_ci_upper:+.6f}]")

    print("=" * 64)


if __name__ == "__main__":
    main()
