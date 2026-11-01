# Experiment B2A-R2 — Historical Feature Failure Analysis

**Status:** Development diagnostics completed
**Implementation commit:** `42aa8e1`
**Validation:** January 23, 2025
**Experiment type:** Failure and subgroup analysis

## Objective

Identify the populations where B2A historical item matching improves, worsens, or does not change CTR candidate ranking.

Investigate failure patterns that aggregate global ROC AUC may conceal.

## Experimental Configuration

The existing B2A scoring formula was preserved.

* Item smoothing: 50
* User propensity weight: 0.5
* History bonus: 0.05
* Validation window: `development`
* Primary metric: Global ROC AUC

No hyperparameter optimisation was performed.

The reserved confirmation window was not accessed.

## Implementation Verification

Ruff checks passed.

All 87 automated tests passed.

The diagnostic implementation reproduced the original B1 and B2A scores.

| Metric                   |    Result |
| ------------------------ | --------: |
| B1 global ROC AUC        |  0.533983 |
| B2A global ROC AUC       |  0.571505 |
| Global improvement       | +0.037522 |
| B1 within-event AUC      |  0.523293 |
| B2A within-event AUC     |  0.562118 |
| Within-event improvement | +0.038825 |

## Population Coverage

| Category                        | Events |
| ------------------------------- | -----: |
| All validation events           | 34,912 |
| Events with no historical match | 26,916 |
| Events with a historical match  |  7,996 |
| Events containing a click       | 21,909 |
| Events without clicks           | 13,003 |

Approximately 77.10% of events have no historical match.

B2A produces identical scores to B1 for these events.

## Ranking Outcomes

Among clicked events:

| Outcome   |  Count |
| --------- | -----: |
| Improved  |  2,167 |
| Worsened  |  1,432 |
| Unchanged | 18,310 |

Among the 5,719 clicked events containing historical matches:

* 37.9% improved.
* 25.0% worsened.
* 37.1% remained unchanged.

These results establish that historical boosting can both improve and damage candidate ranking.

## Historical Match Outcomes

| Metric                         | Count |
| ------------------------------ | ----: |
| History-matched candidates     | 8,829 |
| Matched clicked candidates     | 2,548 |
| Matched non-clicked candidates | 6,281 |

The observed click rate among matched candidates was approximately 28.86%.

Non-clicked matched candidates should not automatically be classified as prediction false positives. A score threshold or ranking error criterion is needed to establish that interpretation.

## History-Length Analysis

| History length | Events | Global AUC improvement |
| -------------- | -----: | ---------------------: |
| 0              |    252 |               0.000000 |
| 1–5            |  4,232 |              +0.048222 |
| 6–10           |  6,199 |              +0.041637 |
| 11–20          | 24,229 |              +0.035170 |
| 21+            |      0 |         Not applicable |

All nonempty history groups exhibited positive development AUC improvements.

These subgroup estimates have not been accompanied by uncertainty intervals.

## Confirmed Findings

Historical matching is predictive within the measured development population.

Most events do not contain an exact historical match.

Some clicked events experience worse candidate rankings after applying the historical bonus.

The feature provides no candidate-level scoring change when no historical match exists.

## Limitations

The analysis uses one development validation period.

The original scoring implementation was independently reconstructed in the diagnostic pipeline, although its aggregate metrics matched the reference implementation.

No paired event-clustered confidence intervals have been calculated.

Subgroup AUC improvements cannot be combined through simple event-count weighting to recover global AUC improvement.

Observed associations do not establish causal effects.

Independent hidden-test generalisation remains unknown.

## Decision

Retain B2A as a development baseline.

Do not change the history bonus based on this diagnostic.

Prioritise uncertainty estimation, ranking-failure investigation, and potential methods for extending candidate-level personalisation beyond exact history matches.

## Next Experiment

B2A-R2b: Paired event-clustered bootstrap.

Estimate uncertainty around global ROC AUC improvement while preserving all candidates belonging to each sampled event.

Continue protecting the reserved confirmation window.
