# Experiment B2A-R2b — Statistical Uncertainty Analysis

**Status:** Development analysis completed
**Experiment:** Paired event-cluster bootstrap
**Validation:** January 23, 2025, UTC
**Primary metric:** Global ROC AUC

## 1. Objective

Estimate uncertainty around B2A's improvement over B1 while preserving recommendation-event structure.

## 2. Experimental Configuration

| Parameter            | Value                         |
| -------------------- | ----------------------------- |
| Validation events    | 34,912                        |
| Clicked events       | 21,909                        |
| Bootstrap replicates | 500                           |
| Random seed          | 2026                          |
| Confidence interval  | 95% percentile                |
| Resampling unit      | Complete recommendation event |
| Model retraining     | None                          |

The same event multiplicities were applied to B1 and B2A.

The model scoring rules were unchanged.

## 3. Global ROC AUC Results

| Metric               |     Value |
| -------------------- | --------: |
| B1 ROC AUC           |  0.533983 |
| B2A ROC AUC          |  0.571505 |
| Observed improvement | +0.037522 |
| 95% CI lower bound   | +0.035569 |
| 95% CI upper bound   | +0.039489 |

The confidence interval lies entirely above zero.

## 4. Within-Event Ranking Results

| Metric               |     Value |
| -------------------- | --------: |
| B1 within-event AUC  |  0.523293 |
| B2A within-event AUC |  0.562118 |
| Observed improvement | +0.038825 |
| 95% CI lower bound   | +0.036653 |
| 95% CI upper bound   | +0.040852 |

The confidence interval lies entirely above zero.

## 5. Implementation Verification

The unweighted global ROC AUC calculation reproduced the previously recorded B1 and B2A development results.

The weighted AUC implementation passed tests involving explicit event duplication, tied scores, zero-weight events, invalid multiplicities, and deterministic resampling.

Ruff checks passed.

All 96 automated tests passed before the real bootstrap evaluation.

## 6. Interpretation

The results provide evidence that B2A's positive development AUC improvement is robust to the specified event-level resampling procedure.

They also support a positive improvement in within-event candidate ranking.

These findings strengthen the justification for retaining B2A as a development baseline.

## 7. Limitations

The confidence intervals are conditional on the fixed models and observed development population.

They do not account for temporal distribution shift, repeated model selection, or model-training uncertainty.

Event-level resampling may not capture dependencies associated with shared item popularity.

The 500-replicate percentile intervals are Monte Carlo estimates.

No independent confirmation or official evaluation labels were used.

## 8. Decision

Retain B2A unchanged as the strongest measured simple development baseline.

Freeze its parameters:

* Item smoothing: 50
* User propensity weight: 0.5
* Historical-match bonus: 0.05

Do not interpret the bootstrap interval as a forecast of official hidden-test performance.

## 9. Next Experiment

B2B — Item-Item Similarity.

Investigate whether candidate-specific similarity to previously clicked items can improve prediction, particularly in recommendation events without exact historical item matches.

The B2B experiment must use temporal leakage safeguards, predefined evaluation criteria, and identical development candidate rows.

The January 24 confirmation window remains reserved.
