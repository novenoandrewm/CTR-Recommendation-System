# Experiment B2A — Historical Item Matching

**Status:** Preregistered — not yet evaluated

## 1. Research Question

Does a candidate item's presence in the user's previous click history improve CTR prediction beyond item popularity and user click propensity?

## 2. Hypothesis

Previously clicked items may be more relevant to a user than items they have never clicked.

A binary historical-match feature may therefore improve candidate-level discrimination.

The hypothesis is not assumed to be true. Historical exposure may also indicate that a user is unlikely to click the same item again.

## 3. Control

B1 — Smoothed item CTR with user click propensity.

Previously measured development global ROC AUC: **0.533983**.

## 4. Treatment

B2A adds a binary feature:

`history_match = 1` if the candidate item appears in the supplied previous-click history; otherwise `0`.

Scoring rule:

`B2A_score = clip(B1_score + 0.05 * history_match, 0, 1)`

The bonus is fixed at 0.05 before evaluation.

No hyperparameter search will be conducted in this initial experiment.

## 5. Dataset and Validation

**Dataset:** Official Huawei TechArena training CSV.

**Fitting observations:** Strictly before January 23, 2025, 00:00 UTC.

**Development validation:** January 23, 2025.

**Expected validation events:** 34,912.

**Expected candidate rows:** 349,120.

**Primary metric:** Global ROC AUC.

The B1 and B2A models must be evaluated on identical candidate row IDs.

## 6. Additional Diagnostics

Record:

* B1 global ROC AUC.
* B2A global ROC AUC.
* Difference in global ROC AUC.
* Number of candidates matching their user's history.
* Number of clicked candidates matching their user's history.
* Number of validation events containing clicks.
* Within-event ranking AUC on events with one click.

Events without clicks must be counted separately rather than silently treated as positive-containing events.

## 7. Leakage Prevention

Historical interactions must occur strictly before the prediction event.

No validation labels may enter fitted item or user statistics.

The model must preserve candidate row IDs.

The existing candidate validator must remain enabled.

## 8. Overfitting Prevention

The scoring rule and bonus are fixed before the experiment.

The reserved confirmation window will remain untouched.

The result will be evaluated once on the development window.

Poor results will be documented.

No parameter tuning will be performed in response to the initial B2A score.

## 9. Interpretation

An improvement on development validation supports further investigation.

It does not establish generalisation to the official evaluation population.

A deterioration is also informative and may indicate that repeated-item interest is not a useful signal under this scoring rule.

## 10. Decision Rule

Retain B2A as a candidate for further investigation if it improves development global ROC AUC and passes the leakage and evaluation-integrity checks.

Do not promote it as the final model without additional robustness evidence.
