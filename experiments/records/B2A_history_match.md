# Experiment B2A — Historical Item Matching

**Status:** Development evaluation completed
**Implementation commit:** `5d862d2`
**Branch:** `experiment/b2-history-match`
**Experiment type:** Preregistered, controlled comparison

## 1. Hypothesis

A candidate item's presence in a user's previous click history provides additional predictive information beyond historical item popularity and user click propensity.

The alternative possibility is that repeated exposure does not improve prediction, or that previous interaction reduces the likelihood of another click.

## 2. Methodology

B2A extends the B1 baseline using a binary historical item-matching feature.

A candidate receives a match indicator of 1 when its item identifier appears in the user's previous-click history.

The predetermined scoring rule is:

`B2A_score = clip(B1_score + 0.05 * history_match, 0, 1)`

The history bonus was fixed at 0.05 before development evaluation.

No hyperparameter optimisation was performed.

## 3. Validation

**Dataset:** Official Huawei TechArena training CSV
**Fitting period:** Before January 23, 2025, 00:00 UTC
**Validation period:** January 23, 2025
**Validation selection:** First eligible event per user
**Metric:** Global candidate-level ROC AUC

**Fitting candidate rows:** 2,951,140
**Validation events:** 34,912
**Validation candidate rows:** 349,120

**Implementation version:** `5d862d2`

**Dataset fingerprint:** Verify against the previously recorded local SHA-256 before claiming exact input identity.

**Training dataset SHA-256:**
e9c7a23ebb56123fe2a659ea3e30435ab181fec62d122d0ad580e7b2c0bbd2a2

**Dataset integrity verification:**
The training CSV produced matching SHA-256 fingerprints
before and after the B2A development experiment.

**Implementation commit:** 5d862d2

## 4. Results

| Metric                   |    Result |
| ------------------------ | --------: |
| B0 global ROC AUC        |  0.524285 |
| B1 global ROC AUC        |  0.533983 |
| B2A global ROC AUC       |  0.571505 |
| B2A improvement over B1  | +0.037522 |
| B1 within-event AUC      |  0.523293 |
| B2A within-event AUC     |  0.562118 |
| Within-event improvement | +0.038825 |

## 5. Population Diagnostics

| Observation                        |  Count |
| ---------------------------------- | -----: |
| Validation events                  | 34,912 |
| Events containing a click          | 21,909 |
| Events without clicks              | 13,003 |
| History-matched candidates         |  8,829 |
| History-matched clicked candidates |  2,548 |

Approximately 2.53% of validation candidates matched the user's previous-click history.

The observed click rate among matched candidates was approximately 28.86%.

The observed click rate among unmatched candidates was approximately 5.69%.

These observations suggest a strong association between historical matching and subsequent clicks in the development population.

They do not establish causality or hidden-test generalisation.

## 6. Implementation Verification

All 64 automated tests passed.

The B2A implementation reproduced the previous B0 and B1 global ROC AUC values.

Regression tests covered historical matching, absence of history matches, and rejection of invalid future history timestamps.

The experiment implementation was committed before the real development evaluation.

## 7. Overfitting Assessment

No hyperparameter search was performed.

The scoring bonus was fixed before evaluation.

The experiment was evaluated on one development window.

No independent temporal confirmation has been performed.

No event-clustered uncertainty estimate has been calculated.

Consequently, generalisation remains unverified.

## 8. Potential Failure Patterns

The improvement may depend on the frequency of repeated-item exposure.

History-match prevalence may differ between development and official evaluation periods.

Performance may be concentrated among users with long interaction histories.

Cold-start users may receive no benefit from historical matching.

Repeated experimentation on the development window could create validation-selection overfitting.

## 9. Conclusion

**Decision:** Retain B2A as a promising candidate for further evaluation.

**Measured development improvement:** +0.037522 global ROC AUC over B1.

The experiment also improved within-event candidate ranking.

This supports further investigation of historical user-item interactions.

B2A must not yet be described as generalising to unseen evaluation data.

## 10. Next Validation

Evaluate the unchanged B2A scoring rule on a separately designated temporal robustness window.

Preserve the reserved confirmation window.

Record differences in history-match prevalence, click rates, and AUC.

Do not adjust the bonus after inspecting robustness results without treating the evaluation as part of model development.
