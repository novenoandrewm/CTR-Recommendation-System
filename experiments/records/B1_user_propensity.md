# Experiment B1 — Historical User Click Propensity

**Status:** Completed
**Implementation commit:** `c0433a5`
**Model:** Smoothed item CTR with user click-propensity adjustment
**Primary metric:** Global ROC AUC

## 1. Hypothesis

Historical user-level clicking behaviour contains predictive information beyond item popularity.

The experiment tests whether adding this information improves global candidate-level discrimination.

## 2. Methodology

B1 extends the B0 item-popularity baseline.

User-level impressions and clicks are calculated exclusively from observations before the validation cutoff.

The scoring function adds a smoothed user-propensity adjustment to the item CTR score.

**Item smoothing:** 50
**User smoothing:** 50
**User-adjustment weight:** 0.5

These parameters were fixed for the experiment rather than selected through a hyperparameter search.

B0 and B1 use identical fitting and validation populations.

## 3. Dataset and Validation

**Dataset:** Official Huawei TechArena training CSV

**Fitting period:** Before January 23, 2025, 00:00 UTC
**Validation period:** January 23, 2025
**Validation selection:** First eligible recommendation event per user

**Fitting candidate rows:** 2,951,140
**Validation events:** 34,912
**Validation candidate rows:** 349,120

**Known-user coverage:** 93.90%

**Dataset SHA-256:** Current local training dataset SHA-256 (measured 2026-10-31):
e9c7a23ebb56123fe2a659ea3e30435ab181fec62d122d0ad580e7b2c0bbd2a2

Provenance limitation:
The fingerprint was captured after this experiment.
Byte-for-byte identity with the dataset used during the original run
has not been independently verified.

## 4. Results

| Metric                        |           B0 |           B1 |
| ----------------------------- | -----------: | -----------: |
| Development global ROC AUC    |     0.524285 |     0.533983 |
| Improvement                   |            — |    +0.009698 |
| Training ROC AUC              | Not measured | Not measured |
| Reserved confirmation ROC AUC | Not measured | Not measured |
| Official evaluation ROC AUC   | Not measured | Not measured |

## 5. Overfitting Assessment

The improvement was measured on one development validation window.

No independent temporal confirmation has been performed.

No training-versus-validation performance gap has been measured.

The model uses a fixed combination weight rather than a tuned parameter.

Consequently, there is insufficient evidence to determine whether the improvement will persist on unseen evaluation data.

## 6. Critical Interpretation

The user-propensity adjustment is constant across candidates belonging to the same recommendation event.

Before clipping, this adjustment cannot change the ranking of candidates within that event. Clipping may create ties but cannot reverse their ordering.

Therefore, B1's improvement demonstrates stronger global candidate-level discrimination on the development population.

It does not establish improved personalised candidate ranking within individual events.

## 7. Potential Failure Modes

User statistics may be unreliable for users with few fitting-period observations.

Known-user coverage does not establish the quality of the available user histories.

The development population may differ from the official evaluation population.

Repeated experimentation on the same development window creates a risk of validation overfitting.

## 8. Conclusion

**Decision:** Retain B1 as a reference experiment.

**Measured improvement over B0:** +0.009698 global ROC AUC.

The result supports the hypothesis that user-level click propensity contains additional predictive information on the development population.

Generalisation remains unverified.
