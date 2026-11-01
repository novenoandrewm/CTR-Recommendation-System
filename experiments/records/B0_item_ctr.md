# Experiment B0 — Smoothed Item CTR Baseline

**Status:** Completed
**Implementation commit:** `04b1989`
**Model:** Smoothed item click-through rate
**Primary metric:** Global ROC AUC

## 1. Hypothesis

Historical item popularity contains useful information for predicting candidate clicks.

This experiment establishes a simple statistical reference against which more sophisticated models can be evaluated.

## 2. Methodology

The model estimates each item's click probability using historical impressions and clicks.

Smoothing is applied using a global click-rate prior.

**Smoothing parameter:** 50

Only fitting-period observations contribute to item statistics.

Validation labels are excluded from fitting.

Predictions are aligned using their original candidate row identifiers.

## 3. Dataset and Validation

**Dataset:** Official Huawei TechArena training CSV

**Fitting period:** Before January 23, 2025, 00:00 UTC
**Validation period:** January 23, 2025
**Validation selection:** First eligible recommendation event per user

**Fitting candidate rows:** 2,951,140
**Validation events:** 34,912
**Validation candidate rows:** 349,120

**Dataset SHA-256:** Current local training dataset SHA-256 (measured 2026-10-31):
e9c7a23ebb56123fe2a659ea3e30435ab181fec62d122d0ad580e7b2c0bbd2a2

Provenance limitation:
The fingerprint was captured after this experiment.
Byte-for-byte identity with the dataset used during the original run
has not been independently verified.

## 4. Results

| Metric                        |        Value |
| ----------------------------- | -----------: |
| Development global ROC AUC    |     0.524285 |
| Training ROC AUC              | Not measured |
| Reserved confirmation ROC AUC | Not measured |
| Official evaluation ROC AUC   | Not measured |

## 5. Overfitting Assessment

No hyperparameter search was performed.

The smoothing parameter was fixed before evaluation.

Because training and independent confirmation metrics were not measured, the experiment does not provide sufficient evidence to quantify overfitting.

The model has low complexity, but low complexity alone does not guarantee generalisation.

## 6. Limitations

The model does not use user identity, interaction histories, or candidate-specific personalisation.

It cannot distinguish between users with different preferences when scoring the same candidate.

Performance has been measured on only one development validation window.

## 7. Conclusion

**Decision:** Retain as the B0 reference baseline.

**Measured development AUC:** 0.524285

This result demonstrates that the implemented pipeline can produce a measurable validation score.

It does not establish expected hidden-test performance.
