# Experiment B2A-R1 — Temporal Robustness

**Status:** Development diagnostic completed
**Model:** B2A — Historical Item Matching
**Development branch:** `experiment/b2a-temporal-robustness`
**Starting repository revision:** `e8bddea`
**Evaluation period:** January 22, 2025 (UTC)

## 1. Objective

Investigate whether the historical item-matching signal observed on January 23 also improves CTR discrimination on an earlier development period.

This is a temporal consistency diagnostic, not an independent confirmation test.

## 2. Experimental Configuration

The B2A model was evaluated without modifying its scoring formula.

| Parameter              | Value                |
| ---------------------- | -------------------- |
| Item smoothing         | 50                   |
| User propensity weight | 0.5                  |
| Historical match bonus | 0.05                 |
| Primary metric         | Global ROC AUC       |
| Validation selection   | First event per user |
| Validation window      | `development_early`  |

No hyperparameter search was conducted.

The reserved January 24 confirmation period was not accessed.

## 3. January 22 Results

| Metric                     |     Value |
| -------------------------- | --------: |
| Fitting observations       | 1,970,170 |
| Validation events          |    35,112 |
| Validation candidates      |   351,120 |
| Events containing a click  |    22,328 |
| Events without clicks      |    12,784 |
| History-matched candidates |     8,983 |
| History-matched clicks     |     2,806 |
| B0 global AUC              |  0.522658 |
| B1 global AUC              |  0.531723 |
| B2A global AUC             |  0.573717 |
| B2A improvement over B1    | +0.041994 |
| B1 within-event AUC        |  0.522145 |
| B2A within-event AUC       |  0.564839 |
| Within-event improvement   | +0.042694 |

## 4. Comparison With January 23

| Metric                   | January 22 | January 23 |
| ------------------------ | ---------: | ---------: |
| B1 global AUC            |   0.531723 |   0.533983 |
| B2A global AUC           |   0.573717 |   0.571505 |
| B2A improvement          |  +0.041994 |  +0.037522 |
| Within-event improvement |  +0.042694 |  +0.038825 |
| History-match prevalence |      2.56% |      2.53% |
| Matched-candidate CTR    |     31.24% |     28.86% |
| Unmatched-candidate CTR  |      5.71% |      5.69% |

## 5. Findings

B2A improved global discrimination and within-event candidate ranking on both development dates.

The historical-match feature exhibited similar coverage across the two validation populations.

Matched candidates had substantially higher observed click rates than unmatched candidates.

These observations provide evidence of temporal consistency within the development data.

## 6. Limitations

The development windows are not statistically independent.

The January 22 data is included in the fitting period used for the January 23 experiment.

The two evaluations use different fitting-data volumes.

No event-clustered confidence intervals have been calculated.

History-length and cold-start subgroup performance have not been measured.

The official evaluation distribution remains unknown.

This diagnostic was proposed after the initial B2A development result; it should not be represented as an independently preregistered confirmatory experiment.

## 7. Decision

**Retain B2A for further development.**

Do not alter the historical-match bonus based on this diagnostic.

Preserve January 24 for restricted confirmation after the model-selection process.

## 8. Next Investigation

Analyse the contribution of historical matching across user-history lengths and other available subgroups.

Quantify uncertainty using paired event-level resampling.

Continue recording both successful and unsuccessful experiments.

Do not claim hidden-test generalisation from these development results.
