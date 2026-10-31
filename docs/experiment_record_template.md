# Experiment Record

**Experiment ID:**
**Experiment name:**
**Date (UTC):**
**Researcher:**
**Status:** Proposed / Completed / Failed / Inconclusive

## 1. Hypothesis

What specific prediction signal or modelling change are we testing?

What result would indicate that the hypothesis is unsupported?

## 2. Reproducibility

**Git commit:**
**Git working tree:** Clean / Dirty
**Python version:**
**Dependency lockfile revision:**
**Random seed:**
**Dataset fingerprint:**
**Hardware:**

## 3. Dataset and Validation

**Fitting period:**
**Validation period:**
**Validation-selection rule:**
**Training rows:**
**Validation events:**
**Validation candidate rows:**
**Positive labels:**

## 4. Model Configuration

**Algorithm:**
**Features:**
**Hyperparameters:**
**Regularisation:**
**Early stopping:**
**Training procedure:**
**Leakage-prevention rules:**

## 5. Results

| Metric           | Training | Development validation | Reserved confirmation |
| ---------------- | -------- | ---------------------- | --------------------- |
| Global ROC AUC   |          |                        |                       |
| Other diagnostic |          |                        |                       |

Only populate a cell when the corresponding measurement has actually been performed.

Document fold-to-fold variation, subgroup performance, and uncertainty where available.

## 6. Overfitting Assessment

Describe the relationship between fitting performance and validation performance.

Identify any unexpected generalisation gaps or unstable temporal results.

Record whether the validation set has been used repeatedly during model selection.

## 7. Failure Analysis

Which observations or user groups performed poorly?

Were any experiments unsuccessful or interrupted?

Were any examples excluded?

What evidence could contradict the observed improvement?

## 8. Conclusion

**Decision:** Keep / Reject / Investigate further

**Supporting evidence:**

**Alternative explanations:**

**Generalisation limitations:**

**Next experiment:**
