# Experiment Policy

## 1. Primary Objective

The primary objective is to develop CTR prediction models that generalise to unseen recommendation events.

Development ROC AUC is a model-selection tool, not definitive evidence of generalisation.

The official evaluation metric is global candidate-level ROC AUC.

## 2. Temporal Separation

All model experiments must use forward-in-time validation.

The existing project configuration defines the following evaluation windows.

| Window             | Period              | Purpose                         |
| ------------------ | ------------------- | ------------------------------- |
| Development        | January 23, 2025    | Model development               |
| Confirmation       | January 24, 2025    | Reserved model confirmation     |
| Weekend robustness | January 25–26, 2025 | Distribution-shift assessment   |
| Longer horizon     | January 23–26, 2025 | Additional temporal diagnostics |

The confirmation window should remain unused during iterative model development.

The longer-horizon window overlaps other windows and must not be treated as an independent test.

If confirmation data has already influenced modelling decisions, that limitation must be disclosed.

## 3. Cross-Validation

Randomly shuffling candidate rows into training and validation folds is prohibited for experiments requiring historical feature availability.

Where sufficient data exists, use expanding-window or rolling-origin validation.

Each validation fold must be chronologically later than its fitting observations.

All feature transformations must be fitted separately within each fold.

Cross-validation results should be reported across folds, including poor-performing folds.

Because the available training period is short, temporal folds may have substantially different training sizes and distributions. These limitations must be reported.

## 4. Data Leakage Prevention

Features may contain only information available at the time of prediction.

Supervised item statistics, user statistics, and item-item associations must exclude future observations.

Training features derived from clicks must not include their own target labels.

Events sharing the same timestamp must not exchange label-derived information.

Validation and evaluation labels must never be used in feature fitting.

Predictions must remain associated with the correct candidate row identifiers.

## 5. Regularisation and Model Complexity

Use appropriate regularisation for each model family.

Examples include L1/L2 penalties, tree-depth constraints, minimum leaf sizes, feature subsampling, and controlled model capacity.

Prefer simpler models when more complex alternatives offer no consistent improvement.

Regularisation settings must be recorded in the experiment configuration.

## 6. Early Stopping

When early stopping is applicable, use only data designated for model development and selection.

Never use the reserved confirmation window or official evaluation data for iterative early stopping.

Record the stopping metric, stopping patience, and selected training iteration.

## 7. Hyperparameter Optimisation

Define a bounded hyperparameter search before evaluating alternatives.

Record all attempted configurations, including unsuccessful ones.

Do not repeatedly tune against the same confirmation set.

A small improvement on one validation window is insufficient evidence that a model generalises better.

## 8. Overfitting Diagnostics

Where applicable, monitor:

* Training versus validation performance.
* Performance across temporal folds.
* Known-user versus unseen-user performance.
* Short versus long interaction histories.
* Weekday versus weekend performance.
* No-click events and cold-start conditions.
* Performance differences across user and item populations.

Investigate unusually large train–validation gaps and improvements concentrated in narrow subgroups.

Report uncertainty when comparing small AUC differences, preferably using resampling that respects event-level dependence.

## 9. Experiment Logging

Every experiment must document its hypothesis before execution.

Record the Git revision, data provenance, environment, configuration, evaluation population, results, failures, and interpretation.

Do not omit failed experiments or rejected models.

A model that performs poorly is still a useful experimental result.

## 10. Final Model Selection

Freeze the model design and hyperparameters before accessing reserved confirmation data.

Compare candidates using the pre-established selection criteria.

Use confirmation and robustness results to assess whether observed development improvements persist.

Do not repeatedly revise the model after inspecting reserved holdout results while continuing to describe that holdout as untouched.

The official evaluation population has no locally available labels. Consequently, hidden-test performance cannot be guaranteed by local validation.

## 11. Reproducibility

Every reported result must be traceable to a specific code revision and experiment configuration.

Use deterministic random seeds where supported.

Record model dependencies and dataset fingerprints.

Any improvement that cannot be reproduced should be labelled unverified.

## 12. Scientific Integrity

Avoid drawing conclusions exclusively from successful experiments.

Investigate failed runs, missing observations, unexpectedly strong performance, distribution shifts, and data excluded during preprocessing.

Distinguish confirmed defects from plausible risks.

The objective is not to achieve the highest possible development AUC at any cost.

The objective is to develop a model whose performance remains reliable under realistic evaluation conditions.
