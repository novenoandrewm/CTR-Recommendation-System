# Temporal Validation Protocol

**Project:** CTR Recommendation System
**Version:** 1.0
**Status:** Proposed; implementation and empirical validation pending

## 1. Objective

Establish a reproducible, leakage-safe evaluation procedure for predicting candidate-level click probabilities in the Huawei TechArena 2026 Phase 1 challenge.

The primary evaluation metric is global ROC AUC.

The protocol prioritises chronological integrity, event grouping, model comparability, and realistic evaluation conditions.

## 2. Official Task

Each recommendation event contains ten candidate impressions.

At most one candidate is clicked.

Evaluation selects the first event at or after the official cutoff for each evaluation user.

Candidate predictions are evaluated using global ROC AUC.

## 3. Temporal Validation

Validation uses chronological cutoffs.

For a cutoff T, fitting observations must satisfy:

`t < T`

Validation observations must satisfy:

`T <= t < validation_end`

All rows belonging to the same event remain together.

The first eligible event per user is selected within each validation window.

Ambiguous earliest timestamps must be reported, and deterministic tie-breaking must not be interpreted as evidence of actual event order.

## 4. Validation Windows

The initial protocol defines:

* Development: 23 January 2025.
* Confirmation: 24 January 2025.
* Weekend robustness: 25–26 January 2025.
* Longer-horizon diagnostic: 23–26 January 2025.

All boundaries use UTC.

Each validation window has its own fitting cutoff.

Results from different windows must not be compared as though they came from identical training populations or were statistically independent.

## 5. Leakage Prevention

Model fitting must exclude labels from the corresponding validation period.

Label-derived statistics must be computed using only permitted fitting observations.

Historical click information may be used when it was available before the target event.

Training features must not contain their own targets through target-derived statistics or incorrectly fitted representations.

All temporal transformations must preserve the relevant information-availability boundary.

## 6. Primary Evaluation

The authoritative metric is global candidate-level ROC AUC.

The metric must be calculated over all eligible candidate rows in the selected validation population.

Event-level ranking metrics may be calculated for diagnostic purposes, but they do not replace global ROC AUC.

## 7. Model Comparison

Candidate and reference models must be evaluated on identical observations within each window.

Every comparison must record:

* Model and configuration identity.
* Dataset identity.
* Training cutoff.
* Validation window.
* Candidate-row and event counts.
* Global ROC AUC.
* Difference from the reference model.
* Relevant segment results.
* Uncertainty estimates where appropriate.

Candidate improvement must be distinguished from sampling variation and repeated model-selection effects.

## 8. Confirmation Policy

Development validation may be used for repeated experimentation.

Confirmation validation is reserved for candidates selected before examining confirmation results.

The candidate's model design and hyperparameters must be frozen before confirmation.

A rejected candidate must not be repeatedly modified using confirmation feedback without acknowledging that the confirmation set has become part of the development process.

## 9. Robustness and Distribution Shift

Evaluation must examine model behaviour across relevant population segments.

At minimum, reports should examine:

* Weekday and weekend behaviour.
* User-history length.
* Candidate-item frequency.
* Previously observed and unseen users.
* Previously observed and unseen items.
* Event arrival timing.

Segment reports must include sample counts and must not imply statistical reliability from small samples.

## 10. Known Limitations

The released training period is short.

As a result, the proposed validation windows cannot perfectly reproduce the official evaluation distribution.

One-day validation windows exclude users whose first eligible events occur later.

Longer-horizon diagnostics overlap the primary validation windows and therefore do not provide independent confirmation evidence.

The amount of available fitting data differs between temporal cutoffs.

These limitations must accompany conclusions drawn from local validation.

## 11. Reproducibility

Every evaluation run must record the data, code, configuration, model, feature, and split identities.

Validation results must be reproducible from independently acquired authorised source data.

No models, feature caches, processed datasets, or generated artifacts from the previous experiment may be imported into the new project.

## 12. Acceptance Criteria

The validation implementation will be considered ready when:

1. Event groups remain intact.
2. Temporal boundaries are enforced.
3. First-event-per-user selection is deterministic and audited.
4. Future-label leakage tests pass.
5. Prediction and label alignment is verified.
6. Global ROC AUC matches trusted reference calculations.
7. Window-level distribution diagnostics are generated.
8. Synthetic regression tests pass.
9. A clean execution reproduces the expected validation partitions.

No advanced modelling experiment should begin before the validation foundation passes these checks.
