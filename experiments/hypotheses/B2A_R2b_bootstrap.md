# Experiment B2A-R2b — Paired Event-Cluster Bootstrap

**Status:** Preregistered — results not yet examined

## Research Question

How sensitive is the observed B2A improvement over B1 to the composition of recommendation events in the January 23 development validation population?

## Models

B1 and B2A are frozen.

No model fitting, feature engineering, scoring parameters, or hyperparameter optimisation will be performed during this experiment.

## Dataset

Official Huawei TechArena training dataset.

Validation window: `development`, January 23, 2025 UTC.

The existing first-event-per-user selection and temporal fitting boundaries will be preserved.

The January 24 confirmation window remains reserved.

## Primary Statistic

Global ROC AUC difference:

`delta = AUC(B2A) - AUC(B1)`

Previously measured development difference: `+0.037522`.

## Secondary Statistic

Difference in within-event ROC AUC on events containing an observed click.

## Bootstrap Design

* Resampling unit: Complete recommendation event.
* Candidate structure: Preserve all candidates within sampled events.
* Sampling: Uniform event sampling with replacement.
* Sample size: Same number of events as the original validation population.
* Replicates: 500.
* Random seed: 2026.
* Confidence interval: 95% percentile bootstrap interval.
* Pairing: B1 and B2A use identical sampled-event multiplicities.
* Model retraining: None.

## Implementation Validation

The weighted global ROC AUC must reproduce the existing unweighted ROC AUC on the original population.

Weighted event multiplicities must agree with physically duplicated event observations in synthetic tests.

Ties must receive half credit.

The original B1 and B2A development AUC values must be reproduced before interpreting bootstrap results.

## Interpretation

An interval entirely above zero would support a positive B2A improvement under event-level resampling of the observed development population.

The interval does not measure temporal distribution shift, model-training uncertainty, or uncertainty introduced by previous model-selection decisions.

The result is not independent confirmation of official evaluation performance.

## Decision Rule

Report the observed improvement and both interval endpoints regardless of the outcome.

Do not change B2A's scoring parameters in response to the confidence interval.

Preserve the January 24 confirmation window.
