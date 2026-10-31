# CTR Recommendation System

### Huawei TechArena 2026 — CTR Prediction Challenge

A reproducible machine-learning project for predicting click-through probabilities in recommendation events.

The project prioritises:

* Accurate CTR prediction.
* Generalisation to unseen observations.
* Prevention of temporal and target leakage.
* Reproducible experiments.
* Reliable evaluation and model comparison.
* Transparent documentation of failures and limitations.

## 1. Problem Definition

The objective is to predict the probability that a user clicks a candidate recommendation.

Each recommendation event contains 10 candidate items, with at most one clicked candidate.

The primary evaluation metric is **global candidate-level ROC AUC**.

Models must produce probabilities associated with their correct candidate row identifiers.

## 2. Project Structure

```text
CTR-Recommendation-System/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── configs/
│   ├── validation.yaml
│   ├── experiments/
│   └── models/
│
├── contracts/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│   ├── validation_protocol.md
│   └── experiment_policy.md
│
├── experiments/
│   └── records/
│
├── src/
│   └── ctr_recommendation/
│       ├── data/
│       ├── evaluation/
│       ├── features/
│       ├── inference/
│       ├── models/
│       └── training/
│
├── tests/
├── runs/
├── submissions/
│
├── .gitignore
├── pyproject.toml
├── uv.lock
└── README.md
```

Official datasets, generated predictions, trained models, and temporary experiment outputs are stored locally and excluded from Git.

## 3. Environment Setup

The project uses Python and the `uv` package manager.

Install the required Python version:

```cmd
uv python install 3.13
```

Install locked dependencies:

```cmd
uv sync --locked
```

Verify formatting:

```cmd
uv run --locked ruff format --check src tests
```

Run linting:

```cmd
uv run --locked ruff check src tests
```

Run automated tests:

```cmd
uv run --locked pytest -q
```

Both `pyproject.toml` and `uv.lock` must be version-controlled to support reproducible development.

## 4. Data

The official competition datasets are not included in this repository.

Place authorised copies at:

```text
data/raw/train.csv
data/raw/evaluation.csv
```

Do not commit competition datasets to GitHub.

## 5. Baseline Experiments

Development validation date: January 23, 2025.

Validation population: 34,912 recommendation events and 349,120 candidate rows.

| Experiment | Method                           | Development ROC AUC |
| ---------- | -------------------------------- | ------------------: |
| B0         | Smoothed Item CTR                |            0.524285 |
| B1         | Item CTR + User Click Propensity |            0.533983 |

These are local development-validation measurements, not hidden-test results.

The improvement observed in B1 does not establish better within-event recommendation ranking or generalisation to the official evaluation population.

## 6. Model Evaluation

All experiments must use appropriate temporal separation.

Our evaluation methodology includes:

* Forward-in-time validation.
* Strictly historical features.
* Training-only fitting of supervised statistics.
* Event-level validation selection.
* Row-ID-aligned predictions.
* Global ROC AUC.
* Independent verification of metric calculations.
* Analysis of distribution shifts and model failures.

The reserved confirmation window must not be repeatedly used to tune models.

## 7. Preventing Overfitting

Generalisation is prioritised over maximising development scores.

Every experiment must document its training period, validation population, model configuration, hyperparameters, regularisation, early-stopping procedure where applicable, and potential signs of overfitting.

Model comparisons must use consistent evaluation populations.

A high training score or improvement on a repeatedly inspected validation dataset is not sufficient evidence of generalisation.

## 8. Reproducibility

Each experiment must record:

* Git commit.
* Dataset version or fingerprint.
* Validation configuration.
* Feature configuration.
* Model configuration.
* Random seed.
* Dependency environment.
* Metrics.
* Failures and limitations.

Experiment summaries are maintained under `experiments/records/`.

Large model artifacts and prediction files remain in ignored local directories.

## 9. Development Workflow

Changes should be developed on feature branches and reviewed before merging into `main`.

Automated checks must pass before merging.

The project uses GitHub Actions for formatting, linting, and dataset-independent unit tests.

## 10. Competition and Data Restrictions

All dataset access and publication decisions must comply with the official competition rules.

The repository should remain private unless public release is explicitly authorised.
