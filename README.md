# CTR Recommendation System

A reproducible machine learning project for the Huawei TechArena
2026 Click-Through Rate Prediction Challenge.

## Objective

Predict click probabilities for candidate items using historical
user interactions and available event information.

The primary evaluation metric is global ROC AUC.

## Project Architecture

- `configs/` — Experiment and model configurations
- `contracts/` — Input and output data contracts
- `data/` — Raw and processed datasets
- `docs/` — Architecture and engineering documentation
- `experiments/` — Experiment hypotheses
- `runs/` — Generated experiment artifacts
- `src/ctr_recommendation/` — Application source code
- `submissions/` — Validated submission artifacts
- `tests/` — Automated tests

## Environment

Python 3.11 or newer is required.

Install dependencies:

    uv sync

Verify the command-line interface:

    uv run ctr --help

Run tests:

    uv run pytest -q

Run lint checks:

    uv run ruff check src tests

## Development Principles

- Use reproducible configurations.
- Prevent temporal and target leakage.
- Maintain immutable experiment artifacts.
- Validate model improvements using global ROC AUC.
- Record successful and unsuccessful experiments.
- Do not reuse previous experiment artifacts.