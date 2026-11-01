"""Tests for validation window access restrictions."""

import pytest

from ctr_recommendation.evaluation.window_policy import (
    ReservedWindowError,
    allowed_development_windows,
    require_development_window,
)


@pytest.fixture
def policy_config() -> dict[str, object]:
    return {
        "model_selection": {
            "allowed_development_windows": [
                "development_early",
                "development",
            ]
        }
    }


def test_development_allowed(
    policy_config: dict[str, object],
) -> None:
    allowed = allowed_development_windows(policy_config)

    require_development_window("development", allowed)
    require_development_window("development_early", allowed)


@pytest.mark.parametrize(
    "window",
    [
        "confirmation",
        "weekend_robustness",
        "longer_horizon",
    ],
)
def test_reserved_windows_rejected(
    policy_config: dict[str, object],
    window: str,
) -> None:
    allowed = allowed_development_windows(policy_config)

    with pytest.raises(
        ReservedWindowError,
        match="reserved",
    ):
        require_development_window(window, allowed)


def test_missing_policy_rejected() -> None:
    with pytest.raises(ValueError, match="model_selection"):
        allowed_development_windows({})


def test_empty_allowlist_rejected() -> None:
    config = {"model_selection": {"allowed_development_windows": []}}

    with pytest.raises(ValueError, match="development windows"):
        allowed_development_windows(config)


def test_invalid_model_selection_type_rejected() -> None:
    """A model-selection policy must be a mapping."""

    config = {"model_selection": "invalid"}

    with pytest.raises(TypeError, match="must be a mapping"):
        allowed_development_windows(config)


def test_invalid_allowlist_type_rejected() -> None:
    """The development-window allowlist must be a list."""

    config = {"model_selection": {"allowed_development_windows": "development"}}

    with pytest.raises(TypeError, match="must be a list"):
        allowed_development_windows(config)
