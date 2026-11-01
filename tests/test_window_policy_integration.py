"""Integration tests for experiment CLI validation-window access."""

import subprocess
import sys
from pathlib import Path

import pytest

from ctr_recommendation.evaluation.window_policy import (
    ReservedWindowError,
    load_development_window,
)

ROOT = Path(__file__).resolve().parents[1]

CONFIG = ROOT / "configs" / "validation.yaml"


@pytest.mark.parametrize(
    "window_name",
    [
        "development_early",
        "development",
    ],
)
def test_real_development_windows_are_allowed(
    window_name: str,
) -> None:
    """Configured development windows must load successfully."""

    window = load_development_window(
        CONFIG,
        window_name,
    )

    assert window.fit_before == window.validate_from
    assert window.validate_before > window.validate_from


@pytest.mark.parametrize(
    "window_name",
    [
        "confirmation",
        "weekend_robustness",
        "longer_horizon",
    ],
)
def test_real_reserved_windows_are_rejected(
    window_name: str,
) -> None:
    """Reserved windows must remain inaccessible during development."""

    with pytest.raises(
        ReservedWindowError,
        match="reserved",
    ):
        load_development_window(
            CONFIG,
            window_name,
        )


@pytest.mark.parametrize(
    "module",
    [
        "ctr_recommendation.training.baseline",
        "ctr_recommendation.training.user_baseline",
        "ctr_recommendation.training.history_match",
    ],
)
def test_experiment_cli_rejects_confirmation_before_data_access(
    module: str,
    tmp_path: Path,
) -> None:
    """Each runner must reject confirmation before opening training data."""

    missing_csv = tmp_path / "does_not_exist.csv"

    command = [
        sys.executable,
        "-m",
        module,
        "--train",
        str(missing_csv),
        "--validation-config",
        str(CONFIG),
        "--window",
        "confirmation",
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    output = result.stdout + result.stderr

    assert result.returncode != 0

    # The policy must be the reason the command failed.
    assert "ReservedWindowError" in output

    # The missing dataset must not have been accessed.
    assert "FileNotFoundError" not in output
