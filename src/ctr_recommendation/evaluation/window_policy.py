"""Enforce permitted validation windows during model development."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from ctr_recommendation.data.splits import (
    ValidationWindow,
    load_validation_windows,
)


class ReservedWindowError(ValueError):
    """Raised when a reserved validation window is accessed."""


def require_development_window(
    window_name: str,
    allowed_windows: Sequence[str],
) -> None:
    """Reject validation windows not authorised for development."""

    if window_name not in allowed_windows:
        raise ReservedWindowError(
            f"Validation window '{window_name}' is reserved "
            "and cannot be used during model development."
        )


def allowed_development_windows(
    config: Mapping[str, object],
) -> tuple[str, ...]:
    """Read the explicitly authorised development windows."""

    # Case 1: The required configuration is missing.
    if "model_selection" not in config:
        raise ValueError("Missing model_selection configuration")

    model_selection = config["model_selection"]

    # Case 2: The configuration exists but has the wrong type.
    if not isinstance(model_selection, Mapping):
        raise TypeError("model_selection must be a mapping")

    # Case 3: The development-window allowlist is missing.
    if "allowed_development_windows" not in model_selection:
        raise ValueError("Missing allowed development windows")

    allowed = model_selection["allowed_development_windows"]

    # Case 4: The allowlist has the wrong type.
    if not isinstance(allowed, list):
        raise TypeError("allowed_development_windows must be a list")

    # Case 5: An empty allowlist is invalid.
    if not allowed:
        raise ValueError("Missing allowed development windows")

    # Case 6: Every window name must be a nonempty string.
    if not all(isinstance(name, str) and bool(name) for name in allowed):
        raise TypeError("Invalid development window configuration")

    return tuple(allowed)


def load_development_window(
    config_path: Path,
    window_name: str,
) -> ValidationWindow:
    """Load a temporal window only if development access is authorised.

    A window must:
        1. Appear in the development allowlist.
        2. Not appear in the reserved-window list.
        3. Have purpose='model_development'.

    Missing or inconsistent policy configuration fails closed.
    """

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, Mapping):
        raise TypeError("Validation configuration must be a mapping")

    # Verify the explicit development allowlist.
    allowed = allowed_development_windows(config)

    require_development_window(
        window_name,
        allowed,
    )

    model_selection = config["model_selection"]

    if "reserved_windows" not in model_selection:
        raise ValueError("Missing reserved_windows configuration")

    reserved = model_selection["reserved_windows"]

    if not isinstance(reserved, list):
        raise TypeError("reserved_windows must be a list")

    if not all(isinstance(name, str) and bool(name) for name in reserved):
        raise TypeError("Invalid reserved_windows configuration")

    # A window cannot simultaneously be permitted and reserved.
    if window_name in reserved:
        raise ReservedWindowError(
            f"Validation window '{window_name}' is reserved "
            "and cannot be used during model development."
        )

    definitions = config.get("windows")

    if not isinstance(definitions, Mapping):
        raise TypeError("Missing or invalid windows configuration")

    # First, check whether the requested window exists.
    if window_name not in definitions:
        raise ValueError(f"Unknown validation window: {window_name}")

    definition = definitions[window_name]

    # Then, validate the window configuration type.
    if not isinstance(definition, Mapping):
        raise TypeError(f"Validation window '{window_name}' must be a mapping")

    # The declared purpose must agree with the access policy.
    if definition.get("purpose") != "model_development":
        raise ReservedWindowError(
            f"Validation window '{window_name}' is not authorised for model development."
        )

    # Parse timestamps using our existing validated splitter.
    windows = load_validation_windows(config_path)

    try:
        return windows[window_name]
    except KeyError as exc:
        raise ValueError(f"Unknown validation window: {window_name}") from exc
