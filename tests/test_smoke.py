"""Basic package installation and import tests."""

from importlib.metadata import version

import ctr_recommendation


# Function test package import
def test_package_import() -> None:
    """The project package must be importable."""
    assert ctr_recommendation is not None


# Function test version version
def test_package_version() -> None:
    """The installed package must expose the expected versin."""
    assert version("ctr-recommendation-system") == "0.1.0"
