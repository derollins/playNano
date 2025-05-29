"""Provides pytest fixtures for test resource paths."""

from pathlib import Path

import pytest


@pytest.fixture
def resource_path():
    """Fixture returning the path to the test resources directory."""
    return Path(__file__).parent / "resources"
