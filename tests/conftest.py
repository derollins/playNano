"""Provides pytest fixtures for test resource paths."""

from pathlib import Path

import pytest

from playNano.processing.filters import register_filters
from playNano.processing.mask_generators import register_masking


@pytest.fixture
def resource_path():
    """Fixture returning the path to the test resources directory."""
    return Path(__file__).parent / "resources"


@pytest.fixture(autouse=True)
def register_all_filters_and_masks():
    """Fixtrue for registering all filteres and masks before tests"""
    # Automatically run before every test
    register_filters()
    register_masking()
