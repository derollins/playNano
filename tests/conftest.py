"""Provides pytest fixtures for test resource paths."""

from pathlib import Path
import os
import sys


import pytest
from PySide6.QtWidgets import QApplication

from playNano.processing.filters import register_filters
from playNano.processing.mask_generators import register_masking


os.environ["QT_QPA_PLATFORM"] = "offscreen"


@pytest.fixture(scope="session", autouse=True)
def ensure_qapplication():
    """
    Make sure there is a single QApplication for all tests that need it.
    This runs before any test, so any import/instantiation of QWidget will
    see a valid QApp and won't blow up.
    """
    app = QApplication.instance()
    if app is None:
        # Passing sys.argv is usually fine; could also do [] if you prefer.
        _app = QApplication(sys.argv)
        return _app
    return app


@pytest.fixture
def resource_path():
    """Fixture returning the path to the test resources directory."""
    return Path(__file__).parent / "resources"


@pytest.fixture(autouse=True)
def register_all_filters_and_masks():
    """Fixtrue for registering all filteres and masks before tests."""
    # Automatically run before every test
    register_filters()
    register_masking()
