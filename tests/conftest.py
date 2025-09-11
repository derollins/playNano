"""Provides pytest fixtures for test resource paths."""

import os
import sys
from pathlib import Path

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


@pytest.fixture(scope="session")
def analysis_pipeline_schema():
    """Create a schema for testing the anlaysis output."""
    return {
        "type": "object",
        "required": ["environment", "analysis", "provenance"],
        "properties": {
            "environment": {"type": "object"},  # Could be more detailed if you want
            "analysis": {
                "type": "object",
                "additionalProperties": {
                    "type": "object"
                },  # Each step's outputs are objects
            },
            "provenance": {
                "type": "object",
                "required": ["steps", "results_by_name", "frame_times"],
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": [
                                "index",
                                "name",
                                "params",
                                "timestamp",
                                "version",
                                "analysis_key",
                            ],
                            "properties": {
                                "index": {"type": "integer"},
                                "name": {"type": "string"},
                                "params": {"type": "object"},
                                "timestamp": {"type": "string", "format": "date-time"},
                                "version": {
                                    "anyOf": [{"type": "string"}, {"type": "null"}]
                                },
                                "analysis_key": {"type": "string"},
                            },
                        },
                    },
                    "results_by_name": {
                        "type": "object",
                        "additionalProperties": {
                            "anyOf": [
                                {"type": "object"},  # old single-output case
                                {
                                    "type": "array",
                                    "items": {"type": "object"},
                                },  # new multi-output case
                            ]
                        },
                    },
                    "frame_times": {"anyOf": [{"type": "array"}, {"type": "null"}]},
                },
                "additionalProperties": False,
            },
        },
        "additionalProperties": False,
    }
