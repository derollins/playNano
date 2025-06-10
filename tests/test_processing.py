"""Tests for the playNano processing modules."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from playNano.cli import actions
from playNano.errors import LoadError
from playNano.processing import core


@patch("playNano.processing.core.AFMImageStack.load_data")
@patch("playNano.processing.core.ProcessingPipeline")
def test_process_stack_runs_pipeline(mock_pipeline_cls, mock_load_data):
    """Test that process_stack loads data and runs the processing pipeline."""
    mock_stack = MagicMock()
    mock_pipeline = MagicMock()
    mock_pipeline_cls.return_value = mock_pipeline
    mock_load_data.return_value = mock_stack

    steps = [("zero_mean", {"axis": 0}), ("flatten", {})]
    result = core.process_stack(Path("file.jpk"), "Height", steps)

    # Check loading
    mock_load_data.assert_called_once_with(Path("file.jpk"), channel="Height")
    # Check pipeline created and filters added
    mock_pipeline_cls.assert_called_once_with(mock_stack)
    mock_pipeline.add_filter.assert_any_call("zero_mean", axis=0)
    mock_pipeline.add_filter.assert_any_call("flatten")
    mock_pipeline.run.assert_called_once()

    assert result is mock_stack


@patch("playNano.processing.core.AFMImageStack.load_data")
@patch("playNano.processing.core.ProcessingPipeline")
def test_process_stack_handles_clear_step(mock_pipeline_cls, mock_load_data):
    """Test that 'clear' step is handled correctly."""
    mock_stack = MagicMock()
    mock_pipeline = MagicMock()
    mock_pipeline_cls.return_value = mock_pipeline
    mock_load_data.return_value = mock_stack

    steps = [("clear", {}), ("flatten", {})]
    result = core.process_stack(Path("data.jpk"), "Deflection", steps)

    mock_pipeline.clear_mask.assert_called_once()
    mock_pipeline.add_filter.assert_called_once_with("flatten")
    mock_pipeline.run.assert_called_once()
    assert result is mock_stack


@patch(
    "playNano.cli.actions.AFMImageStack.load_data",
    side_effect=RuntimeError("file missing"),  # noqa: E501
)
def test_process_stack_raises_on_load_error(mock_load_data):
    """Test that process_stack raises LoadError on loading failure."""
    with pytest.raises(LoadError, match="Failed to load"):
        actions.process_stack(Path("missing.jpk"), "Height", [])

    mock_load_data.assert_called_once_with(Path("missing.jpk"), channel="Height")
