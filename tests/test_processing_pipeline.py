"""Tests for the ProcessingPipeline class."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from playNano.afm_stack import AFMImageStack
from playNano.processing.pipeline import ProcessingPipeline


@pytest.fixture
def toy_stack():
    """Create a small dummy AFMImageStack with mock internals."""
    data = np.ones((3, 4, 4), dtype=float)  # 3 frames of 4x4 pixels
    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_path = Path(tmpdirname)
        # pixel_size_nm=1.0, channel=“h”, dummy metadata
        stack = AFMImageStack(data.copy(), 1.0, "h", temp_path, "dum")
        stack.processed = {}

    # Patch resolution and execution methods
    stack._resolve_step = MagicMock()
    stack._execute_mask_step = MagicMock()
    stack._execute_filter_step = MagicMock()

    return stack


def test_add_mask_adds_step(toy_stack):
    """Test that adding a mask step correctly appends to the pipeline."""
    pipe = ProcessingPipeline(toy_stack).add_mask("test_mask", threshold=0.5)
    assert pipe.steps == [("test_mask", {"threshold": 0.5})]


def test_add_filter_adds_step(toy_stack):
    """Test that adding a filter step correctly appends to the pipeline."""
    pipe = ProcessingPipeline(toy_stack).add_filter("gauss", sigma=1.0)
    assert pipe.steps == [("gauss", {"sigma": 1.0})]


def test_clear_mask_adds_clear_step(toy_stack):
    """Test that clear_mask adds a 'clear' step to the pipeline."""
    pipe = ProcessingPipeline(toy_stack).clear_mask()
    assert pipe.steps == [("clear", {})]


def test_run_stores_raw_if_not_present(toy_stack):
    """Test that run() snapshots raw data if not already done."""
    toy_stack._resolve_step.return_value = ("filter", lambda *a, **k: a[0])
    toy_stack._execute_filter_step.side_effect = (
        lambda fn, arr, mask, name, **kwargs: arr + 1
    )

    pipe = ProcessingPipeline(toy_stack).add_filter("dummy")
    pipe.run()
    assert "raw" in toy_stack.processed
    np.testing.assert_array_equal(toy_stack.processed["raw"], np.ones((3, 4, 4)))


def test_run_updates_stack_data(toy_stack):
    """Test that run() updates the stack's data with the final result."""
    toy_stack._resolve_step.return_value = ("filter", lambda *a, **k: a[0])
    toy_stack._execute_filter_step.side_effect = (
        lambda fn, arr, mask, name, **kwargs: arr * 2
    )

    pipe = ProcessingPipeline(toy_stack).add_filter("double")
    pipe.run()

    np.testing.assert_array_equal(toy_stack.data, np.ones((3, 4, 4)) * 2)


def test_run_applies_mask_and_filter(toy_stack):
    """Test that run() applies both mask and filter steps correctly."""
    toy_stack._resolve_step.side_effect = [("mask", "mask_fn"), ("filter", "filter_fn")]

    toy_stack._execute_mask_step.return_value = np.ones((3, 4, 4), dtype=bool)
    toy_stack._execute_filter_step.side_effect = (
        lambda fn, arr, mask, name, **kwargs: arr + 10
    )

    pipe = ProcessingPipeline(toy_stack)
    pipe.add_mask("threshold", value=0.5)
    pipe.add_filter("smooth", radius=2)
    result = pipe.run()

    # Extract actual args
    args, kwargs = toy_stack._execute_mask_step.call_args
    assert args[0] == "mask_fn"
    np.testing.assert_array_equal(args[1], np.ones((3, 4, 4)))
    assert kwargs == {"value": 0.5}

    # Also check result shape and content
    assert result.shape == (3, 4, 4)
    assert (result == 11).all()

    np.testing.assert_array_equal(result, np.ones((3, 4, 4)) + 10)


def test_clear_mask_resets_mask(toy_stack):
    """Test that clear_mask resets the current mask to None."""
    toy_stack._resolve_step.side_effect = [
        ("mask", "mask_fn"),
        ("clear", None),
        ("filter", "filter_fn"),
    ]

    toy_stack._execute_mask_step.return_value = np.zeros((3, 4, 4), dtype=bool)
    toy_stack._execute_filter_step.side_effect = lambda fn, arr, mask, name, **kwargs: (
        arr + 2 if mask is None else arr
    )  # noqa: E501

    pipe = ProcessingPipeline(toy_stack)
    pipe.add_mask("mask1").clear_mask().add_filter("add_2")
    result = pipe.run()

    # The filter should be applied with mask = None after clearing
    assert result[0, 0, 0] == 3  # 1 (original) + 2 (filter)


def test_multiple_filters_chain(toy_stack):
    """Test that multiple filter steps are applied in sequence."""
    toy_stack._resolve_step.side_effect = [("filter", "f1"), ("filter", "f2")]

    first_out = np.ones((3, 4, 4)) * 2
    second_out = np.ones((3, 4, 4)) * 4

    toy_stack._execute_filter_step.side_effect = [first_out, second_out]

    pipe = ProcessingPipeline(toy_stack)
    pipe.add_filter("double1").add_filter("double2")
    result = pipe.run()

    np.testing.assert_array_equal(toy_stack.processed["step_1_double1"], first_out)
    np.testing.assert_array_equal(toy_stack.processed["step_2_double2"], second_out)
    np.testing.assert_array_equal(result, second_out)


def test_pipeline_preserves_step_order(toy_stack):
    """Test that steps are preserved in the order they were added."""
    pipe = ProcessingPipeline(toy_stack)
    pipe.add_mask("m1", level=1).clear_mask().add_filter("f1", sigma=2)

    expected_steps = [("m1", {"level": 1}), ("clear", {}), ("f1", {"sigma": 2})]
    assert pipe.steps == expected_steps


def test_run_does_not_override_raw_if_present(toy_stack):
    """Test that run() does not override 'raw' if it already exists."""
    toy_stack.processed["raw"] = np.zeros((3, 4, 4))
    toy_stack._resolve_step.return_value = ("filter", "f")
    toy_stack._execute_filter_step.side_effect = lambda *args, **kwargs: toy_stack.data

    pipe = ProcessingPipeline(toy_stack).add_filter("noop")
    pipe.run()
    # Ensure "raw" was untouched
    np.testing.assert_array_equal(toy_stack.processed["raw"], np.zeros((3, 4, 4)))


def make_toy_stack():
    """Create a simple 3D AFMImageStack for testing."""
    # A 3-frame, 4×4 stack with a simple pattern
    data = np.zeros((3, 4, 4), dtype=float)
    data[0] = np.arange(16).reshape(4, 4)
    data[1] = np.arange(16, 32).reshape(4, 4)
    data[2] = np.arange(32, 48).reshape(4, 4)
    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_path = Path(tmpdirname)
        # pixel_size_nm=1.0, channel=“h”, dummy metadata
        return AFMImageStack(data.copy(), 1.0, "h", temp_path, "dum")


def test_pipeline_eq_apply_simple_filter():
    """Test that a simple filter applied directly matches the pipeline output."""
    stack1 = make_toy_stack()
    # Direct apply
    out1 = stack1.apply(["remove_plane"])

    # Pipeline apply
    stack2 = make_toy_stack()
    pipeline = ProcessingPipeline(stack2)
    pipeline.add_filter("remove_plane")
    out2 = pipeline.run()

    assert np.allclose(out1, out2)
    # Ensure 'raw' snapshot exists in processed
    assert "raw" in stack2.processed
    assert "step_1_remove_plane" in stack2.processed


def test_pipeline_invalid_step_raises():
    """Test that an invalid step raises a ValueError."""
    stack = make_toy_stack()
    pipeline = ProcessingPipeline(stack)
    pipeline.add_filter("nonexistent_filter")
    with pytest.raises(ValueError):
        _ = pipeline.run()


def test_pipeline_combines_multiple_masks():
    """Test that pipeline combines masks without clear"""
    # Create dummy 3D data: 2 frames of 4x4
    data = np.zeros((2, 4, 4), dtype=bool)

    # Create fake mask outputs
    mask1 = np.zeros_like(data)
    mask1[:, 0:2, 0:2] = True  # top-left

    mask2 = np.zeros_like(data)
    mask2[:, 2:4, 2:4] = True  # bottom-right

    # Setup mock AFMImageStack
    stack = MagicMock(spec=AFMImageStack)
    stack.data = data.copy()
    stack.processed = {}
    stack.masks = {}

    def resolve_step_mock(name):
        """Return 'mask' type and dummy function for both mask steps"""
        return ("mask", lambda d, **kwargs: mask1 if name == "mask1" else mask2)

    stack._resolve_step.side_effect = resolve_step_mock
    stack._execute_mask_step.side_effect = lambda fn, d, **kwargs: fn(d, **kwargs)

    # Create pipeline and add two mask steps
    pipeline = ProcessingPipeline(stack)
    pipeline.add_mask("mask1").add_mask("mask2")

    # Run the pipeline
    pipeline.run()

    # Check if two masks are stored
    assert len(stack.masks) == 2

    # Check the last mask is a logical OR of both
    combined_key = list(stack.masks)[-1]
    combined_mask = stack.masks[combined_key]
    expected = np.logical_or(mask1, mask2)
    np.testing.assert_array_equal(combined_mask, expected)

    # Check naming pattern
    assert "mask2" in combined_key


def test_mask_overlay_fallback_name_without_error():
    """Test that if no previous mask is found when overlaying 'overlay' is used"""
    data = np.zeros((2, 4, 4), dtype=bool)

    # Initial mask
    mask1 = np.zeros_like(data)
    mask1[:, 0:2, 0:2] = True

    # Second mask to overlay
    mask2 = np.zeros_like(data)
    mask2[:, 2:4, 2:4] = True

    stack = MagicMock(spec=AFMImageStack)
    stack.data = data.copy()
    stack.processed = {}
    stack.masks = {}  # <- No previously saved masks

    def resolve_step_mock(name):
        """Mock resove step"""
        return ("mask", lambda d, **kwargs: mask1 if name == "mask1" else mask2)

    stack._resolve_step.side_effect = resolve_step_mock
    stack._execute_mask_step.side_effect = lambda fn, d, **kwargs: fn(d, **kwargs)

    pipeline = ProcessingPipeline(stack)
    pipeline.add_mask("mask1").add_mask("mask2")

    # Run and ensure no error occurs
    result = pipeline.run()

    assert result.shape == data.shape
    assert len(stack.masks) == 2
    mask_key = list(stack.masks)[-1]
    assert "overlay" in mask_key or "mask2" in mask_key


def test_mask_overlay_raises_value_error_if_previous_mask_missing():
    """Test that an value error is raised if previous mask isn't found"""
    data = np.zeros((2, 4, 4), dtype=bool)

    mask1 = np.zeros_like(data)
    mask2 = np.zeros_like(data)
    mask2[:, 2:4, 2:4] = True

    stack = MagicMock(spec=AFMImageStack)
    stack.data = data.copy()
    stack.processed = {}

    # Use a MagicMock instead of a real dict so we can mock __setitem__
    mock_masks = MagicMock()
    stack.masks = mock_masks

    def resolve_step_mock(name):
        return ("mask", lambda d, **kwargs: mask1 if name == "mask1" else mask2)

    stack._resolve_step.side_effect = resolve_step_mock
    stack._execute_mask_step.side_effect = lambda fn, d, **kwargs: fn(d, **kwargs)

    pipeline = ProcessingPipeline(stack)
    pipeline.add_mask("mask1").add_mask("mask2")

    def broken_mask_assign(key, value):
        """Simulate failure when attempting to assign fallback overlay mask"""
        if "overlay" in key:
            raise ValueError("Previous mask not accessible.")

    mock_masks.__setitem__.side_effect = broken_mask_assign

    # Should raise ValueError when fallback naming hits
    with pytest.raises(ValueError, match="Previous mask not accessible"):
        pipeline.run()
