"""Tests for the stack class."""

import logging
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np
import pytest

import playNano.afm_stack as afm_stack_module
from playNano.afm_stack import AFMImageStack, normalize_timestamps


def test_init_invalid_data_type():
    """Test that AFMImageStack raises TypeError for invalid data type."""
    with pytest.raises(TypeError):
        AFMImageStack(
            data=[[1, 2, 3]], pixel_size_nm=5.0, channel="height", file_path="dummy"
        )


def test_init_invalid_data_dim():
    """Test that AFMImageStack raises ValueError for invalid data dimensions."""
    data_2d = np.ones((10, 10))
    with pytest.raises(ValueError):
        AFMImageStack(
            data=data_2d, pixel_size_nm=5.0, channel="height", file_path="dummy"
        )


def test_init_invalid_pixel_size():
    """Test that AFMImageStack raises ValueError for invalid pixel size."""
    data = np.ones((5, 10, 10))
    with pytest.raises(ValueError):
        AFMImageStack(data=data, pixel_size_nm=0, channel="height", file_path="dummy")


def dummy_filter(image, **kwargs):
    """Filter dummy for testing."""
    return image + 1


patched_filters = {"dummy_filter": dummy_filter}


def test_normalize_timestamps_various_formats():
    """Convert ISO strings, datetimes, and numeric timestamps into floats or None."""
    md_list = [
        {"timestamp": "2025-05-20T12:00:00Z"},
        {"timestamp": datetime(2025, 5, 20, 12, 0, 1)},
        {"timestamp": 2.5},
        {"timestamp": None},
        {"timestamp": "not-a-date"},
    ]
    normalized = normalize_timestamps(md_list)
    assert isinstance(normalized[0]["timestamp"], float)
    assert isinstance(normalized[1]["timestamp"], float)
    assert normalized[2]["timestamp"] == 2.5
    assert normalized[3]["timestamp"] is None
    assert normalized[4]["timestamp"] is None


def test_metadata_padding_when_shorter():
    """Pad frame_metadata with empty dicts if shorter than data frames."""
    data = np.zeros((3, 4, 4))
    small_meta = [{"timestamp": 0.0}]
    stack = AFMImageStack(
        data=data,
        pixel_size_nm=1.0,
        channel="ch",
        file_path=".",
        frame_metadata=small_meta,
    )
    assert len(stack.frame_metadata) == 3
    assert stack.frame_metadata[1] == {"timestamp": None}
    assert stack.frame_metadata[2] == {"timestamp": None}


def test_metadata_error_when_longer():
    """Raise ValueError if frame_metadata length exceeds number of data frames."""
    data = np.zeros((2, 4, 4))
    long_meta = [{"timestamp": 0.0}, {"timestamp": 0.1}, {"timestamp": 0.2}]
    with pytest.raises(ValueError):
        AFMImageStack(
            data=data,
            pixel_size_nm=1.0,
            channel="ch",
            file_path=".",
            frame_metadata=long_meta,
        )


def test_get_frame_and_metadata():
    """Retrieve correct frame data and metadata by index."""
    arr = np.arange(9, dtype=float).reshape((1, 3, 3))
    meta = [{"timestamp": 5.0, "foo": "bar"}]
    stack = AFMImageStack(
        data=arr, pixel_size_nm=1.0, channel="ch", file_path=".", frame_metadata=meta
    )
    np.testing.assert_array_equal(stack.get_frame(0), arr[0])
    assert stack.get_frame_metadata(0) == {"timestamp": 5.0, "foo": "bar"}


def test_get_frame_metadata_index_error():
    """Raise IndexError when requesting metadata for out-of-range index."""
    data = np.zeros((1, 2, 2))
    stack = AFMImageStack(
        data=data, pixel_size_nm=1.0, channel="ch", file_path=".", frame_metadata=[{}]
    )
    with pytest.raises(IndexError):
        stack.get_frame_metadata(5)


def test_snapshot_raw_and_apply(monkeypatch):
    """Test that the raw data is snapshoted and functions applied."""
    data = np.ones((2, 2, 2))
    stack = AFMImageStack(
        data=data.copy(),
        pixel_size_nm=1.0,
        channel="ch",
        file_path=".",
        frame_metadata=[{}, {}],
    )

    # Define a trivial processing function that doubles every pixel
    def double(arr, **kwargs):
        return arr * 2

    # Monkeypatch the FILTER_MAP to include our custom filter
    monkeypatch.setitem(afm_stack_module.FILTER_MAP, "doubled", double)
    monkeypatch.setitem(afm_stack_module.FILTER_MAP, "quadrupled", double)

    # Apply the "doubled" filter
    stack.apply(["doubled"])
    assert "raw" in stack.processed
    np.testing.assert_array_equal(stack.processed["raw"], data)
    np.testing.assert_array_equal(stack.processed["doubled"], data * 2)
    np.testing.assert_array_equal(stack.data, data * 2)

    # Apply the "quadrupled" filter (doubles again)
    stack.apply(["quadrupled"])
    np.testing.assert_array_equal(stack.processed["raw"], data)
    np.testing.assert_array_equal(stack.processed["doubled"], data * 2)
    np.testing.assert_array_equal(stack.processed["quadrupled"], data * 4)
    np.testing.assert_array_equal(stack.data, data * 4)


def test_flatten_images_uses_apply(monkeypatch):
    """Ensure applying 'topostats_flatten' updates data and stores result."""
    # 1) Create a tiny 2×2×2 “stack of ones”
    data = np.ones((2, 2, 2))
    stack = AFMImageStack(
        data=data.copy(),
        pixel_size_nm=1.0,
        channel="ch",
        file_path=".",
        frame_metadata=[{}, {}],
    )

    # 2) What we expect after “flatten”: every pixel = 7.0
    fake_flat = np.full_like(data, 7.0)

    # 3) Patch out any possibility of loading a real plugin:
    #    Make AFMImageStack._load_plugin(...) return None so
    # the code falls back to FILTER_MAP.
    monkeypatch.setattr(
        "playNano.afm_stack.AFMImageStack._load_plugin", lambda self, name: None
    )

    # 4) Now override the module‐level FILTER_MAP entry for "topostats_flatten"
    # patch _reslve_step to directly return our fake function
    monkeypatch.setattr(
        stack,
        "_resolve_step",
        lambda step: ("filter", lambda frame, **kwargs: np.full_like(frame, 7.0)),
    )

    # 5) Call apply([...]) – now it must pick up our fake function from FILTER_MAP
    out = stack.apply(["topostats_flatten"])

    # 6) Now it should be equal to fake_flat
    np.testing.assert_array_equal(out, fake_flat)

    # 7) Also verify that stack.processed["topostats_flatten"] was set to fake_flat
    assert "topostats_flatten" in stack.processed
    np.testing.assert_array_equal(stack.processed["topostats_flatten"], fake_flat)


def test_frames_with_metadata_iterator():
    """Yield correct (index, frame, metadata) tuples in sequence."""
    arr = np.array([[[1]], [[2]]])
    meta = [{"a": 1}, {"b": 2}]
    stack = AFMImageStack(
        data=arr, pixel_size_nm=1.0, channel="ch", file_path=".", frame_metadata=meta
    )
    results = list(stack.frames_with_metadata())
    assert results == [
        (0, arr[0], {"a": 1, "timestamp": None}),
        (1, arr[1], {"b": 2, "timestamp": None}),
    ]


def test_normalize_timestamps_mixed():
    """Convert various timestamp formats into floats or None."""
    md_list = [
        {"timestamp": "2025-05-20T12:00:00Z"},
        {"timestamp": datetime(2025, 5, 20, 12, 0, 1)},
        {"timestamp": 2.5},
        {"timestamp": None},
        {"timestamp": "invalid"},
    ]
    normalized = normalize_timestamps(md_list)
    assert isinstance(normalized[0]["timestamp"], float)
    assert isinstance(normalized[1]["timestamp"], float)
    assert normalized[2]["timestamp"] == 2.5
    assert normalized[3]["timestamp"] is None
    assert normalized[4]["timestamp"] is None


def test_getitem_single_frame():
    """Test that __getitem__ returns a single frame as a numpy array."""
    data = np.arange(27).reshape(3, 3, 3).astype(float)
    stack = AFMImageStack(data.copy(), 1.0, "height", "dummy")
    frame1 = stack[1]
    assert isinstance(frame1, np.ndarray)
    assert frame1.shape == (3, 3)
    assert np.allclose(frame1, data[1])


def test_getitem_slice_creates_new_stack():
    """Test that __getitem__ with a slice returns a new AFMImageStack."""
    data = np.random.rand(4, 5, 5)
    stack = AFMImageStack(data.copy(), 2.0, "height", "dummy")
    substack = stack[1:3]
    assert isinstance(substack, AFMImageStack)
    assert substack.n_frames == 2
    assert np.allclose(substack.data, data[1:3])
    # Check metadata length matches too:
    assert len(substack.frame_metadata) == 2


def test_getitem_invalid_index():
    """Test that __getitem__ raises IndexError for invalid index."""
    data = np.random.rand(2, 2, 2)
    stack = AFMImageStack(data.copy(), 1.0, "height", "dummy")
    with pytest.raises(TypeError):
        _ = stack["not_an_int_or_slice"]


def test_restore_raw(monkeypatch):
    """
    Test that restore_raw resets self.data to "raw".

    Test that the restore_raw method correctly resets self.data
    to the original raw snapshot stored in self.processed['raw'].

    - Applies a simple processing function (doubling pixel values)
    - Checks that processed data is different from original
    - Calls restore_raw and checks data matches the original again
    """
    data = np.ones((2, 2, 2))
    stack = AFMImageStack(data.copy(), 1.0, "ch", ".", [{}] * 2)

    def double(arr):
        return arr * 2

    monkeypatch.setitem(afm_stack_module.FILTER_MAP, "doubled", double)

    stack.apply(["doubled"])
    assert np.all(stack.data == 2)

    restored = stack.restore_raw()
    assert np.all(restored == 1)
    assert np.all(stack.data == 1)


class DummyStack:
    """Dummy class that delegates filter execution to AFMImageStack."""

    def _execute_filter_step(self, filter_fn, arr, mask, step_name, **kwargs):
        """Execute a filter step using AFMImageStack's implementation."""
        return AFMImageStack._execute_filter_step(
            self, filter_fn, arr, mask, step_name, **kwargs
        )


@pytest.fixture
def arr_and_mask():
    """Fixture that returns a sample array and corresponding boolean mask."""
    arr = np.ones((2, 3, 3), dtype=float)
    mask = np.zeros_like(arr, dtype=bool)
    return arr, mask


def test_masked_filter_success(arr_and_mask):
    """Test successful application of a masked filter function."""
    arr, mask = arr_and_mask
    masked_fn = Mock(return_value=np.zeros((3, 3)))

    with patch("playNano.afm_stack.MASK_FILTERS_MAP", {"dummy": masked_fn}):
        out = DummyStack()._execute_filter_step(None, arr, mask, "dummy")
        assert np.all(out == 0)
        assert masked_fn.call_count == 2


def test_masked_filter_typeerror_fallback(arr_and_mask):
    """Test fallback behavior when masked filter raises TypeError."""
    arr, mask = arr_and_mask

    def fail_on_kwargs(frame, m, **kwargs):  # noqa
        raise TypeError("ignore kwargs")

    fallback_fn = Mock(return_value=np.ones((3, 3)))
    fallback_wrapper = Mock(
        side_effect=[
            TypeError("ignore"),
            fallback_fn(arr[0], mask[0]),
            fallback_fn(arr[1], mask[1]),
        ]
    )

    with patch("playNano.cli.utils.MASK_FILTERS_MAP", {"dummy": fallback_wrapper}):
        out = DummyStack()._execute_filter_step(None, arr, mask, "dummy", foo=42)
        assert np.all(out == 1)


def test_masked_filter_fallback_on_error(arr_and_mask):
    """Test fallback to original array when masked filter raises an error."""
    arr, mask = arr_and_mask

    def always_fail(*args, **kwargs):
        raise ValueError("bad frame")

    with patch("playNano.cli.utils.MASK_FILTERS_MAP", {"dummy": always_fail}):
        out = DummyStack()._execute_filter_step(None, arr, mask, "dummy")
        assert np.all(out == arr)  # fallback to original


def test_unmasked_filter_success(arr_and_mask):
    """Test successful application of an unmasked filter function."""
    arr, _ = arr_and_mask
    fn = Mock(return_value=np.full((3, 3), 7))
    out = DummyStack()._execute_filter_step(fn, arr, None, "noop")
    assert np.all(out == 7)


def test_unmasked_filter_typeerror_fallback(arr_and_mask):
    """Test fallback behavior when unmasked filter raises TypeError."""
    arr, _ = arr_and_mask

    def fail_kwargs(frame, **kwargs):
        raise TypeError("bad kwargs")

    fallback = Mock(return_value=np.ones((3, 3)))

    wrapped = Mock(side_effect=[TypeError("bad"), fallback(arr[0]), fallback(arr[1])])
    out = DummyStack()._execute_filter_step(wrapped, arr, None, "noop", bad=True)
    assert np.all(out == 1)


def test_unmasked_filter_fallback_on_error(arr_and_mask):
    """Test fallback to original array when unmasked filter raises an error."""
    arr, _ = arr_and_mask

    def fail(frame, **kwargs):
        raise RuntimeError("oops")

    out = DummyStack()._execute_filter_step(fail, arr, None, "noop")
    assert np.all(out == arr)


def test_masked_filter_typeerror_then_exception(caplog, arr_and_mask):
    """Test fallback & log when masked filter raises TypeError & another exception."""
    arr, mask = arr_and_mask

    # Raise TypeError first, then ValueError on second attempt
    def faulty_fn(a, m):
        if isinstance(m, np.ndarray) and m.shape == a.shape:
            raise ValueError("Deliberate failure after fallback")
        raise TypeError("Deliberate TypeError")

    with patch("playNano.afm_stack.MASK_FILTERS_MAP", {"dummy": faulty_fn}):
        with caplog.at_level(logging.ERROR):
            out = DummyStack()._execute_filter_step(None, arr, mask, "dummy")

    # Output should fallback to original array
    assert np.all(out == arr)
    # Logs should show fallback error message
    assert "failed on frame 0" in caplog.text
    assert "Deliberate failure after fallback" in caplog.text


def test_masked_filter_general_exception(caplog, arr_and_mask):
    """Test fallback and logging when masked filter raises a general exception."""
    arr, mask = arr_and_mask

    # Immediately raise some other exception
    def faulty_fn(a, m):
        raise RuntimeError("Immediate failure")

    with patch("playNano.afm_stack.MASK_FILTERS_MAP", {"dummy": faulty_fn}):
        with caplog.at_level(logging.ERROR):
            out = DummyStack()._execute_filter_step(None, arr, mask, "dummy")

    assert np.all(out == arr)
    assert "Masked filter 'dummy' failed on frame 0" in caplog.text
    assert "Immediate failure" in caplog.text


# --- Fixtures for AFMImageStack with time metadata ---


@pytest.fixture
def stack_with_times():
    """AFMImageStack with explicit and implicit timestamps."""
    # Create small data and metadata
    data = np.zeros((4, 2, 2), dtype=float)
    # frame_metadata: first has timestamp 0.0, second missing, third 2.5, fourth missing
    meta = [{"timestamp": 0.0}, {}, {"timestamp": 2.5}, {}]
    # Use TemporaryDirectory for file_path
    with TemporaryDirectory() as td:
        stack = AFMImageStack(
            data.copy(),
            pixel_size_nm=1.0,
            channel="h",
            file_path=Path(td),
            frame_metadata=meta,
        )
        yield stack


# --- Tests for AFMImageStack time methods ---


def test_time_for_frame_with_and_without_timestamp(stack_with_times):
    """time_for_frame should return timestamp or index as float."""
    stack = stack_with_times
    assert stack.time_for_frame(0) == 0.0
    # missing timestamp: fallback to index
    assert stack.time_for_frame(1) == 1.0
    assert pytest.approx(stack.time_for_frame(2)) == 2.5
    assert stack.time_for_frame(3) == 3.0


def test_get_frame_times(stack_with_times):
    """get_frame_times should return list of 4 floats with fallbacks."""
    stack = stack_with_times
    times = stack.get_frame_times()
    assert isinstance(times, list) and len(times) == 4
    assert times == [0.0, 1.0, 2.5, 3.0]
