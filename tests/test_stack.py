"""Tests for the stack class."""

from datetime import datetime

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
    assert stack.frame_metadata[1] == {}
    assert stack.frame_metadata[2] == {}


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
    assert results == [(0, arr[0], {"a": 1}), (1, arr[1], {"b": 2})]


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
