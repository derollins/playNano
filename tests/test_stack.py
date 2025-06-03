"""Tests for the stack class."""

from datetime import datetime

import numpy as np
import pytest

from playNano.stack.afm_stack import FILTER_MAP, AFMImageStack, normalize_timestamps


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
    monkeypatch.setitem(FILTER_MAP, "doubled", double)
    monkeypatch.setitem(FILTER_MAP, "quadrupled", double)

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
    import numpy as np

    from playNano.stack.afm_stack import AFMImageStack

    data = np.ones((2, 2, 2))
    stack = AFMImageStack(
        data=data.copy(),
        pixel_size_nm=1.0,
        channel="ch",
        file_path=".",
        frame_metadata=[{}, {}],
    )

    # Create a fake flattened array
    fake_flat = np.full_like(data, fill_value=7.0)

    # Patch the FILTER_MAP to simulate the flattening filter
    def fake_flatten(frame, **kwargs):
        return np.full_like(frame, 7.0)

    monkeypatch.setitem(
        stack.__class__.__dict__["apply"].__globals__["FILTER_MAP"],
        "topostats_flatten",
        fake_flatten,
    )

    out = stack.apply(["topostats_flatten"])
    np.testing.assert_array_equal(out, fake_flat)
    assert "flattened" in stack.processed or "topostats_flatten" in stack.processed
    np.testing.assert_array_equal(stack.data, fake_flat)


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

    monkeypatch.setitem(FILTER_MAP, "doubled", double)

    stack.apply(["doubled"])
    assert np.all(stack.data == 2)

    restored = stack.restore_raw()
    assert np.all(restored == 1)
    assert np.all(stack.data == 1)
