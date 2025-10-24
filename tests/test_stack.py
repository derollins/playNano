"""Unit tests for AFMImageStack class and related timestamp/filter/mask utilities."""

import json
import logging
import types
from datetime import datetime
from importlib import metadata
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np
import pytest

import playNano.afm_stack as afm_stack_module
from playNano.afm_stack import AFMImageStack, normalize_timestamps

logger = logging.getLogger(__name__)


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


def test_get_frames_returns_all_frames(stack_with_times):
    """Test get_frames returns a list of all 2D frames using get_frame method."""
    # Assuming n_frames property is derived from self.data.shape[0]
    stack_with_times.data = np.zeros((3, 10, 10))  # 3 frames, each 10x10

    # Mock get_frame to return corresponding frames from self.data
    def mock_get_frame(i):
        return stack_with_times.data[i]

    stack_with_times.get_frame = mock_get_frame

    frames = stack_with_times.get_frames()

    assert isinstance(frames, list)
    assert len(frames) == 3

    for i, frame in enumerate(frames):
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (10, 10)
        # Check frame matches the underlying data slice
        assert np.array_equal(frame, stack_with_times.data[i])


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


def test_snapshot_raw_creates_copy_and_is_idempotent(stack_with_times):
    """Test _snapshot_raw copies data on first call and does nothing on later calls."""
    # Setup initial data and empty processed dict
    stack_with_times.data = [1, 2, 3, 4]  # can be list or numpy array
    stack_with_times.processed = {}

    # Call method first time - should create 'raw' copy
    stack_with_times._snapshot_raw()

    assert "raw" in stack_with_times.processed
    # The stored 'raw' should equal the original data but be a different object (copy)
    assert stack_with_times.processed["raw"] == stack_with_times.data
    assert stack_with_times.processed["raw"] is not stack_with_times.data

    # Change original data and call _snapshot_raw again
    if hasattr(stack_with_times.data, "append"):
        stack_with_times.data.append(5)  # if list
    else:
        import numpy as np

        stack_with_times.data = np.append(stack_with_times.data, 5)

    # Call again, should do nothing
    stack_with_times._snapshot_raw()

    # The 'raw' stored data should remain unchanged (no 5 appended)
    assert stack_with_times.processed["raw"] != stack_with_times.data
    assert 5 not in stack_with_times.processed["raw"]


def test_channel_for_frame_with_and_without_override():
    """Test that channel_for_frame returns per-frame channel if present, else global."""
    data = np.zeros((2, 2, 2), dtype=float)
    meta = [
        {"channel": "frame-specific"},  # overrides
        {},  # uses global
    ]

    stack = AFMImageStack(
        data=data,
        pixel_size_nm=1.0,
        channel="global-channel",
        file_path="dummy",
        frame_metadata=meta,
    )

    assert stack.channel_for_frame(0) == "frame-specific"
    assert stack.channel_for_frame(1) == "global-channel"


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


def test_get_plugin_version_known_module():
    """Test _get_plugin_version returns version for a standard library or package."""
    # Use a known function from numpy
    import numpy as np

    version = AFMImageStack._get_plugin_version(np.mean)
    assert isinstance(version, str)
    assert version == metadata.version("numpy")


def test_get_plugin_version_fake_module(monkeypatch):
    """Test _get_plugin_version returns None for non-existent module."""
    # Create a fake function with a fake module
    fake_fn = lambda x: x  # noqa
    fake_fn.__module__ = "nonexistent_fake_package.sub"

    version = AFMImageStack._get_plugin_version(fake_fn)
    assert version is None


def test_get_plugin_version_error(monkeypatch):
    """Test _get_plugin_version handles unexpected exceptions."""
    # Create a function that pretends to be from a real module
    fn = lambda x: x  # noqa
    fn.__module__ = "numpy"

    # Patch metadata.version to raise a generic error
    monkeypatch.setattr(
        metadata, "version", lambda _: (_ for _ in ()).throw(Exception("boom"))
    )

    version = AFMImageStack._get_plugin_version(fn)
    assert version is None


def test_load_plugin(monkeypatch):
    """Test _load_plugin loads a valid plugin and raises error for unknown plugin."""
    data = np.ones((2, 2, 2))
    stack = AFMImageStack(data.copy(), 1.0, "ch", ".", [{}] * 2)

    # Patch metadata.entry_points to mock a plugin
    fake_ep = Mock()
    fake_ep.name = "dummy"
    fake_ep.value = "some.module:dummy"
    fake_ep.load = Mock(return_value=lambda x: x + 1)

    monkeypatch.setattr(
        metadata,
        "entry_points",
        lambda group=None: [fake_ep] if group == "playNano.filters" else [],
    )

    plugin_fn = stack._load_plugin("dummy")
    assert callable(plugin_fn)
    assert plugin_fn(1) == 2

    # Check error for unknown plugin
    with pytest.raises(ValueError):
        stack._load_plugin("not_exist")


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


def test_restore_raw_raises_keyerror_when_missing():
    """Test that restore_raw raises KeyError if no 'raw' snapshot exists."""
    data = np.zeros((2, 2, 2), dtype=float)
    stack = AFMImageStack(
        data=data,
        pixel_size_nm=1.0,
        channel="test",
        file_path="dummy",
        frame_metadata=[{}, {}],
    )

    # Ensure 'raw' is not in processed
    assert "raw" not in stack.processed

    with pytest.raises(KeyError, match="No raw data snapshot available to restore."):
        stack.restore_raw()


class DummyStack:
    """Dummy class that delegates processing steps to AFMImageStack."""

    def __init__(self):
        """Initiate the dummy class."""
        self.frame_metadata = [{"frame": i} for i in range(3)]
        self.state_backups = {}

    def _execute_filter_step(self, filter_fn, arr, mask, step_name, **kwargs):
        """Execute a filter step using AFMImageStack's implementation."""
        return AFMImageStack._execute_filter_step(
            self, filter_fn, arr, mask, step_name, **kwargs
        )

    def _execute_stack_edit_step(self, stack_edit_fn, arr, **kwargs):
        """Execute a stack edit step using AFMImageStack's implementation."""
        return AFMImageStack._execute_stack_edit_step(
            self, stack_edit_fn, arr, **kwargs
        )

    def _execute_video_processing_step(self, video_filter_fn, arr, **kwargs):
        """Execute a video filter step using AFMImageStack's implementation."""
        return AFMImageStack._execute_video_processing_step(
            self, video_filter_fn, arr, **kwargs
        )


def test_execute_video_processing_success():
    """Test that a video processing function successfully modifies the image stack."""
    arr = np.ones((5, 10, 10))
    video_fn = Mock(return_value=arr * 2)
    stack = DummyStack()
    result = stack._execute_video_processing_step(video_fn, arr)
    assert np.array_equal(result, arr * 2)


def test_execute_video_processing_typeerror_then_success():
    """Test fallback execution of video function w/o kwargs when TypeError raised."""
    arr = np.ones((5, 10, 10))

    def video_fn(arr):
        return arr * 3

    stack = DummyStack()
    result = stack._execute_video_processing_step(video_fn, arr)
    assert np.array_equal(result, arr * 3)


def test_execute_video_processing_failure():
    """Test original array is returned when the video function raises an exception."""
    arr = np.ones((5, 10, 10))

    def video_fn(arr, **kwargs):
        raise ValueError("fail")

    stack = DummyStack()
    result = stack._execute_video_processing_step(video_fn, arr)
    assert np.array_equal(result, arr)


def test_execute_stack_edit_step_successful_edit():
    """Test successful structural edit and metadata update with frame removal."""
    arr = np.ones((5, 10, 10))
    new_arr = arr[:3]

    def edit_fn(arr, **kwargs):
        return arr[:3]

    stack = DummyStack()
    stack.frame_metadata = ["f0", "f1", "f2", "f3", "f4"]
    stack.state_backups = {}

    result = stack._execute_stack_edit_step(edit_fn, arr, indices_to_drop=[3, 4])
    assert np.array_equal(result, new_arr)
    assert stack.frame_metadata == ["f0", "f1", "f2"]
    assert stack.state_backups["frame_metadata_before_edit"] == [
        "f0",
        "f1",
        "f2",
        "f3",
        "f4",
    ]


def test_execute_stack_edit_step_metadata_mismatch():
    """Test a RuntimeError is raised when metadata length mismatches edited array."""
    arr = np.ones((5, 10, 10))

    def edit_fn(arr, **kwargs):
        return arr[:2]  # mismatch with metadata

    stack = DummyStack()
    stack.frame_metadata = ["f0", "f1", "f2", "f3", "f4"]
    stack.state_backups = {}

    with pytest.raises(RuntimeError, match="frame_metadata length mismatch"):
        stack._execute_stack_edit_step(edit_fn, arr, indices_to_drop=[3, 4])


def test_execute_video_processing_typeerror_then_fallback_failure():
    """Test that original array is returned when both video_fn calls fail."""
    arr = np.ones((5, 10, 10))

    class DualFail:
        def __init__(self):
            self.called_with_kwargs = False

        def __call__(self, arr, **kwargs):
            if kwargs:
                self.called_with_kwargs = True
                raise TypeError("unexpected kwargs")
            else:
                raise ValueError("still fails")

        @property
        def __name__(self):
            return "dual_fail"

    stack = DummyStack()
    video_fn = DualFail()
    result = stack._execute_video_processing_step(video_fn, arr, unexpected_kwarg=True)
    assert np.array_equal(result, arr)


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


def test_execute_filter_step_masked_fn_raises_inner_exception(monkeypatch, caplog):
    """Verify errors logged and original frames returned when masked filter fails."""
    # Setup dummy input data
    n_frames, H, W = 2, 3, 3
    arr = np.arange(n_frames * H * W).reshape((n_frames, H, W))
    mask = np.ones_like(arr, dtype=bool)

    call_count = {"count": 0}

    def masked_fn_wrapper(frame, mask_arg, **kwargs):
        if call_count["count"] == 0:
            call_count["count"] += 1
            raise TypeError("Expected TypeError with kwargs")
        else:
            raise ValueError("Expected ValueError without kwargs")

    test_step_name = "test_step"
    from playNano.afm_stack import MASK_FILTERS_MAP, AFMImageStack

    MASK_FILTERS_MAP[test_step_name] = masked_fn_wrapper

    class TestClass:
        _execute_filter_step = AFMImageStack._execute_filter_step

    obj = TestClass()

    with caplog.at_level(logging.ERROR):
        output = obj._execute_filter_step(
            filter_fn=None,
            arr=arr,
            mask=mask,
            step_name=test_step_name,
        )

    # --- Assertions ---

    # 1. Check output shape and type
    assert isinstance(output, np.ndarray)
    assert output.shape == arr.shape

    # 2. Because the masked filter fails, the output frames should equal original frames
    #    (since the fallback copies original frame when filter fails)
    for i in range(n_frames):
        np.testing.assert_array_equal(output[i], arr[i])

    # 3. Check that error logs contain the expected messages for each failed frame
    error_logs = [
        record.message for record in caplog.records if record.levelname == "ERROR"
    ]  # noqa
    expected_error_msg = f"Masked filter '{test_step_name}' failed on frame 0: Expected ValueError without kwargs"  # noqa
    assert any(expected_error_msg in msg for msg in error_logs)

    expected_error_msg_1 = f"Masked filter '{test_step_name}' failed on frame 1: Expected ValueError without kwargs"  # noqa
    assert any(expected_error_msg_1 in msg for msg in error_logs)

    # Cleanup test entry from MASK_FILTERS_MAP
    del MASK_FILTERS_MAP[test_step_name]


def test_execute_mask_step_direct_exception(caplog):
    """Test _execute_mask_step logs error and returns False mask on direct exception."""
    # Input: 2-frame dummy array
    arr = np.ones((2, 3, 3))

    # Define a mock mask_fn that raises ValueError immediately (not TypeError)
    def faulty_mask_fn(frame, **kwargs):
        raise ValueError("Direct failure with kwargs")

    faulty_mask_fn.__name__ = "faulty_mask_fn"

    # Dummy instance with _execute_mask_step
    from playNano.afm_stack import AFMImageStack

    class Dummy:
        _execute_mask_step = AFMImageStack._execute_mask_step

    obj = Dummy()

    # Run the mask step
    with caplog.at_level(logging.ERROR):
        mask = obj._execute_mask_step(faulty_mask_fn, arr, kwarg1=True)

    # Assert shape and type
    assert isinstance(mask, np.ndarray)
    assert mask.shape == arr.shape
    assert not np.any(mask)  # all False since it failed

    # Logs should reflect failure from the outer except block
    errors = [rec.message for rec in caplog.records if rec.levelno == logging.ERROR]
    assert any(
        "Mask generator 'faulty_mask_fn' failed on frame 0" in msg for msg in errors
    )
    assert any(
        "Mask generator 'faulty_mask_fn' failed on frame 1" in msg for msg in errors
    )


@patch("playNano.afm_stack.MASK_MAP", {"mask_dummy": lambda frame, **kwargs: frame > 0})
@patch(
    "playNano.afm_stack.FILTER_MAP", {"filter_dummy": lambda frame, **kwargs: frame + 2}
)
def test_apply_clear_mask_filter_sequence(monkeypatch):
    """Test 'clear' step resets mask, 'mask' sets new mask, 'filter' applies it."""
    arr = np.ones((2, 3, 3))
    stack = AFMImageStack(arr.copy(), 1.0, "h", ".", [{}] * 2)

    # Spy on private methods
    with (
        patch.object(
            stack, "_execute_mask_step", wraps=stack._execute_mask_step
        ) as mock_mask,
        patch.object(
            stack, "_execute_filter_step", wraps=stack._execute_filter_step
        ) as mock_filter,
    ):
        result = stack.apply(["clear", "mask_dummy", "filter_dummy"])

        assert mock_mask.called
        assert mock_filter.called

        # Output should reflect +2 applied to all (from dummy filter)
        np.testing.assert_array_equal(result, arr + 2)
        assert "filter_dummy" in stack.processed


def test_apply_clear_does_not_crash_and_skips_processing(monkeypatch):
    """Test that 'clear' step does not crash and mask is reset."""
    arr = np.ones((2, 2, 2))
    stack = AFMImageStack(arr.copy(), 1.0, "h", ".", [{}] * 2)

    with (
        patch.object(stack, "_execute_mask_step") as mask_mock,
        patch.object(stack, "_execute_filter_step") as filter_mock,
    ):
        stack.apply(["clear"])

        mask_mock.assert_not_called()
        filter_mock.assert_not_called()


def test_execute_mask_step_typeerror_then_failure(caplog):
    """Test that execute_mask_step raises typeError and fails."""
    # Input: a 3D array (2 frames, 3x3 each)
    arr = np.ones((2, 3, 3))

    # Create a mock mask_fn that raises:
    # - TypeError when called with kwargs
    # - ValueError when called without kwargs
    call_state = {"calls": 0}

    def mock_mask_fn(frame, **kwargs):
        call_state["calls"] += 1
        raise TypeError("Expected TypeError with kwargs")

    def mock_mask_fn_no_kwargs(frame):
        raise ValueError("Fails even without kwargs")

    # Combined function to simulate both paths
    def mock_mask_fn_combined(frame, **kwargs):
        if kwargs:
            raise TypeError("Expected TypeError with kwargs")
        raise ValueError("Fails even without kwargs")

    # Add __name__ to the mock function (used in logger)
    mock_mask_fn_combined.__name__ = "mock_mask_fn_combined"

    # Patch _execute_mask_step into a dummy class instance
    from playNano.afm_stack import AFMImageStack

    class Dummy:
        _execute_mask_step = AFMImageStack._execute_mask_step

    obj = Dummy()

    with caplog.at_level(logging.ERROR):
        mask = obj._execute_mask_step(mock_mask_fn_combined, arr, some_kwarg=True)

    # Verify shape and type
    assert isinstance(mask, np.ndarray)
    assert mask.shape == arr.shape
    assert mask.dtype == bool

    # All values should be False since the function failed
    assert not np.any(mask)

    # Check logs contain expected error messages
    errors = [rec.message for rec in caplog.records if rec.levelno == logging.ERROR]
    assert any(
        "Mask generator 'mock_mask_fn_combined' failed on frame 0" in e for e in errors
    )
    assert any(
        "Mask generator 'mock_mask_fn_combined' failed on frame 1" in e for e in errors
    )


# --- Fixtures for AFMImageStack with time metadata ---


@pytest.fixture
def stack_with_times():
    """Test AFMImageStack with explicit and implicit timestamps."""
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


# --- Export logs ---


def test_export_analysis_log(tmp_path, monkeypatch):
    """Test export_analysis_log raises errors or writes log with analysis results."""
    # Create minimal AFMImageStack instance with required data shape
    data = np.zeros((1, 2, 2))
    stack = AFMImageStack(
        data=data,
        pixel_size_nm=1.0,
        channel="ch",
        file_path=str(tmp_path),
        frame_metadata=[{}],
    )

    # 1) No analysis_results attribute → should raise ValueError
    if hasattr(stack, "analysis_results"):
        delattr(stack, "analysis_results")
    with pytest.raises(ValueError):
        stack.export_analysis_log(str(tmp_path / "some_path.json"))

    # 2) Empty analysis_results → should raise ValueError
    stack.analysis_results = {}
    with pytest.raises(ValueError):
        stack.export_analysis_log(str(tmp_path / "some_path.json"))

    # 3) Provide valid analysis_results, patch NumpyEncoder
    stack.analysis_results = {
        "environment": {"python": "3.10"},
        "analysis": {"step1": {"result": 42}},
        "provenance": {
            "steps": ["step1"],
            "results_by_name": {"step1": {"result": 42}},
            "frame_times": [0.0, 1.0],
        },
    }

    monkeypatch.setattr("playNano.analysis.utils.common.NumpyEncoder", json.JSONEncoder)

    nested_path = tmp_path / "subdir" / "log.json"
    stack.export_analysis_log(str(nested_path))

    assert nested_path.exists()

    with open(nested_path) as f:
        loaded = json.load(f)
    assert loaded == stack.analysis_results


def test_export_processing_log(tmp_path, monkeypatch):
    """Test export_processing_log writes JSON file with correct provenance data."""

    class DummyStack:
        def __init__(self):
            self.provenance = {
                "environment": {"python": "3.10"},
                "processing": {
                    "steps": ["step1", "step2"],
                    "keys_by_name": {"step1": {"param": 1}},
                },
            }


def test_export_processing_log_creates_json_file(tmp_path, stack_with_times):
    """Test export_processing_log makes correct JSON output from mocked provenance."""
    dummy_env = {"python_version": "3.10"}
    dummy_processing = {
        "steps": ["filter", "segment"],
        "keys_by_name": {"filter": "gauss"},
    }

    # Create a mock object to act as `stack_with_times.stack` with a provenance dict
    mock_stack = types.SimpleNamespace(
        provenance={
            "environment": dummy_env,
            "processing": dummy_processing,
        }
    )

    # Attach this mock_stack as .stack attribute on stack_with_times
    stack_with_times.stack = mock_stack

    log_file = tmp_path / "logs" / "processing_log.json"
    stack_with_times.export_processing_log(str(log_file))

    assert log_file.exists()
    with open(log_file, "r") as f:
        data = json.load(f)

    assert data["environment"] == dummy_env
    assert data["processing"] == dummy_processing


@pytest.fixture
def dummy_stack(tmp_path):
    """Create dummy stack for testing."""
    data = np.random.rand(3, 4, 4)
    meta = [{"timestamp": float(i)} for i in range(3)]
    return AFMImageStack(
        data=data,
        pixel_size_nm=1.0,
        channel="height",
        file_path=tmp_path,
        frame_metadata=meta,
    )


def test_export_processing_log_creates_file_and_dir(tmp_path, dummy_stack):
    """Test that export_processng_log creates file and folder."""
    dummy_stack.provenance["processing"]["steps"].append({"step": "dummy"})
    dummy_stack.provenance["environment"] = {"python": "3.11"}

    # Create nested path
    log_path = tmp_path / "logs" / "proc.json"
    dummy_stack.stack = dummy_stack  # simulate self.stack access inside method
    dummy_stack.export_processing_log(str(log_path))

    assert log_path.exists()
    with open(log_path) as f:
        content = json.load(f)
    assert "processing" in content
    assert "environment" in content


def test_restore_raw_missing_key_raises(dummy_stack):
    """Test that restore_raw raises error when missing key."""
    dummy_stack.processed.pop("raw", None)
    with pytest.raises(KeyError, match="No raw data snapshot available"):
        dummy_stack.restore_raw()


def test_frames_with_metadata_skips_none_frame(caplog):
    """Test that frames with None are skipped."""
    # Step 1: create a valid AFMImageStack
    data = np.stack([np.ones((2, 2)), np.ones((2, 2)), np.zeros((2, 2))])
    meta = [{"timestamp": 0.0}, {"timestamp": 1.0}, {"timestamp": 2.0}]
    stack = AFMImageStack(
        data=data,
        pixel_size_nm=1.0,
        channel="height",
        file_path="dummy",
        frame_metadata=meta,
    )

    # Step 2: overwrite .data with object array containing real None
    obj_data = np.empty(3, dtype=object)
    obj_data[0] = np.ones((2, 2))
    obj_data[1] = None  # <-- this will trigger the warning
    obj_data[2] = np.zeros((2, 2))
    stack.data = obj_data  # bypass validation — OK for this test

    # Step 3: trigger the method
    caplog.set_level("WARNING", logger="playNano.afm_stack")
    results = list(stack.frames_with_metadata())

    # Step 4: assert warning was issued and frame 1 was skipped
    assert len(results) == 2
    assert results[0][0] == 0
    assert results[1][0] == 2
    assert "Frame 1 is None and skipped" in caplog.text


def test_resolve_step_method_returns_bound_method():
    """Test that resolve_step method returns method."""
    # Set up a basic stack
    stack = AFMImageStack(
        data=np.ones((2, 4, 4)),
        pixel_size_nm=1.0,
        channel="height",
        file_path="dummy",
        frame_metadata=[{"timestamp": 0.0}, {"timestamp": 1.0}],
    )

    # 'get_frame' is a real method on AFMImageStack
    step_type, fn = stack._resolve_step("get_frame")

    assert step_type == "method"
    assert callable(fn)
    assert fn.__name__ == "get_frame"


def test_resolve_step_plugin(monkeypatch):
    """Test that plugin steps are resolved."""
    # Create dummy stack
    stack = AFMImageStack(
        data=np.ones((2, 4, 4)),
        pixel_size_nm=1.0,
        channel="height",
        file_path="dummy",
        frame_metadata=[{"timestamp": 0.0}, {"timestamp": 1.0}],
    )

    # Create mock entry point with .name and .load()
    mock_fn = lambda x: x  # noqa
    mock_ep = Mock()
    mock_ep.name = "mock_filter"
    mock_ep.load.return_value = mock_fn

    # Patch importlib.metadata.entry_points to return our mock
    mock_eps = Mock()
    mock_eps.__iter__ = lambda self: iter([mock_ep])
    monkeypatch.setattr(
        "playNano.afm_stack.metadata.entry_points",
        lambda group=None: [mock_ep] if group == "playNano.filters" else [],
    )

    # Call _resolve_step and assert behavior
    step_type, fn = stack._resolve_step("mock_filter")
    assert step_type == "plugin"
    assert fn is mock_fn
    mock_ep.load.assert_called_once()
