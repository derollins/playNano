"""Tests for built in analysis modules."""

import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from playNano.afm_stack import AFMImageStack
from playNano.analysis.base import AnalysisModule
from playNano.analysis.modules import feature_detection, x_means_clustering
from playNano.analysis.modules.count_nonzero import CountNonzeroModule
from playNano.analysis.modules.dbscan_clustering import DBSCANClusteringModule
from playNano.analysis.modules.feature_detection import MASK_MAP, FeatureDetectionModule
from playNano.analysis.modules.k_means_clustering import KMeansClusteringModule
from playNano.analysis.modules.log_blob_detection import LoGBlobDetectionModule
from playNano.analysis.modules.particle_tracking import ParticleTrackingModule
from playNano.analysis.modules.x_means_clustering import XMeansClusteringModule

# --- Test for abstract base class ---


def test_unimplemented_analysismodule_raises():
    """Attempt to instantiate a raw subclass with neither name nor run should fail."""

    class RawModule(AnalysisModule):
        pass  # implements nothing

    with pytest.raises(
        TypeError,
        match=r"abstract class .* (with|without) (an implementation for )?abstract method[s]? (name|run|'name'(, 'run')?)",  # noqa: E501
    ):
        RawModule()


def test_missing_name_property_raises():
    """Test that subclass without `name` property raises TypeError."""

    class MissingName(AnalysisModule):
        def run(self, stack, previous_results=None, **params):
            return {}

    with pytest.raises(
        TypeError,
        match=r"abstract class .* (with|without) (an implementation for )?abstract method[s]? '?name'?",  # noqa: E501
    ):
        MissingName()


def test_missing_run_method_raises():
    """Test that subclass without `run()` method raises TypeError."""

    class MissingRun(AnalysisModule):
        @property
        def name(self):
            return "dummy"

    with pytest.raises(
        TypeError,
        match=r"abstract class .* (with|without) (an implementation for )?abstract method[s]? '?run'?",  # noqa: E501
    ):
        MissingRun()


def test_cannot_instantiate_abstract_base_class():
    """Test that ABC raises error if not instantiated correctly."""
    with pytest.raises(TypeError):
        AnalysisModule()


class IncompleteModule(AnalysisModule):
    """Create a in incomplete analysis module class."""

    pass


def test_incomplete_subclass_instantiation_fails():
    """Test that an inclomplete subclass causes instantiation failure."""
    with pytest.raises(TypeError):
        IncompleteModule()


class DummyModule(AnalysisModule):
    """Dummy module for testing analysis module initilisation."""

    @property
    def name(self):
        """Define the name of the module."""
        return super().name  # Calls the base abstract property to cover it

    def run(self, stack, previous_results=None, **params):
        """Define the run method of this dummy module."""
        return super().run(
            stack, previous_results, **params
        )  # Calls base abstract method to cover it


def test_abstract_methods_raise():
    """Test that an error is raised if a module doesn't follow the ABC."""
    dummy = DummyModule()
    with pytest.raises(NotImplementedError):
        _ = dummy.name  # should raise because base is abstract

    with pytest.raises(NotImplementedError):
        dummy.run(None)


# --- Tests for feature_detection ---


class DummyStackNoData:
    """Create dummy class with no data."""

    data = None


def test_run_raises_if_no_data():
    """Test that run raises a ValueError if there is no data attribute."""
    fd = FeatureDetectionModule()
    stack = DummyStackNoData()
    with pytest.raises(ValueError, match="AFMImageStack has no data"):
        fd.run(stack, mask_fn=lambda f: f > 0)


class DummyStackWrongShape:
    """Simulate an AFM stack with invalid data shape."""

    data = np.array([1, 2, 3])  # 1D array instead of 3D


def test_run_raises_if_data_not_3d():
    """Test that run raises ValueError if stack.data exists but is not 3D."""
    fd = FeatureDetectionModule()
    stack = DummyStackWrongShape()
    with pytest.raises(ValueError, match="stack.data must be a 3D numpy array"):
        fd.run(stack, mask_fn=lambda f: f > 0)


def test_mask_fn_type_error_fallback():
    """Test for something to do with a type error."""
    import numpy as np

    class DummyStack:
        def __init__(self, data):
            self.data = data

        def time_for_frame(self, i):
            return i

    data = np.ones((1, 2, 2))
    stack = DummyStack(data)
    fd = FeatureDetectionModule()

    def mask_fn(frame, **kwargs):
        if kwargs:
            raise TypeError("forced")
        return frame > 0

    result = fd.run(stack, mask_fn=mask_fn)
    assert "features_per_frame" in result


def test_run_resolves_registered_mask_string(monkeypatch):
    """Test that a registered mask key string resolves to the correct function."""
    fd = FeatureDetectionModule()

    # Dummy stack: 1 frame, 3x3 image
    class DummyStackFeatures:
        data = np.ones((1, 3, 3))

        def time_for_frame(self, i):
            return i

    stack = DummyStackFeatures()

    # Pick a registered mask key
    registered_key = "dummy_mask_key"

    # Dummy mask function: all True
    def dummy_mask(frame):
        return np.ones_like(frame, dtype=bool)

    # Patch MASK_MAP to include dummy mask
    monkeypatch.setitem(MASK_MAP, registered_key, dummy_mask)

    # Run module, disable remove_edge to allow small frame region
    result = fd.run(stack, mask_fn=registered_key, remove_edge=False, min_size=1)

    # Check output structure
    assert "features_per_frame" in result
    assert "labeled_masks" in result
    assert "summary" in result

    # Since mask is all True, labeled mask should have one region
    assert result["labeled_masks"][0].max() == 1


def test_skip_empty_vals_region():
    """Test that empty values are skipped."""

    class DummyStack:
        def __init__(self, data):
            self.data = data

        def time_for_frame(self, i):
            return i

    # Create data with shape (1, 5, 5)
    data = np.zeros((1, 5, 5), dtype=float)
    # Create a mask_fn that returns a mask with one labeled region
    # but frame is zero everywhere so vals is empty or zero size? Actually
    # vals.size > 0 for zeros
    # To create empty vals, label a mask that doesn't intersect with frame?
    # A hack: override label() to create a region with label but empty pixels

    # Instead, test code path executes without error when vals.size == 0, so
    # patch regionprops to return a prop with empty mask_pixels

    fd = FeatureDetectionModule()

    stack = DummyStack(data)

    # Patch regionprops to produce a prop with vals.size == 0
    original_regionprops = feature_detection.regionprops

    def fake_regionprops(labeled, intensity_image=None):
        """Create a fake region prop."""

        class FakeProp:
            area = 10
            bbox = (1, 1, 4, 4)
            label = 1
            centroid = (2.0, 2.0)

            def __init__(self):
                pass

        # Return a list with one FakeProp, but mask_pixels is empty
        # We'll override the mask inside run by patching
        # 'labeled == prop.label' to be empty
        return [FakeProp()]

    feature_detection.regionprops = fake_regionprops

    def mask_fn(frame, **kwargs):
        return np.ones_like(frame, dtype=bool)

    try:
        result = fd.run(stack, mask_fn=mask_fn)
        assert "features_per_frame" in result
    finally:
        feature_detection.regionprops = original_regionprops


def test_time_for_frame_exception():
    """Test that time_for_frame raises an exception."""

    class DummyStack:
        def __init__(self, data):
            self.data = data

        def time_for_frame(self, i):
            raise RuntimeError("forced error")

    data = np.zeros((1, 5, 5))
    data[0, 2, 2] = 1  # single bright pixel as feature
    stack = DummyStack(data)
    fd = FeatureDetectionModule()

    def mask_fn(frame, **kwargs):
        return frame > 0  # mask covers the bright pixel

    result = fd.run(stack, mask_fn=mask_fn, min_size=1)

    # Now features_per_frame[0][0] should exist
    assert abs(result["features_per_frame"][0][0]["frame_timestamp"] - 0) < 1e-6


@pytest.fixture
def stack_1frame_with_timestamps():
    """
    Create AFMImageStack with 1 frame of 3x3 data and an explicit timestamp.

    frame_metadata contains a 'timestamp' key.
    """
    data = np.arange(9, dtype=float).reshape(1, 3, 3)
    meta = [{"timestamp": 1.5}]
    with TemporaryDirectory() as td:
        stack = AFMImageStack(
            data.copy(),
            pixel_size_nm=1.0,
            channel="height",
            file_path=Path(td),
            frame_metadata=meta,
        )
        yield stack


@pytest.fixture
def stack_2frames_no_timestamps():
    """
    Make AFMImageStack with 2 frames of 3x3 data, but missing timestamps in metadata.

    time_for_frame will return None, module should default timestamp to frame index.
    """
    data = np.stack([np.zeros((3, 3)), np.ones((3, 3))], axis=0)
    # frame_metadata entries without 'timestamp'
    meta = [{}, {}]
    with TemporaryDirectory() as td:
        stack = AFMImageStack(
            data.copy(),
            pixel_size_nm=1.0,
            channel="height",
            file_path=Path(td),
            frame_metadata=meta,
        )
        yield stack


def simple_center_mask(frame: np.ndarray, **kwargs) -> np.ndarray:
    """Mask only the center pixel of a 3x3 frame."""
    H, W = frame.shape
    mask = np.zeros((H, W), dtype=bool)
    # center at index (1,1)
    mask[1, 1] = True
    return mask


def full_mask(frame: np.ndarray, **kwargs) -> np.ndarray:
    """Mask all pixels True."""
    return np.ones_like(frame, dtype=bool)


def hole_mask(frame: np.ndarray, **kwargs) -> np.ndarray:
    """
    Create a mask with a hole in the center for a 3x3 frame.

    True on border, False at center.
    """
    H, W = frame.shape
    mask = np.ones((H, W), dtype=bool)
    # hole at center
    mask[1, 1] = False
    return mask


def test_requires_mask_fn_or_key(stack_1frame_with_timestamps):
    """Test that module requires either a mask funciton or key."""
    module = FeatureDetectionModule()
    stack = stack_1frame_with_timestamps
    # Neither mask_fn nor mask_key provided => ValueError
    with pytest.raises(ValueError):
        module.run(stack)


def test_invalid_data_shape():
    """Test that invalid data shapes raise ValueError."""
    module = FeatureDetectionModule()

    # Create a stack-like object with data not 3D
    class Dummy:
        data = np.ones((3, 3))  # 2D

        def time_for_frame(self, idx):
            return None

    dummy_stack = Dummy()
    with pytest.raises(ValueError):
        module.run(dummy_stack, mask_fn=simple_center_mask)


def test_mask_fn_returns_invalid_shape(stack_1frame_with_timestamps):
    """Test that mask_fn returns ivalid data shapes raise ValueError."""
    module = FeatureDetectionModule()
    stack = stack_1frame_with_timestamps

    # Define mask_fn returning wrong shape
    def bad_mask(frame: np.ndarray, **kwargs):
        return np.zeros((2, 2), dtype=bool)

    with pytest.raises(ValueError):
        module.run(stack, mask_fn=bad_mask)


def test_mask_key_not_in_previous_results(stack_1frame_with_timestamps):
    """Test that if mask_key not in pervious result KeyError."""
    module = FeatureDetectionModule()
    stack = stack_1frame_with_timestamps
    # previous_results empty => KeyError
    with pytest.raises(KeyError):
        module.run(stack, mask_key="nonexistent")


def test_mask_key_wrong_type_or_shape(stack_1frame_with_timestamps):
    """Test that if mask_key is wrong shape or type ValueError."""
    module = FeatureDetectionModule()
    stack = stack_1frame_with_timestamps
    # previous_results contains wrong dtype
    wrong = np.zeros((1, 3, 3), dtype=float)  # not bool
    with pytest.raises(ValueError):
        module.run(stack, previous_results={"m": wrong}, mask_key="m")
    # previous_results contains wrong shape
    wrong2 = np.zeros((2, 3, 3), dtype=bool)
    with pytest.raises(ValueError):
        module.run(stack, previous_results={"m": wrong2}, mask_key="m")


def test_single_feature_detection_center(stack_1frame_with_timestamps):
    """Test that for 1 frame, use simple_center_mask. Expect exactly one feature at center."""  # noqa
    module = FeatureDetectionModule()
    stack = stack_1frame_with_timestamps
    out = module.run(stack, mask_fn=simple_center_mask, min_size=1, remove_edge=False)
    # Check keys
    assert "features_per_frame" in out
    assert "labeled_masks" in out
    assert "summary" in out
    fpf = out["features_per_frame"]
    assert isinstance(fpf, list) and len(fpf) == 1
    feats = fpf[0]
    # One feature detected
    assert len(feats) == 1
    feat = feats[0]
    # Check fields
    assert feat["label"] == 1
    # area should be 1 (single pixel)
    assert feat["area"] == 1
    # centroid should be roughly (1.0, 1.0)
    assert pytest.approx(feat["centroid"][0]) == 1.0
    assert pytest.approx(feat["centroid"][1]) == 1.0
    # frame_timestamp: explicit 1.5
    assert feat["frame_timestamp"] == pytest.approx(1.5)
    # labeled_masks: one array with label 1 at center
    lm = out["labeled_masks"][0]
    assert lm.shape == (3, 3)
    # Only center pixel labeled 1
    mask_positions = np.argwhere(lm == 1)
    assert mask_positions.shape == (1, 2)
    assert (mask_positions[0] == np.array([1, 1])).all()
    # Summary
    summary = out["summary"]
    assert summary["total_frames"] == 1
    assert summary["total_features"] == 1
    assert summary["avg_features_per_frame"] == pytest.approx(1.0)


def test_full_mask_filtered_out_by_remove_edge(stack_1frame_with_timestamps):
    """Test if mask covers entire frame and remove_edge=True, region touches edges and should be discarded."""  # noqa
    module = FeatureDetectionModule()
    stack = stack_1frame_with_timestamps
    out = module.run(stack, mask_fn=full_mask, min_size=1, remove_edge=True)
    # No features remain
    assert out["features_per_frame"][0] == []
    summary = out["summary"]
    assert summary["total_frames"] == 1
    assert summary["total_features"] == 0
    assert summary["avg_features_per_frame"] == pytest.approx(0.0)
    # labeled_masks: after filtering, filtered_mask is all False,
    # so labeled array all zeros
    lm = out["labeled_masks"][0]
    assert np.all(lm == 0)


def test_full_mask_keep_when_remove_edge_false(stack_1frame_with_timestamps):
    """Test if mask covers entire frame but remove_edge=False, region kept (area=9)."""
    module = FeatureDetectionModule()
    stack = stack_1frame_with_timestamps
    out = module.run(stack, mask_fn=full_mask, min_size=1, remove_edge=False)
    # One feature with area 3×3=9
    feats = out["features_per_frame"][0]
    assert len(feats) == 1
    feat = feats[0]
    assert feat["area"] == 9
    # centroid of full 3x3 is at (1,1)
    assert pytest.approx(feat["centroid"][0]) == 1.0
    assert pytest.approx(feat["centroid"][1]) == 1.0
    summary = out["summary"]
    assert summary["total_features"] == 1
    assert summary["avg_features_per_frame"] == pytest.approx(1.0)


def test_fill_holes_behavior(stack_1frame_with_timestamps):
    """
    Test that fill_holes=True fills the hole in hole_mask.

    For remove_edge=False to keep the region after filling.
    """
    module = FeatureDetectionModule()
    stack = stack_1frame_with_timestamps
    # Without filling holes: hole_mask yields border True, center False.
    out_no_fill = module.run(
        stack, mask_fn=hole_mask, min_size=1, remove_edge=False, fill_holes=False
    )
    # The mask has two separate regions? Actually border is one region touching edges;
    # but since remove_edge=False, it's kept as a single region labeled 1,
    # but note that regionprops labels contiguous True;
    # border pixels connected along edges.
    # There may be multiple connected components along edges depending on
    # connectivity; skimage.label uses connectivity=1 by default.
    # However, hole remains; area = number of True pixels = 8.
    feats_no_fill = out_no_fill["features_per_frame"][0]
    # Expect one region of area 8
    assert len(feats_no_fill) == 1
    assert feats_no_fill[0]["area"] == 8

    # With filling holes (hole_area=None): hole at center
    # filled => mask all True => area 9
    out_fill = module.run(
        stack, mask_fn=hole_mask, min_size=1, remove_edge=False, fill_holes=True
    )
    feats_fill = out_fill["features_per_frame"][0]
    assert len(feats_fill) == 1
    assert feats_fill[0]["area"] == 9
    # centroid still (1,1)
    assert pytest.approx(feats_fill[0]["centroid"][0]) == 1.0
    assert pytest.approx(feats_fill[0]["centroid"][1]) == 1.0

    # Summary updated accordingly
    assert out_fill["summary"]["total_features"] == 1
    assert out_fill["summary"]["avg_features_per_frame"] == pytest.approx(1.0)


def test_mask_key_path(stack_1frame_with_timestamps):
    """Test using mask_key from previous_results."""
    module = FeatureDetectionModule()
    stack = stack_1frame_with_timestamps
    # Prepare a boolean mask array same shape: e.g., center only
    data = stack.data
    mask_arr = np.zeros_like(data, dtype=bool)
    mask_arr[:, 1, 1] = True
    previous_results = {"mymask": mask_arr}
    out = module.run(
        stack,
        previous_results=previous_results,
        mask_key="mymask",
        min_size=1,
        remove_edge=False,
    )
    feats = out["features_per_frame"][0]
    assert len(feats) == 1
    assert feats[0]["area"] == 1


def test_skip_empty_vals(monkeypatch):
    """Test that empty values are skipped."""

    class DummyStack:
        def __init__(self, data):
            self.data = data

        def time_for_frame(self, i):
            return i

    data = np.ones((1, 3, 3))
    stack = DummyStack(data)
    fd = feature_detection.FeatureDetectionModule()

    # Patch regionprops to return one region with label 1
    # but the labeled mask will have no pixels == 1
    def fake_regionprops(labeled, intensity_image=None):
        class FakeProp:
            area = 10
            bbox = (0, 0, 2, 2)
            label = 1
            centroid = (1.0, 1.0)

        return [FakeProp()]

    monkeypatch.setattr(feature_detection, "regionprops", fake_regionprops)

    # Patch label function to return labeled mask with no pixels == 1
    def fake_label(mask):
        return np.zeros_like(mask, dtype=int)  # no labels

    monkeypatch.setattr(feature_detection, "label", fake_label)

    def mask_fn(frame, **kwargs):
        """Imitate a masking funciton."""
        return np.ones_like(frame, dtype=bool)

    result = fd.run(stack, mask_fn=mask_fn, min_size=1)

    # If we reach here without error, line `if vals.size == 0: continue` was executed
    assert "features_per_frame" in result

    class DummyStack:
        def __init__(self, data):
            self.data = data

        def time_for_frame(self, i):
            return i

    data = np.ones((1, 2, 2))
    stack = DummyStack(data)
    fd = FeatureDetectionModule()

    # Define a mask_fn that raises TypeError when called with kwargs,
    # but works when called with only the frame.
    def mask_fn(frame, **kwargs):
        if kwargs:
            raise TypeError("forced error")
        return frame > 0

    result = fd.run(stack, mask_fn=mask_fn, some_kwarg=123)
    assert "features_per_frame" in result
    # Just check that fallback call happened and mask computed successfully


def test_zero_frames_stack():
    """Test if stack.data has zero frames (shape (0, H, W)), expect summary zero and empty lists."""  # noqa
    # Create AFMImageStack with zero frames: data shape (0, 3, 3)
    data = np.zeros((0, 3, 3), dtype=float)
    # frame_metadata empty list
    with TemporaryDirectory() as td:
        stack = AFMImageStack(
            data.copy(),
            pixel_size_nm=1.0,
            channel="height",
            file_path=Path(td),
            frame_metadata=[],
        )
        module = FeatureDetectionModule()
        # Since n_frames=0, mask_fn is still required; but run loop won't iterate.
        # Provide a dummy mask_fn that wouldn't be called.
        out = module.run(
            stack, mask_fn=simple_center_mask, min_size=1, remove_edge=False
        )
        # Expect features_per_frame empty list, labeled_masks empty list
        assert out["features_per_frame"] == []
        assert out["labeled_masks"] == []
        summary = out["summary"]
        assert summary["total_frames"] == 0
        assert summary["total_features"] == 0
        assert summary["avg_features_per_frame"] == 0


def test_fill_holes_with_hole_area(stack_1frame_with_timestamps):
    """
    Test fill_holes with hole_area limiting fill.

    For hole_mask of 3x3, hole_area=1 should fill only holes smaller than area 1;
    but hole size=1, so area_threshold=1: remove_small_holes
    fills holes with area < area_threshold:since area == 1 is not < 1, it will
    NOT fill. So behavior matches no-fill.
    """
    module = FeatureDetectionModule()
    stack = stack_1frame_with_timestamps
    # hole_area = 1: hole size=1, not filled => area remains 8
    out = module.run(
        stack,
        mask_fn=hole_mask,
        min_size=1,
        remove_edge=False,
        fill_holes=True,
        hole_area=1,
    )
    feats = out["features_per_frame"][0]
    # Expect area 8 as in no-fill
    assert len(feats) == 1
    assert feats[0]["area"] == 8


def test_invalid_mask_key_type(stack_1frame_with_timestamps):
    """Test if mask_key provided but previous_results is None => KeyError."""
    module = FeatureDetectionModule()
    stack = stack_1frame_with_timestamps
    with pytest.raises(KeyError):
        module.run(stack, previous_results=None, mask_key="m")


def test_invalid_mask_fn_in_previous_results(stack_1frame_with_timestamps):
    """Test if previous_results[mask_key] is not boolean ndarray of correct shape => ValueError."""  # noqa
    module = FeatureDetectionModule()
    stack = stack_1frame_with_timestamps
    # Wrong dtype
    wrong = np.zeros_like(stack.data, dtype=int)
    with pytest.raises(ValueError):
        module.run(stack, previous_results={"m": wrong}, mask_key="m")
    # Wrong shape
    wrong2 = np.zeros((1, 2, 2), dtype=bool)
    with pytest.raises(ValueError):
        module.run(stack, previous_results={"m": wrong2}, mask_key="m")


def test_mask_fn_raises_inside(stack_1frame_with_timestamps):
    """
    Test if mask_fn raises TypeError or other inside.

    Should propagate/log as ValueErrorf,for instance mask_fn
    raising ValueError on certain frame.
    """
    module = FeatureDetectionModule()
    stack = stack_1frame_with_timestamps

    def bad_mask(frame):
        raise RuntimeError("mask failure")

    # mask_fn raises: caught in run?
    # In current code, mask_fn errors bubble as not TypeError,
    # so caught by outer except?
    # The code does:
    #    try: mf = mask_fn(frame, **mask_kwargs)
    #    except TypeError: ...
    #    if not valid mask: ValueError
    # But if mask_fn raises RuntimeError,
    # it's not caught by TypeError branch, so escapes and aborts.
    with pytest.raises(RuntimeError):
        module.run(stack, mask_fn=bad_mask)


def test_two_separate_regions(stack_1frame_with_timestamps):
    """
    Create a mask_fn that yields two pixels in a 4x4 frame at (1,1) and (2,2).

    With default 8-connectivity, these form one region of area 2.
    """
    data = np.zeros((1, 4, 4), dtype=float)
    meta = [{"timestamp": 0.0}]
    with TemporaryDirectory() as td:
        stack = AFMImageStack(
            data.copy(),
            pixel_size_nm=1.0,
            channel="height",
            file_path=Path(td),
            frame_metadata=meta,
        )

        # mask_fn: True at (1,1) and (2,2) only
        def two_pixel_mask(frame, **kwargs):
            mask = np.zeros_like(frame, dtype=bool)
            mask[1, 1] = True
            mask[2, 2] = True
            return mask

        module = FeatureDetectionModule()
        out = module.run(stack, mask_fn=two_pixel_mask, min_size=1, remove_edge=False)

        feats = out["features_per_frame"][0]
        # With 8-connectivity, these diagonals merge into one region:
        assert len(feats) == 1

        # That region’s area should be 2
        region = feats[0]
        assert region["area"] == 2

        summary = out["summary"]
        assert summary["total_frames"] == 1
        # total_features is count of regions = 1
        assert summary["total_features"] == 1
        assert summary["avg_features_per_frame"] == pytest.approx(1.0)

        # Check labeled_masks: exactly 2 pixels labeled (regardless of label value)
        lm = out["labeled_masks"][0]
        assert lm.shape == (4, 4)
        assert np.count_nonzero(lm) == 2


# --- Tests for particle_tracking ---


class MockAFMImageStack:
    """Mock AFMImageStack for testing."""

    def __init__(self, n_frames):
        """
        Initialize the mock AFM image stack.

        Parameters:
            n_frames (int): Number of frames in the mock image stack.
        """
        self.n_frames = n_frames


@pytest.fixture
def mock_stack():
    """Provide a mock AFMImageStack with 3 frames."""
    return MockAFMImageStack(n_frames=3)


@pytest.fixture
def mock_feature_detection_outputs():
    """Provide mock feature detection outputs with centroids and labels."""
    return {
        "features_per_frame": [
            [{"centroid": (0, 0), "label": 1}],
            [{"centroid": (1, 1), "label": 2}],
            [{"centroid": (2, 2), "label": 3}],
        ],
        "labeled_masks": [
            np.array([[0, 1], [1, 0]]),
            np.array([[0, 2], [2, 0]]),
            np.array([[0, 3], [3, 0]]),
        ],
    }


def make_dummy_stack(n_frames=3, H=2, W=2) -> AFMImageStack:
    """Provide the minimal required AFMImageStack constructor arguments here."""
    dummy_data = np.zeros((n_frames, H, W))
    return AFMImageStack(
        data=dummy_data, pixel_size_nm=1.0, channel="height", file_path="dummy.jpk"
    )


def test_missing_coordinate_keys_raise_keyerror():
    """Test that a key error is raised when the coordinate key is missing."""
    mod = ParticleTrackingModule()
    stack = make_dummy_stack()

    # Features missing both coord_columns keys and 'centroid'
    previous_results = {
        "feature_detection": {
            "features_per_frame": [
                [
                    {"some_key": 123}
                ],  # missing 'centroid_x', 'centroid_y', and 'centroid'
            ],
            "labeled_masks": [np.array([[0]])],
        }
    }

    with pytest.raises(KeyError, match="Missing coordinate keys"):
        mod.run(
            stack,
            previous_results=previous_results,
            coord_columns=("centroid_x", "centroid_y"),
        )


def test_tracking_module_name():
    """Return correct module name."""
    mod = ParticleTrackingModule()
    assert mod.name == "particle_tracking"


def test_tracking_requires_feature_detection():
    """Require 'feature_detection' in previous_results."""
    mod = ParticleTrackingModule()
    assert "feature_detection" in mod.requires


def test_tracking_raises_without_feature_detection(mock_stack):
    """Raise error if 'feature_detection' is missing."""
    mod = ParticleTrackingModule()
    with pytest.raises(RuntimeError):
        mod.run(mock_stack, previous_results={})


def test_tracking_output_structure(mock_stack, mock_feature_detection_outputs):
    """Return expected keys and track structure."""
    mod = ParticleTrackingModule()
    result = mod.run(
        mock_stack,
        previous_results={"feature_detection": mock_feature_detection_outputs},
    )

    # Top-level structure
    assert "tracks" in result
    assert "track_masks" in result
    assert "n_tracks" in result

    # Type checks
    assert isinstance(result["tracks"], list)
    assert isinstance(result["track_masks"], dict)
    assert isinstance(result["n_tracks"], int)

    # Check structure of first track (if any)
    if result["tracks"]:
        trk = result["tracks"][0]
        assert "id" in trk
        assert "frames" in trk
        assert "point_indices" in trk
        assert "coords" in trk
        assert all(isinstance(coord, tuple) for coord in trk["coords"])


def test_tracking_links_features():
    """Test that ParticleTrackingModule links features by nearest neighbor."""
    features_per_frame = [
        [{"centroid": (0.0, 0.0), "label": 1}],  # Frame 0
        [{"centroid": (0.5, 0.5), "label": 2}],  # Frame 1
        [{"centroid": (1.0, 1.0), "label": 3}],  # Frame 2
    ]

    labeled_masks = [
        np.array([[0, 1], [1, 0]]),
        np.array([[0, 2], [2, 0]]),
        np.array([[0, 3], [3, 0]]),
    ]

    mod = ParticleTrackingModule()
    result = mod.run(
        MockAFMImageStack(n_frames=3),
        previous_results={
            "feature_detection": {
                "features_per_frame": features_per_frame,
                "labeled_masks": labeled_masks,
            }
        },
        max_distance=2.0,
    )

    assert result["n_tracks"] == 1
    track = result["tracks"][0]

    # Frame order
    assert track["frames"] == [0, 1, 2]

    # Check coordinates
    assert track["coords"] == [
        (0.0, 0.0),
        (0.5, 0.5),
        (1.0, 1.0),
    ]

    # Check point indices
    assert track["point_indices"] == [0, 0, 0]


def test_tracking_handles_empty_frames(mock_stack):
    """Handle frames with no features."""
    fd_out = {
        "features_per_frame": [
            [{"centroid": (0, 0), "label": 1}],
            [],
            [{"centroid": (2, 2), "label": 2}],
        ],
        "labeled_masks": [
            np.array([[0, 1], [1, 0]]),
            np.array([[0, 0], [0, 0]]),
            np.array([[0, 2], [2, 0]]),
        ],
    }
    mod = ParticleTrackingModule()
    result = mod.run(
        mock_stack, previous_results={"feature_detection": fd_out}, max_distance=2.0
    )
    assert result["n_tracks"] == 2
    track_ids = [trk["id"] for trk in result["tracks"]]
    assert set(track_ids) == {0, 1}


def test_tracking_overlapping_centroids(mock_stack):
    """Handle multiple features with same centroid."""
    fd_out = {
        "features_per_frame": [
            [{"centroid": (1, 1), "label": 1}, {"centroid": (1, 1), "label": 2}],
            [{"centroid": (1, 1), "label": 3}],
        ],
        "labeled_masks": [np.array([[0, 1], [1, 2]]), np.array([[0, 3], [3, 0]])],
    }
    mod = ParticleTrackingModule()
    result = mod.run(
        mock_stack, previous_results={"feature_detection": fd_out}, max_distance=1.0
    )
    assert result["n_tracks"] >= 1
    for trk in result["tracks"]:
        assert isinstance(trk["coords"], list)
        assert isinstance(trk["point_indices"], list)


class DummyStack2:
    """Stub mimicking AFMImageStack: holds 3D .data and timestamps."""

    def __init__(self, data=None, times=None):
        """Initialise the dummy class."""
        # Ensure data is 3D array
        self.data = np.array(data) if data is not None else np.empty((0, 0, 0))
        # Frame timestamps
        if times is not None:
            if len(times) != self.data.shape[0]:
                raise ValueError("Length of times must match number of frames")
            self._times = list(times)
        else:
            self._times = list(range(self.data.shape[0]))

    def time_for_frame(self, idx):
        """Return timestamp for frame index or raise IndexError."""
        try:
            return float(self._times[idx])
        except IndexError:
            raise IndexError(f"Frame index {idx} out of range for DummyStack") from None


@pytest.fixture
def single_blob_stack():
    """Test single frame with one bright spot at (5,5) and timestamp 0.5."""
    # Single frame, bright spot at (5,5)
    img = np.zeros((1, 11, 11), dtype=float)
    img[0, 5, 5] = 1.0
    return DummyStack2(data=img, times=[0.5])


@pytest.fixture
def multi_blob_stack():
    """Test two frames each with two bright spots and timestamps 0.0, 1.0."""
    # Two frames with two spots each
    f0 = np.zeros((10, 10))
    f0[2, 2] = f0[7, 7] = 1.0
    f1 = np.zeros((10, 10))
    f1[2, 7] = f1[7, 2] = 1.0
    data = np.stack([f0, f1])
    return DummyStack2(data=data, times=[0.0, 1.0])


@pytest.fixture
def empty_stack():
    """Test zero-frame stack yields empty data and no timestamps."""
    # Zero-frame stack
    return DummyStack2(data=np.zeros((0, 5, 5)), times=[])


def test_name_property():
    """Test LoGBlobDetectionModule.name equals 'log_blob_detection'."""
    mod = LoGBlobDetectionModule()
    assert mod.name == "log_blob_detection"


def test_detect_single_blob(single_blob_stack):
    """Detect single blob with radius included and correct summary."""
    mod = LoGBlobDetectionModule()
    out = mod.run(
        single_blob_stack,
        min_sigma=1.0,
        max_sigma=1.0,
        num_sigma=1,
        threshold=0.2,
        overlap=0.5,
        include_radius=True,
    )
    feats = out["features_per_frame"]
    assert isinstance(feats, list) and len(feats) == 1
    assert len(feats[0]) == 1
    blob = feats[0][0]
    assert blob["frame_timestamp"] == pytest.approx(0.5)
    assert blob["y"] == pytest.approx(5.0)
    assert blob["x"] == pytest.approx(5.0)
    assert blob["sigma"] == pytest.approx(1.0)
    assert blob["radius"] == pytest.approx(1.0 * np.sqrt(2))
    summary = out["summary"]
    assert summary == {
        "total_frames": 1,
        "total_blobs": 1,
        "avg_blobs_per_frame": pytest.approx(1.0),
    }


def test_include_radius_false(single_blob_stack):
    """Test radius field omitted when include_radius=False."""
    mod = LoGBlobDetectionModule()
    out = mod.run(
        single_blob_stack,
        min_sigma=1,
        max_sigma=1,
        num_sigma=1,
        threshold=0.2,
        include_radius=False,
    )
    blob = out["features_per_frame"][0][0]
    assert "radius" not in blob


def test_detect_multiple_blobs(multi_blob_stack):
    """Detect two blobs per frame and correct overall summary."""
    mod = LoGBlobDetectionModule()
    out = mod.run(
        multi_blob_stack,
        min_sigma=1,
        max_sigma=1,
        num_sigma=1,
        threshold=0.3,
        overlap=0.5,
    )
    feats = out["features_per_frame"]
    assert len(feats) == 2
    assert all(len(frame_feats) == 2 for frame_feats in feats)
    summary = out["summary"]
    assert summary["total_frames"] == 2
    assert summary["total_blobs"] == 4
    assert summary["avg_blobs_per_frame"] == pytest.approx(2.0)


def test_threshold_too_high(single_blob_stack):
    """Test high threshold yields zero blobs and zero average."""
    mod = LoGBlobDetectionModule()
    out = mod.run(single_blob_stack, threshold=2.0)
    assert out["features_per_frame"][0] == []
    assert out["summary"]["total_blobs"] == 0
    assert out["summary"]["avg_blobs_per_frame"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "min_sigma,max_sigma,num_sigma",
    [
        (0.5, 2.0, 5),
        (2.0, 5.0, 5),
    ],
)
def test_sigma_range(single_blob_stack, min_sigma, max_sigma, num_sigma):
    """Detect only when sigma range includes spot scale."""
    mod = LoGBlobDetectionModule()
    out = mod.run(
        single_blob_stack,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=0.2,
    )
    count = len(out["features_per_frame"][0])
    expected = 1 if min_sigma <= 1.0 else 0
    assert count == expected


def test_empty_stack(empty_stack):
    """Test empty stack returns empty results and zero summary."""
    mod = LoGBlobDetectionModule()
    out = mod.run(empty_stack)
    assert out["features_per_frame"] == []
    assert out["summary"]["total_frames"] == 0
    assert out["summary"]["total_blobs"] == 0
    assert out["summary"]["avg_blobs_per_frame"] == 0


def test_invalid_data_shape_logblog():
    """Test missing timestamp raises IndexError during run."""
    mod = LoGBlobDetectionModule()

    class BadStack:
        data = np.zeros((5, 5))  # not 3D

        def time_for_frame(self, i):
            return 0.0

    with pytest.raises((AttributeError, ValueError)):
        mod.run(BadStack())


def test_time_for_frame_out_of_range(single_blob_stack):
    """Raises IndexError if frame timestamp is requested out of range."""
    mod = LoGBlobDetectionModule()
    # corrupt times
    single_blob_stack._times = []
    with pytest.raises(IndexError):
        mod.run(single_blob_stack)


# --- Fixtures and dummy stacks for particle clustering  ---


@pytest.fixture(autouse=True)
def patch_numpy_warnings():
    """Monkeypatch NumPy warnings for cleaner test output."""
    import numpy as _np

    _np.warnings = warnings
    yield
    del _np.warnings


# A minimal “stack” stub:
class DummyStack:
    """Minimal AFMImageStack stub with only .time_for_frame support."""

    def __init__(self, times):
        """
        Initialize the dummy stack with frame timestamps.

        Parameters:
            times (list): List of timestamps, one per frame.
        """
        # times: list of timestamps, one per frame
        self._times = times

    def time_for_frame(self, idx):
        """
        Return the timestamp for a given frame index.

        Parameters:
            idx (int): Index of the frame.

        Returns:
            The timestamp corresponding to the given frame index.
        """
        return self._times[idx]


@pytest.fixture
def simple_per_frame():
    """Make two-frame, two-feature-per-frame mock data for clustering tests."""
    # two frames, two features each, forming two separable clusters in x-y
    # frame 0: cluster A at (0,0), cluster B at (10,10)
    # frame 1: same clusters moved slightly
    return [
        [
            {"centroid": (0.0, 0.0)},
            {"centroid": (10.0, 10.0)},
        ],
        [
            {"centroid": (1.0, -1.0)},
            {"centroid": (11.0, 9.0)},
        ],
    ]


def make_prev(simple_per_frame, key="features_per_frame"):
    """Create helper to wrap features into feature_detection previous_results dict."""
    return {"feature_detection": {key: simple_per_frame}}


# --- Tests for x_mean_clustering ---


def test_missing_dependency():
    """Test XMeans raises if 'feature_detection' is missing from previous_results."""
    mod = XMeansClusteringModule()
    with pytest.raises(RuntimeError):
        mod.run(stack=None, previous_results={})  # no 'feature_detection'


def test_empty_input():
    """Test XMeans returns zero clusters on empty input."""
    stack = DummyStack([0.0, 1.0])
    empty_prev = {"feature_detection": {"features_per_frame": [[], []]}}
    mod = XMeansClusteringModule()
    out = mod.run(stack, empty_prev, min_k=1, max_k=3)
    assert out["clusters"] == []
    assert out["cluster_centers"].shape == (0, 3)
    assert out["summary"]["n_clusters"] == 0


@pytest.mark.parametrize(
    "normalise,time_weight,expected_clusters",
    [
        (False, None, 2),
        (True, None, 2),
        (True, 0.1, 2),
    ],
)
def test_basic_two_clusters(
    simple_per_frame, normalise, time_weight, expected_clusters
):
    """Test XMeans detects 2 expected clusters in separable synthetic data."""
    # build dummy stack with arbitrary times
    stack = DummyStack([0.0, 1.0])
    prev = make_prev(simple_per_frame)
    mod = XMeansClusteringModule()
    out = mod.run(
        stack, prev, min_k=2, max_k=2, normalise=normalise, time_weight=time_weight
    )
    # should find exactly two clusters
    assert out["summary"]["n_clusters"] == expected_clusters
    # each cluster has two members
    counts = list(out["summary"]["members_per_cluster"].values())
    assert sorted(counts) == [2, 2]
    # cluster_centers shape correct
    centers = out["cluster_centers"]
    assert centers.ndim == 2 and centers.shape[1] == 3  # x,y,time


def test_coord_columns_override(simple_per_frame):
    """Test XMeans uses provided coord_columns instead of default centroid."""
    # test that giving coord_columns works (explicit centroid_x,centroid_y)
    # first massage the features to have explicit keys
    pf = [
        [
            {"centroid_x": 0.0, "centroid_y": 0.0},
            {"centroid_x": 5.0, "centroid_y": 5.0},
        ]
    ]
    stack = DummyStack([0.0])
    prev = {"feature_detection": {"features_per_frame": pf}}
    mod = XMeansClusteringModule()
    out = mod.run(
        stack,
        prev,
        min_k=1,
        max_k=1,
        coord_columns=("centroid_x", "centroid_y"),
        use_time=False,
        normalise=False,
    )
    assert out["summary"]["n_clusters"] == 1
    # coords should exactly reproduce inputs
    coords = out["clusters"][0]["coords"]
    assert set(coords) == {(0.0, 0.0), (5.0, 5.0)}


def test_xmeans_clusters_and_members_are_initialized():
    """Ensure run() starts clustering and hits cluster initialization."""
    mod = XMeansClusteringModule()
    pf = [[{"centroid": (0, 0)}], [{"centroid": (5, 5)}]]
    stack = DummyStack([0.0, 1.0])
    prev = make_prev(pf)

    result = mod.run(stack, prev, min_k=1, max_k=2)
    assert isinstance(result["clusters"], list)
    assert "summary" in result


def test_xmeans_triggers_cluster_split():
    """Test that XMeans performs a cluster split and hits split loop."""
    pf = [
        [{"centroid": (0, 0)}],
        [{"centroid": (0.1, 0.1)}],
        [{"centroid": (10, 10)}],
        [{"centroid": (10.1, 10.1)}],
    ]
    stack = DummyStack([0.0, 1.0, 2.0, 3.0])
    prev = make_prev(pf)
    mod = XMeansClusteringModule()

    # Use low split threshold to force a split
    result = mod.run(stack, prev, min_k=1, max_k=4, bic_threshold=0.01)

    # Should have split into at least two clusters
    assert result["summary"]["n_clusters"] >= 2


def test_xmeans_skips_negative_cluster_labels():
    """Ensure negative cluster labels are skipped during output formatting."""
    mod = XMeansClusteringModule()
    pf = [
        [{"centroid": (0, 0)}],
        [{"centroid": (1000, 1000)}],
    ]  # Far apart — force split
    stack = DummyStack([0.0, 1.0])
    prev = make_prev(pf)

    # Temporarily monkeypatch core_xmeans to produce a -1 label
    from playNano.analysis.modules import x_means_clustering

    def fake_core_xmeans(data, **kwargs):
        return np.array([0, -1])  # One normal cluster, one invalid

    original_fn = x_means_clustering.core_xmeans
    x_means_clustering.core_xmeans = fake_core_xmeans

    try:
        result = mod.run(stack, prev, min_k=1, max_k=2)
        assert result["summary"]["n_clusters"] == 1
        assert all(c["id"] != -1 for c in result["clusters"])
    finally:
        x_means_clustering.core_xmeans = original_fn


def test_core_xmeans_handles_negative_labels(monkeypatch):
    """
    Test that the core_xmeans function properly skips negative cluster labels.

    We patch `np.unique` inside core_xmeans to force a negative label (-1),
    which will cause the `continue` line inside the loop to run.
    """

    # Prepare some dummy data
    data = np.array([[0, 0, 0], [1, 1, 1], [10, 10, 10]])

    # Patch np.unique to return an array including a negative label
    def fake_unique(labels):
        return np.array([-1, 0, 1])

    # Patch np.unique only within the core_xmeans module
    monkeypatch.setattr(x_means_clustering.np, "unique", fake_unique)

    # Run core_xmeans with dummy params, we only care about the label loop coverage
    labels_out, centers_out = x_means_clustering.core_xmeans(
        data,
        init_k=2,
        max_k=3,
        min_cluster_size=1,
        distance="sqeuclidean",
        replicates=1,
        max_iter=10,
        bic_threshold=0.0,
    )

    # Assert outputs are valid shapes
    assert labels_out.shape[0] == data.shape[0]
    assert centers_out.shape[1] == data.shape[1]


def test_core_xmeans_skips_small_clusters():
    """Test that core_xmeans skips small clusters when run directly."""
    # Create data with two clear clusters but one cluster has only one point
    data = np.array(
        [
            [0, 0, 0],  # cluster 0
            [0.1, 0.1, 0],  # cluster 0
            [100, 100, 0],  # cluster 1 (only one point)
        ]
    )

    init_k = 2
    max_k = 2
    min_cluster_size = 2  # require at least 2 points per cluster to split
    distance = "sqeuclidean"
    replicates = 1
    max_iter = 100
    bic_threshold = 0.0

    labels, centers = x_means_clustering.core_xmeans(
        data,
        init_k=init_k,
        max_k=max_k,
        min_cluster_size=min_cluster_size,
        distance=distance,
        replicates=replicates,
        max_iter=max_iter,
        bic_threshold=bic_threshold,
    )

    # Check output clusters count <= initial clusters (since one is too small to split)
    assert len(centers) <= max_k

    # Check the cluster with only one point was not split (center still included)
    assert any(np.allclose(center, [100, 100, 0]) for center in centers)


def test_run_skips_negative_cluster_ids(monkeypatch):
    """Test that run skips negative cluster id numbers."""
    module = XMeansClusteringModule()
    stack = DummyStack([0.0, 1.0])

    previous_results = {
        "feature_detection": {
            "features_per_frame": [
                [{"centroid_x": 0.1, "centroid_y": 0.2}],
                [{"centroid_x": 0.3, "centroid_y": 0.4}],
            ]
        }
    }

    # Patch core_xmeans to return a negative label
    def fake_core_xmeans(*args, **kwargs):
        """Make a fake core_xmeans function."""
        labels = np.array([-1, 0])
        centers = np.array([[0.1, 0.2, 0.0], [0.3, 0.4, 1.0]])  # 3D centers
        return labels, centers

    monkeypatch.setattr(
        "playNano.analysis.modules.x_means_clustering.core_xmeans", fake_core_xmeans
    )

    result = module.run(stack, previous_results)

    # Assert only the non-negative cluster is returned
    assert len(result["clusters"]) == 1
    assert result["clusters"][0]["id"] == 0


def test_continue_skips_negative_cluster_ids():
    """Test that negative clusters are skipped."""
    labels = np.array([0, -1, 1])  # Include a negative ID to trigger `continue`
    data = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    metadata = [(0, 0), (0, 1), (0, 2)]

    skipped_ids = []
    clusters_out, members = [], {}
    for cid in np.unique(labels):
        if cid < 0:
            skipped_ids.append(cid)
            continue
        idxs = np.where(np.atleast_1d(labels == cid))[0]
        frames, coords_list, p_inds = [], [], []
        for idx in idxs:
            f_idx, p_idx = metadata[idx]
            frames.append(f_idx)
            p_inds.append(p_idx)
            coords_list.append(tuple(data[idx]))
        clusters_out.append(
            {
                "id": int(cid),
                "frames": frames,
                "point_indices": p_inds,
                "coords": coords_list,
            }
        )
        members[int(cid)] = len(idxs)

    # Assert that the negative ID was skipped
    assert skipped_ids == [-1]
    assert all(cluster["id"] >= 0 for cluster in clusters_out)
    assert set(members.keys()) == {0, 1}


def test_compute_bic_triggers_eps_fallback():
    """Test that when bic triggers the fallback."""
    # All points are identical → variance = 0
    points = np.array([[1.0, 1.0], [1.0, 1.0]])
    center = np.array([[1.0, 1.0]])

    bic = x_means_clustering.compute_bic(points, center)

    # Just check that it returns a float (and doesn't crash)
    assert isinstance(bic, float)


def test_centroid_fallback(simple_per_frame):
    """Test that XMeans falls back to 'centroid' if coord_columns are missing."""
    # ensure that missing coord_columns triggers fallback to 'centroid' tuple
    stack = DummyStack([0.0])
    prev = make_prev([simple_per_frame[0]])  # only one frame
    mod = XMeansClusteringModule()
    # choose nonsense coord_columns so KeyError path triggers the centroid fallback
    out = mod.run(
        stack,
        prev,
        min_k=1,
        max_k=1,
        coord_columns=("x", "y"),  # neither 'x' nor 'y' present
        use_time=False,
        normalise=False,
    )
    # should succeed and find exactly 1 cluster of size 2
    assert out["summary"]["n_clusters"] == 1
    assert out["summary"]["members_per_cluster"][0] == 2


def test_time_weight_effect(simple_per_frame):
    """Test XMeans time_weight controls clustering across identical XY positions."""
    # make two clusters separated only in time, and very slightly in x-y;
    # with time_weight=0 they collapse to single cluster, with large weight they split.
    pf = [
        [{"centroid": (0.001, 0)}],
        [{"centroid": (0, 0)}],
        [{"centroid": (0.0002, 0)}],
        [{"centroid": (0, 0.005)}],
    ]
    stack = DummyStack([0.0, 1.0, 2.0, 3.0])
    prev = make_prev(pf)
    mod = XMeansClusteringModule()
    # with no weighting, all points cluster into one
    out1 = mod.run(stack, prev, min_k=1, max_k=4, normalise=True, time_weight=0.0)
    assert out1["summary"]["n_clusters"] == 1
    # with strong time weighting, should split into multiple clusters (up to max_k)
    out2 = mod.run(stack, prev, min_k=1, max_k=4, normalise=True, time_weight=1e7)
    assert out2["summary"]["n_clusters"] >= 2


def test_missing_coord_keys_raises_keyerror():
    """Test that XMeans raises KeyError if feature lacks required coordinate keys."""
    stack = DummyStack([0.0])
    # Feature with no required keys and no fallback centroid
    per_frame = [[{"area": 123}]]
    prev = {"feature_detection": {"features_per_frame": per_frame}}

    mod = XMeansClusteringModule()

    with pytest.raises(KeyError, match=r"Missing keys.*in feature"):
        mod.run(stack, prev, coord_columns=("x", "y"), use_time=False)


# --- Tests for k_means_clustering ---


@pytest.mark.parametrize(
    "normalise,time_weight",
    [
        (False, None),
        (True, None),
        (True, 0.5),
    ],
)
def test_kmeans_two_clusters(normalise, time_weight):
    """Test that KMeans correctly separates two spatially distinct clusters."""
    # two well-separated clusters in XY
    per_frame = [
        [{"centroid": (0.0, 0.0)}, {"centroid": (10.0, 10.0)}],
        [{"centroid": (1.0, -1.0)}, {"centroid": (11.0, 9.0)}],
    ]
    stack = DummyStack([0.0, 1.0])
    prev = make_prev(per_frame)

    mod = KMeansClusteringModule()
    out = mod.run(
        stack,
        prev,
        k=2,
        normalise=normalise,
        time_weight=time_weight,
    )

    # Expect exactly 2 clusters
    assert out["summary"]["n_clusters"] == 2
    # Each cluster must have at least one member
    for cnt in out["summary"]["members_per_cluster"].values():
        assert cnt >= 1


def test_kmeans_empty():
    """Test that KMeans returns empty results for no input features."""
    # no features at all
    per_frame = [[], []]
    stack = DummyStack([0.0, 1.0])
    prev = make_prev(per_frame)

    mod = KMeansClusteringModule()
    out = mod.run(stack, prev, k=3, normalise=False)
    assert out["summary"]["n_clusters"] == 0
    assert out["clusters"] == []
    assert out["cluster_centers"].shape == (0, 3)  # 3 dims by default


def test_kmeans_missing_dependency():
    """Test that KMeans raises if 'feature_detection' is missing."""
    stack = DummyStack([0.0])
    mod = KMeansClusteringModule()
    with pytest.raises(RuntimeError):
        mod.run(stack, previous_results={}, k=1)


def test_kmeans_missing_keys():
    """Test KMeans raises if coordinate keys are missing and no fallback present."""
    # feature dict missing centroid_x/centroid_y and no 'centroid' fallback
    per_frame = [[{"foo": 1}]]
    stack = DummyStack([0.0])
    prev = make_prev(per_frame)

    mod = KMeansClusteringModule()
    with pytest.raises(KeyError):
        mod.run(stack, prev, k=1, normalise=False)


# --- Tests for dbscan_clustering ---


@pytest.mark.parametrize(
    "eps,min_samples,expected_n",
    [
        # with very large eps and min_samples=1, everything collapses to one cluster
        (20.0, 1, 1),
        # with tiny eps & min_samples=1, each point stands alone → 2 clusters
        (0.1, 1, 2),
    ],
)
def test_dbscan_basic(eps, min_samples, expected_n):
    """Tets DBSCAN forms clusters based on eps and min_samples parameters."""
    per_frame = [
        [{"centroid": (0.0, 0.0)}],
        [{"centroid": (10.0, 0.0)}],
    ]
    stack = DummyStack([0.0, 1.0])
    prev = make_prev(per_frame)

    mod = DBSCANClusteringModule()
    out = mod.run(
        stack, prev, eps=eps, min_samples=min_samples, normalise=True, time_weight=None
    )
    assert out["summary"]["n_clusters"] == expected_n
    if expected_n > 0:
        # ensure cluster_centers count matches
        assert out["cluster_centers"].shape[0] == expected_n


def test_dbscan_empty():
    """Test DBSCAN returns zero clusters when no features are present."""
    per_frame = [[], []]
    stack = DummyStack([0.0, 1.0])
    prev = make_prev(per_frame)

    mod = DBSCANClusteringModule()
    out = mod.run(stack, prev, eps=1.0, min_samples=1)
    assert out["summary"]["n_clusters"] == 0
    assert out["clusters"] == []
    assert out["cluster_centers"].shape == (0, 3)


def test_dbscan_missing_dependency():
    """Test DBSCAN raises if required previous_results are missing."""
    stack = DummyStack([0.0])
    mod = DBSCANClusteringModule()
    with pytest.raises(RuntimeError):
        mod.run(stack, previous_results={}, eps=1.0, min_samples=1)


def test_dbscan_missing_keys():
    """Test DBSCAN raises if coordinate keys are missing in features."""
    per_frame = [[{"foo": 1}]]
    stack = DummyStack([0.0])
    prev = make_prev(per_frame)

    mod = DBSCANClusteringModule()
    with pytest.raises(KeyError):
        mod.run(stack, prev, eps=1.0, min_samples=1)


@pytest.mark.parametrize(
    "time_weight,expected_n",
    [
        (0.0, 1),  # time_weight=0 collapses to 1 cluster
        (10.0, 3),  # heavy time_weight separates all into 3 clusters
    ],
)
def test_dbscan_time_weight_effect(time_weight, expected_n):
    """Test DBSCAN splits clusters over time based on time_weight sensitivity."""
    # three identical-XY features on frames 0,1,2
    per_frame = [
        [{"centroid": (0.0, 0.0)}],
        [{"centroid": (0.0, 0.0)}],
        [{"centroid": (0.0, 0.0)}],
    ]
    stack = DummyStack([0.0, 1.0, 2.0])
    prev = make_prev(per_frame)

    mod = DBSCANClusteringModule()

    # suppress the divide-by-zero warning in the undo step:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        out = mod.run(
            stack,
            prev,
            eps=0.5,  # small XY radius so only time separation can split
            min_samples=1,
            normalise=True,
            time_weight=time_weight,
        )

    # correct cluster count
    assert out["summary"]["n_clusters"] == expected_n

    # only check center-time values when time_weight > 0
    if time_weight:
        centers = out["cluster_centers"]
        # time is third column
        times = centers[:, 2]
        # should all be in the original time range [0,2]
        assert np.all((times >= 0.0) & (times <= 2.0))


# --- Tests for count non zero pixels ---


class MockStack:
    """Minimal AFMImageStack mock with .data attribute."""

    def __init__(self, data):
        """Initialise the MockStack class with data."""
        self.data = data


def test_count_nonzero_basic():
    """Test that non-zero counts are computed correctly."""
    data = np.array(
        [
            [[0, 1], [2, 0]],  # 2 non-zeros
            [[0, 0], [0, 0]],  # 0 non-zeros
            [[3, 4], [5, 6]],  # 4 non-zeros
        ]
    )
    stack = MockStack(data)
    mod = CountNonzeroModule()
    result = mod.run(stack)

    expected = np.array([2, 0, 4])
    np.testing.assert_array_equal(result["counts"], expected)


def test_count_nonzero_all_zero():
    """Test with a stack where all pixels are zero."""
    data = np.zeros((5, 10, 10), dtype=int)
    stack = MockStack(data)
    mod = CountNonzeroModule()
    result = mod.run(stack)

    expected = np.zeros(5, dtype=int)
    np.testing.assert_array_equal(result["counts"], expected)


def test_count_nonzero_all_nonzero():
    """Test with a stack where all pixels are non-zero."""
    data = np.ones((3, 4, 4), dtype=int)  # 3 frames of 4x4 ones → 16 non-zeros each
    stack = MockStack(data)
    mod = CountNonzeroModule()
    result = mod.run(stack)

    expected = np.full(3, 16)
    np.testing.assert_array_equal(result["counts"], expected)


def test_count_nonzero_module_metadata():
    """Test version and name properties."""
    mod = CountNonzeroModule()
    assert mod.version == "0.1.0"
    assert mod.name == "count_nonzero"


# --- Test the previous results detection ---


@pytest.mark.parametrize(
    "ModuleClass",
    [
        XMeansClusteringModule,
        KMeansClusteringModule,
        DBSCANClusteringModule,
        ParticleTrackingModule,
    ],
)
def test_fallback_logic(ModuleClass):
    """Test The module fallback logic in grouping modules."""
    mod = ModuleClass()
    stack = make_dummy_stack()

    # Prepare arguments for run
    extra_kwargs = {}
    if ModuleClass is KMeansClusteringModule:
        extra_kwargs["k"] = 1  # required for KMeans

    # Raises if previous_results is None
    with pytest.raises(RuntimeError, match="requires previous results"):
        mod.run(stack, previous_results=None, **extra_kwargs)

    # Raises if required modules are missing
    with pytest.raises(RuntimeError, match="requires one of"):
        mod.run(stack, previous_results={"some_irrelevant_module": {}}, **extra_kwargs)

    dummy_features = {
        "features_per_frame": [[{"centroid": (0, 0), "label": 1}]],
        "labeled_masks": [np.array([[0, 1]])],
    }
    fallback_module = mod.requires[-1]
    previous_results = {fallback_module: dummy_features}

    result = mod.run(stack, previous_results=previous_results, **extra_kwargs)
    assert isinstance(result, dict)
