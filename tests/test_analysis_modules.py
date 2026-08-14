"""Tests for built in analysis modules."""

import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from playnano.afm_stack import AFMImageStack
from playnano.analysis.base import AnalysisModule
from playnano.analysis.modules import feature_detection, x_means_clustering
from playnano.analysis.modules.count_nonzero import CountNonzeroModule
from playnano.analysis.modules.dbscan_clustering import DBSCANClusteringModule
from playnano.analysis.modules.feature_detection import MASK_MAP, FeatureDetectionModule
from playnano.analysis.modules.k_means_clustering import KMeansClusteringModule
from playnano.analysis.modules.log_blob_detection import LoGBlobDetectionModule
from playnano.analysis.modules.particle_region_extraction import (
    ParticleRegionExtractionModule,
    _centered_fixed_bbox,
    _pad_bbox,
    _square_bbox,
    _tight_bbox_for_label,
)
from playnano.analysis.modules.particle_region_media_export import (
    ParticleRegionMediaExportModule,
    _assemble_track_stack,
    _filter_and_sort_tracks,
    _pad_crop_to_size,
)
from playnano.analysis.modules.particle_tracking import ParticleTrackingModule
from playnano.analysis.modules.tracked_particle_boundary_size import BoundarySizeModule
from playnano.analysis.modules.x_means_clustering import XMeansClusteringModule

# ==============================================================================
# Abstract base class
# ==============================================================================


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
    """Provide a dummy module for testing analysis module initialisation."""

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


# ==============================================================================
# FeatureDetectionModule
# ==============================================================================


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
    """Test that module requires either a mask function or key."""
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
    """Test that mask_fn returns invalid data shapes raise ValueError."""
    module = FeatureDetectionModule()
    stack = stack_1frame_with_timestamps

    # Define mask_fn returning wrong shape
    def bad_mask(frame: np.ndarray, **kwargs):
        return np.zeros((2, 2), dtype=bool)

    with pytest.raises(ValueError):
        module.run(stack, mask_fn=bad_mask)


def test_mask_key_not_in_previous_results(stack_1frame_with_timestamps):
    """Test that if mask_key not in previous result KeyError."""
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
    # One feature with area 3x3=9
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
        """Imitate a masking function."""
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
        morph_opening=False,
        sep_radius=6,
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


@pytest.fixture
def fd():
    """Return a bare FeatureDetectionModule instance."""
    return FeatureDetectionModule()


def test_separate_touching_zero_radius_returns_input(fd):
    """selem_radius <= 0 should return the input unchanged."""
    labeled = np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]], dtype=np.int32)
    result = fd._separate_touching_by_opening(labeled, selem_radius=0)
    np.testing.assert_array_equal(result, labeled)


def test_separate_touching_background_only(fd):
    """All-background mask should return all zeros."""
    labeled = np.zeros((5, 5), dtype=np.int32)
    result = fd._separate_touching_by_opening(labeled, selem_radius=1)
    assert result.sum() == 0


def test_separate_touching_single_large_blob_preserved(fd):
    """A single large blob that survives opening should be relabeled as label 1."""
    # 10x10 filled square — large enough to survive a radius-1 opening
    labeled = np.zeros((12, 12), dtype=np.int32)
    labeled[1:11, 1:11] = 1
    result = fd._separate_touching_by_opening(labeled, selem_radius=1)
    unique = np.unique(result)
    # Background + exactly one object
    assert set(unique) == {0, 1}
    # Object pixel count should be close to original (opening may trim edges slightly)
    assert (result == 1).sum() > 0


def test_separate_touching_two_touching_blobs_split(fd):
    """Two blobs touching along a thin bridge should be split into two labels."""
    # Two 7x7 squares connected by a single-pixel-wide bridge
    labeled = np.zeros((7, 20), dtype=np.int32)
    labeled[:, :7] = 1  # left blob
    labeled[3, 7:13] = 1  # thin bridge
    labeled[:, 13:] = 1  # right blob
    result = fd._separate_touching_by_opening(labeled, selem_radius=2)
    unique_labels = np.unique(result[result > 0])
    assert len(unique_labels) == 2


def test_separate_touching_small_blob_fallback(fd):
    """A blob too small to survive opening should fall back to its original shape."""
    # 3x3 blob — will be entirely eroded by radius-2 opening
    labeled = np.zeros((10, 10), dtype=np.int32)
    labeled[4:7, 4:7] = 1
    orig_area = int((labeled == 1).sum())
    result = fd._separate_touching_by_opening(
        labeled, selem_radius=2, max_area_loss=0.6
    )
    # Should fall back: area preserved
    assert (result > 0).sum() == orig_area


def test_separate_touching_labels_are_contiguous_from_one(fd):
    """Output labels should start from 1 and be contiguous integers."""
    labeled = np.zeros((10, 30), dtype=np.int32)
    labeled[1:9, 1:9] = 1
    labeled[1:9, 11:19] = 2
    labeled[1:9, 21:29] = 3
    result = fd._separate_touching_by_opening(labeled, selem_radius=1)
    unique = sorted(np.unique(result[result > 0]))
    assert unique == list(range(1, len(unique) + 1))


def test_separate_touching_output_dtype(fd):
    """Output array should have dtype int32."""
    labeled = np.zeros((8, 8), dtype=np.int32)
    labeled[2:6, 2:6] = 1
    result = fd._separate_touching_by_opening(labeled, selem_radius=1)
    assert result.dtype == np.int32


def test_separate_touching_output_shape_preserved(fd):
    """Output shape must match input shape."""
    labeled = np.zeros((15, 20), dtype=np.int32)
    labeled[2:8, 2:8] = 1
    result = fd._separate_touching_by_opening(labeled, selem_radius=1)
    assert result.shape == labeled.shape


def test_separate_touching_max_area_loss_zero_always_fallback(fd):
    """With max_area_loss=0, any area loss triggers fallback to original shape."""
    labeled = np.zeros((10, 10), dtype=np.int32)
    labeled[3:7, 3:7] = 1  # 4x4 blob, opening will trim corners
    orig_area = int((labeled == 1).sum())
    result = fd._separate_touching_by_opening(
        labeled, selem_radius=1, max_area_loss=0.0
    )
    assert (result > 0).sum() == orig_area


# ==============================================================================
# ParticleTrackingModule
# ==============================================================================


class MockAFMImageStack:
    """Simulate an AFMImageStack for testing particle tracking."""

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


class _StubTracker:
    """
    Minimal stub exposing only what _get_detection_outputs needs.

    Contains:
    - self.name
    - self.requires
    - _get_detection_outputs method (copied from your implementation)
    """

    name = "ParticleTrackingModule"
    requires = ["another_detection", "feature_detection"]

    def _get_detection_outputs(
        self,
        previous_results: dict[str, Any],
        *,
        detection_module: str,
        coord_key: str,
    ):
        if detection_module in previous_results:
            chosen = detection_module
        else:
            available = [m for m in reversed(self.requires) if m in previous_results]
            if not available:
                raise RuntimeError(
                    f"{self.name!r} requires one of {self.requires}, but none were found in previous results."  # noqa
                )
            chosen = available[0]

        fd_out = previous_results[chosen]

        if coord_key not in fd_out:
            raise RuntimeError(
                f"{self.name!r} expected detection output {chosen!r} to contain {coord_key!r}."  # noqa
            )
        if "labeled_masks" not in fd_out:
            raise RuntimeError(
                f"{self.name!r} expected detection output {chosen!r} to contain 'labeled_masks'."  # noqa
            )

        return fd_out[coord_key], fd_out["labeled_masks"]


def _mk_feats_masks():
    feats = [
        [{"centroid": (0.0, 0.0), "label": 1}],
        [{"centroid": (0.5, 0.5), "label": 2}],
    ]
    masks = [
        np.array([[0, 1], [1, 0]], dtype=int),
        np.array([[0, 2], [2, 0]], dtype=int),
    ]
    return feats, masks


def test_no_suitable_detection_module_found_raises():
    """Test that RuntimeError is triggered by no required detection module."""
    tracker = _StubTracker()
    previous_results = {
        # Intentionally empty or containing unrelated modules
        "unrelated": {}
    }

    with pytest.raises(RuntimeError) as ei:
        tracker._get_detection_outputs(
            previous_results,
            detection_module="feature_detection",  # preferred not present
            coord_key="features_per_frame",
        )

    msg = str(ei.value)
    assert "requires one of" in msg
    # Optional: ensure it mentions the requires list
    for req in tracker.requires:
        assert req in msg


def test_missing_coord_key_raises():
    """Test that a RuntimeError is raised when coord_key is missing."""
    tracker = _StubTracker()
    feats, masks = _mk_feats_masks()
    previous_results = {
        "feature_detection": {
            # "features_per_frame": feats,  # intentionally missing
            "labeled_masks": masks,
        }
    }

    with pytest.raises(RuntimeError) as ei:
        tracker._get_detection_outputs(
            previous_results,
            detection_module="feature_detection",
            coord_key="features_per_frame",
        )

    msg = str(ei.value)
    assert (
        "expected detection output 'feature_detection' to contain 'features_per_frame'"  # noqa
        in msg
    )


def test_missing_labeled_masks_raises():
    """Test that missing labeled masks raise a Runtime error."""
    tracker = _StubTracker()
    feats, _ = _mk_feats_masks()
    previous_results = {
        "feature_detection": {
            "features_per_frame": feats,
            # "labeled_masks": ...  # intentionally missing
        }
    }

    with pytest.raises(RuntimeError) as ei:
        tracker._get_detection_outputs(
            previous_results,
            detection_module="feature_detection",
            coord_key="features_per_frame",
        )

    msg = str(ei.value)
    assert (
        "expected detection output 'feature_detection' to contain 'labeled_masks'"  # noqa
        in msg
    )


def test_fallback_to_requires_latest_available_and_returns_data():
    """
    Test if the preferred detection_module is missing.

    Select the most recent available from self.requires in reverse order.
    requires = ["another_detection", "feature_detection"] → reversed is
    ["feature_detection", "another_detection"] If "feature_detection" exists
    in previous_results, it should be chosen.
    """
    tracker = _StubTracker()
    feats, masks = _mk_feats_masks()

    previous_results = {
        # preferred "another_detection" is NOT present
        "feature_detection": {
            "features_per_frame": feats,
            "labeled_masks": masks,
        }
    }

    out_feats, out_masks = tracker._get_detection_outputs(
        previous_results,
        detection_module="another_detection",  # preferred missing
        coord_key="features_per_frame",
    )

    # Should have selected "feature_detection" as it is
    # the first in reversed(self.requires) that exists
    assert out_feats == feats
    assert isinstance(out_masks, list)
    assert len(out_masks) == len(masks)


def test_preferred_present_is_used_even_if_others_exist():
    """Test that the preferred detection_module is present."""
    tracker = _StubTracker()
    feats1, masks1 = _mk_feats_masks()
    feats2, masks2 = _mk_feats_masks()

    previous_results = {
        "feature_detection": {
            "features_per_frame": feats1,
            "labeled_masks": masks1,
        },
        "another_detection": {
            "features_per_frame": feats2,
            "labeled_masks": masks2,
        },
    }

    # Preferred is "another_detection" and it exists; it must be chosen
    out_feats, out_masks = tracker._get_detection_outputs(
        previous_results,
        detection_module="another_detection",
        coord_key="features_per_frame",
    )

    assert out_feats == feats2
    assert out_masks == masks2


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


@pytest.mark.parametrize(
    "distance_scale,max_distance,expected_n_tracks",
    [
        ("constant", 0.6, 3),  # 0.707 > 0.6 → no links → 3 singleton tracks
        ("constant", 0.8, 1),  # 0.707 < 0.8 → link 0→1→2 → 1 track
        ("sqrt", 0.8, 1),  # for dt=1 these behave the same threshold-wise
        ("linear", 0.8, 1),
        ("constant", 0.6, 3),
    ],
    ids=lambda p: str(p),
)
def test_tracking_links_features(distance_scale, max_distance, expected_n_tracks):
    """
    Test that ParticleTrackingModule links features by nearest neighbor.

    Checks that that distance thresholds/scale modes gate linking as expected.
    """
    # 3 frames with a point moving along the diagonal by sqrt(0.5^2 + 0.5^2) ≈ 0.707
    features_per_frame = [
        [{"centroid": (0.0, 0.0), "label": 1}],  # frame 0
        [{"centroid": (0.5, 0.5), "label": 2}],  # frame 1
        [{"centroid": (1.0, 1.0), "label": 3}],  # frame 2
    ]
    labeled_masks = [
        np.array([[0, 1], [1, 0]], dtype=int),
        np.array([[0, 2], [2, 0]], dtype=int),
        np.array([[0, 3], [3, 0]], dtype=int),
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
        max_distance=max_distance,
        distance_scale=distance_scale,
    )

    # 1) Track count is as expected
    assert result["n_tracks"] == expected_n_tracks

    # Sort tracks by their first frame for stable assertions
    tracks = sorted(result["tracks"], key=lambda t: t["frames"][0])

    if expected_n_tracks == 1:
        # One long track across frames 0→1→2
        t = tracks[0]
        assert t["frames"] == [0, 1, 2]
        assert t["coords"] == [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]
        # The detected point is index 0 in each frame (there's only one point per frame)
        assert t["point_indices"] == [0, 0, 0]
    else:
        # Three singleton tracks (no links): frames [[0],[1],[2]]
        assert all(len(t["frames"]) == 1 for t in tracks)
        assert [t["frames"] for t in tracks] == [[0], [1], [2]]
        assert [t["coords"] for t in tracks] == [
            [(0.0, 0.0)],
            [(0.5, 0.5)],
            [(1.0, 1.0)],
        ]
        assert [t["point_indices"] for t in tracks] == [[0], [0], [0]]


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


# ==============================================================================
# LoGBlobDetectionModule
# ==============================================================================


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


# ==============================================================================
# Particle clustering — shared fixtures and helpers
# ==============================================================================


@pytest.fixture(autouse=True)
def patch_numpy_warnings():
    """Monkeypatch NumPy warnings for cleaner test output."""
    import numpy as _np

    _np.warnings = warnings
    yield
    del _np.warnings


# A minimal “stack” stub:
class DummyStack:
    """Provide a minimal AFMImageStack stub with only .time_for_frame support."""

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


# ==============================================================================
# XMeansClusteringModule
# ==============================================================================


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
    from playnano.analysis.modules import x_means_clustering

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
        "playnano.analysis.modules.x_means_clustering.core_xmeans", fake_core_xmeans
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


# ==============================================================================
# KMeansClusteringModule
# ==============================================================================


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


# ==============================================================================
# DBSCANClusteringModule
# ==============================================================================


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


# ==============================================================================
# CountNonzeroModule
# ==============================================================================


class MockStack:
    """Provide a minimal AFMImageStack mock exposing a .data attribute."""

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


# ==============================================================================
# Previous-results fallback detection
# ==============================================================================


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


# ==============================================================================
# ParticleRegionExtractionModule
# ==============================================================================
# ---------------------------------------------------------------------------
# Shared stack stub and helpers
# ---------------------------------------------------------------------------

_H, _W = 4, 4  # frame dimensions used throughout


class _RegionStack:
    """Provide a minimal stack stub supporting integer indexing and time_for_frame."""

    def __init__(self, data: np.ndarray, timestamps=None, pixel_size_nm: float = 1.0):
        self._data = data  # shape (N, H, W)
        self._timestamps = (
            timestamps if timestamps is not None else list(range(len(data)))
        )
        self.pixel_size_nm = pixel_size_nm
        self.file_path = "dummy.nhf"

    def __getitem__(self, idx: int) -> np.ndarray:
        return self._data[idx]

    def time_for_frame(self, idx: int) -> float:
        return float(self._timestamps[idx])


def _make_stack(n_frames: int = 2, value: float = 5.0) -> _RegionStack:
    """Return a 4x4 stack with a 2x2 particle at rows 1–2, cols 1–2 in every frame."""
    data = np.zeros((n_frames, _H, _W))
    data[:, 1:3, 1:3] = value
    return _RegionStack(data, timestamps=[i * 0.5 for i in range(n_frames)])


def _make_labeled_mask() -> np.ndarray:
    """Return a 4x4 labeled mask with a single particle labeled as 1 in the center."""
    lm = np.zeros((_H, _W), dtype=int)
    lm[1:3, 1:3] = 1
    return lm


def _make_feature() -> dict:
    return {"label": 1, "centroid": (1.5, 1.5), "area": 4, "mean": 5.0}


# ---------------------------------------------------------------------------
# Fixtures : extraction module
# ---------------------------------------------------------------------------


@pytest.fixture
def exmod():
    """Return a ParticleRegionExtractionModule instance."""
    return ParticleRegionExtractionModule()


@pytest.fixture
def small_stack():
    """Return a small two-frame 4x4 test stack."""
    return _make_stack()


@pytest.fixture
def detection_out():
    """Return a minimal feature detection output dict for testing."""
    lm = _make_labeled_mask()
    feat = _make_feature()
    return {
        "labeled_masks": [lm.copy(), lm.copy()],
        "features_per_frame": [[feat], [feat]],
    }


@pytest.fixture
def tracking_out():
    """Return a minimal particle tracking output dict for testing."""
    return {
        "tracks": [
            {
                "id": 0,
                "frames": [0, 1],
                "point_indices": [0, 0],
                "coords": [(1.5, 1.5), (1.5, 1.5)],
            }
        ]
    }


@pytest.fixture
def prev(detection_out, tracking_out):
    """Return a previous_results dict containing detection and tracking outputs."""
    return {
        "feature_detection": detection_out,
        "particle_tracking": tracking_out,
    }


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


def test_extraction_name(exmod):
    """Check the name of the extraction module."""
    assert exmod.name == "particle_region_extraction"


def test_extraction_version(exmod):
    """Check the version property of the extraction module."""
    assert exmod.version == "0.1.0"


def test_extraction_requires_particle_tracking(exmod):
    """Test that extraction module requires tracking."""
    assert "particle_tracking" in exmod.requires


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_extraction_none_previous_results_raises(exmod, small_stack):
    """Test that passing None for previous_results raises a RuntimeError."""
    with pytest.raises(RuntimeError, match="requires previous results"):
        exmod.run(small_stack, previous_results=None)


def test_extraction_missing_tracking_module_raises(exmod, small_stack, detection_out):
    """Test missing tracking_module in previous_results raises a RuntimeError."""
    with pytest.raises(RuntimeError, match="tracking_module"):
        exmod.run(small_stack, {"feature_detection": detection_out})


def test_extraction_missing_tracks_key_raises(exmod, small_stack, detection_out):
    """Test missing 'tracks' key in particle_tracking raises a RuntimeError."""
    with pytest.raises(RuntimeError, match="'tracks'"):
        exmod.run(
            small_stack,
            {
                "feature_detection": detection_out,
                "particle_tracking": {"not_tracks": []},
            },
        )


def test_extraction_missing_detection_module_raises(exmod, small_stack, tracking_out):
    """Test that missing detections raises a RuntimeError."""
    with pytest.raises(RuntimeError, match="detection_module"):
        exmod.run(small_stack, {"particle_tracking": tracking_out})


def test_extraction_missing_labeled_masks_raises(exmod, small_stack, tracking_out):
    """Test that a RuntimeError is raised when labeled masks are missing."""
    with pytest.raises(RuntimeError, match="labeled_masks"):
        exmod.run(
            small_stack,
            {
                "feature_detection": {"features_per_frame": []},
                "particle_tracking": tracking_out,
            },
        )


def test_extraction_invalid_fixed_size_mode_raises(exmod, small_stack, prev):
    """Tets that invalid fixed_size_mode raises ValueError."""
    with pytest.raises(ValueError, match="fixed_size_mode"):
        exmod.run(small_stack, prev, fixed_size_mode="not_valid")


def test_extraction_missing_label_key_raises(exmod, small_stack):
    """Test that missing 'label' key in feature dict raises RuntimeError."""
    lm = _make_labeled_mask()
    bad_det = {
        "labeled_masks": [lm],
        "features_per_frame": [[{"no_label_here": 1}]],
    }
    track = {
        "tracks": [
            {"id": 0, "frames": [0], "point_indices": [0], "coords": [(1.5, 1.5)]}
        ]
    }
    with pytest.raises(RuntimeError, match="'label'"):
        exmod.run(
            small_stack, {"feature_detection": bad_det, "particle_tracking": track}
        )


def test_extraction_bad_stack_indexing_raises(exmod):
    """Stack that doesn't support integer indexing raises RuntimeError."""

    class _BadStack:
        pixel_size_nm = 1.0

        def __getitem__(self, idx):
            raise TypeError("not subscriptable")

        def time_for_frame(self, idx):
            return 0.0

    lm = _make_labeled_mask()
    det = {"labeled_masks": [lm], "features_per_frame": [[_make_feature()]]}
    track = {
        "tracks": [
            {"id": 0, "frames": [0], "point_indices": [0], "coords": [(1.5, 1.5)]}
        ]
    }
    with pytest.raises(RuntimeError, match=r"stack\[frame_idx\]"):
        exmod.run(_BadStack(), {"feature_detection": det, "particle_tracking": track})


# ---------------------------------------------------------------------------
# Empty and zero-frame cases
# ---------------------------------------------------------------------------


def test_extraction_zero_frames_returns_empty(exmod, small_stack, tracking_out):
    """Test that running with zero frames produces empty per_track and summary."""
    out = exmod.run(
        small_stack,
        {
            "feature_detection": {"labeled_masks": [], "features_per_frame": []},
            "particle_tracking": tracking_out,
        },
    )
    assert out["per_track"] == []
    assert out["flat_table"] == []
    assert out["summary"]["n_tracks"] == 0


def test_extraction_no_tracks_returns_empty(exmod, small_stack, detection_out):
    """Test that no tracks in tracking output produces empty per_track and summary."""
    out = exmod.run(
        small_stack,
        {
            "feature_detection": detection_out,
            "particle_tracking": {"tracks": []},
        },
    )
    assert out["per_track"] == []
    assert out["summary"]["n_tracks"] == 0


# ---------------------------------------------------------------------------
# Core functionality
# ---------------------------------------------------------------------------


def test_extraction_bbox_tight(exmod, small_stack, prev):
    """Test that bbox_tight is correctly computed from the labeled mask."""
    out = exmod.run(small_stack, prev)
    region = out["per_track"][0]["regions"][0]
    assert region["bbox_tight"] == (1, 1, 3, 3)


def test_extraction_bbox_padded_equals_tight_when_no_padding(exmod, small_stack, prev):
    """Test that bbox_padded equals bbox_tight when padding is zero."""
    out = exmod.run(small_stack, prev)
    region = out["per_track"][0]["regions"][0]
    assert region["bbox_padded"] == region["bbox_tight"]


def test_extraction_image_crop_shape(exmod, small_stack, prev):
    """Test that the image_crop shape matches the tight bbox."""
    # tight bbox (1,1,3,3) → 2x2 crop
    out = exmod.run(small_stack, prev)
    crop = out["per_track"][0]["regions"][0]["image_crop"]
    assert crop.shape == (2, 2)


def test_extraction_image_crop_values(exmod, small_stack, prev):
    """Test image_crop values match expected constant value in the small test stack."""
    out = exmod.run(small_stack, prev)
    crop = out["per_track"][0]["regions"][0]["image_crop"]
    np.testing.assert_array_equal(crop, 5.0 * np.ones((2, 2)))


def test_extraction_mask_crop_all_true(exmod, small_stack, prev):
    """Test that the mask_crop is all True for the small test stack region."""
    out = exmod.run(small_stack, prev)
    mask = out["per_track"][0]["regions"][0]["mask_crop"]
    assert mask.shape == (2, 2)
    assert mask.all()


def test_extraction_region_label(exmod, small_stack, prev):
    """Test that the region label matches the feature label."""
    out = exmod.run(small_stack, prev)
    assert out["per_track"][0]["regions"][0]["label"] == 1


def test_extraction_timestamps(exmod, small_stack, prev):
    """Test that timestamps are correctly extracted from the stack."""
    out = exmod.run(small_stack, prev)
    assert out["per_track"][0]["timestamps"] == pytest.approx([0.0, 0.5])


def test_extraction_two_frames_two_regions(exmod, small_stack, prev):
    """Test that a track with two frames produces two regions in the output."""
    out = exmod.run(small_stack, prev)
    regions = out["per_track"][0]["regions"]
    assert len(regions) == 2
    assert all(r is not None for r in regions)


# ---------------------------------------------------------------------------
# include_* flags
# ---------------------------------------------------------------------------


def test_extraction_include_image_false(exmod, small_stack, prev):
    """Test that include_image=False removes image_crop from region dict."""
    out = exmod.run(small_stack, prev, include_image=False)
    region = out["per_track"][0]["regions"][0]
    assert "image_crop" not in region


def test_extraction_include_mask_false(exmod, small_stack, prev):
    """Test that include_mask=False removes mask_crop from region dict."""
    out = exmod.run(small_stack, prev, include_mask=False)
    region = out["per_track"][0]["regions"][0]
    assert "mask_crop" not in region


def test_extraction_include_bbox_false_removes_region_keys(exmod, small_stack, prev):
    """Test that include_bbox=False removes bbox keys from region dict."""
    out = exmod.run(small_stack, prev, include_bbox=False)
    region = out["per_track"][0]["regions"][0]
    assert "bbox_tight" not in region
    assert "bbox_padded" not in region


def test_extraction_include_bbox_false_removes_flat_table_columns(
    exmod, small_stack, prev
):
    """Test that include_bbox=False removes bbox columns from flat_table."""
    out = exmod.run(small_stack, prev, include_bbox=False)
    row = out["flat_table"][0]
    assert "bbox_tight_minr" not in row
    assert "bbox_padded_minr" not in row


# ---------------------------------------------------------------------------
# Missing detections (dense tracks)
# ---------------------------------------------------------------------------


def test_extraction_none_pt_idx_produces_none(exmod, small_stack, detection_out):
    """Test that point_indices=None produces None region and counts as missing."""
    track_with_gap = {
        "tracks": [
            {
                "id": 0,
                "frames": [0, 1],
                "point_indices": [None, 0],
                "coords": [None, (1.5, 1.5)],
            }
        ]
    }
    out = exmod.run(
        small_stack,
        {"feature_detection": detection_out, "particle_tracking": track_with_gap},
    )
    regions = out["per_track"][0]["regions"]
    assert regions[0] is None
    assert regions[1] is not None


def test_extraction_none_pt_idx_increments_missing(exmod, small_stack, detection_out):
    """Test a track with None point_indices increments the missing count in summary."""
    track_with_gap = {
        "tracks": [
            {
                "id": 0,
                "frames": [0, 1],
                "point_indices": [None, 0],
                "coords": [None, (1.5, 1.5)],
            }
        ]
    }
    out = exmod.run(
        small_stack,
        {"feature_detection": detection_out, "particle_tracking": track_with_gap},
    )
    assert out["summary"]["n_missing_region_measurements"] == 1


def test_extraction_none_pt_idx_flat_table_has_nan_bbox(
    exmod, small_stack, detection_out
):
    """Test that flat_table has NaN for bbox values when point_indices is None."""
    track_with_gap = {
        "tracks": [
            {
                "id": 0,
                "frames": [0],
                "point_indices": [None],
                "coords": [None],
            }
        ]
    }
    out = exmod.run(
        small_stack,
        {"feature_detection": detection_out, "particle_tracking": track_with_gap},
    )
    row = out["flat_table"][0]
    assert np.isnan(row["bbox_tight_minr"])


# ---------------------------------------------------------------------------
# Out-of-range guards
# ---------------------------------------------------------------------------


def test_extraction_out_of_range_frame_warns_and_nones(
    exmod, small_stack, detection_out
):
    """Test track with a frame index out of range produces  warning and None region."""
    bad_track = {
        "tracks": [{"id": 0, "frames": [99], "point_indices": [0], "coords": [(0, 0)]}]
    }
    with pytest.warns(UserWarning, match="out of range"):
        out = exmod.run(
            small_stack,
            {"feature_detection": detection_out, "particle_tracking": bad_track},
        )
    assert out["per_track"][0]["regions"][0] is None
    assert out["summary"]["n_skipped_index_errors"] == 1


def test_extraction_out_of_range_point_warns_and_nones(
    exmod, small_stack, detection_out
):
    """Test out-of-range point_indices in tracks produce None regions and a warning."""
    bad_track = {
        "tracks": [{"id": 0, "frames": [0], "point_indices": [99], "coords": [(0, 0)]}]
    }
    with pytest.warns(UserWarning, match="out of range"):
        out = exmod.run(
            small_stack,
            {"feature_detection": detection_out, "particle_tracking": bad_track},
        )
    assert out["per_track"][0]["regions"][0] is None
    assert out["summary"]["n_skipped_index_errors"] == 1


def test_extraction_mismatched_lengths_truncates_not_raises(
    exmod, small_stack, detection_out
):
    """strict=False fix: mismatched frames/point_indices should truncate, not raise."""
    mismatched = {
        "tracks": [
            {
                "id": 0,
                "frames": [0, 1],
                "point_indices": [0],
                "coords": [(1.5, 1.5)],
            }
        ]
    }
    with pytest.warns(UserWarning, match="Truncating to shortest"):
        out = exmod.run(
            small_stack,
            {"feature_detection": detection_out, "particle_tracking": mismatched},
        )
    # Should produce 1 region (truncated to shortest), not raise ValueError
    assert len(out["per_track"][0]["regions"]) == 1


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------


def test_extraction_padding_expands_padded_bbox(exmod, small_stack, prev):
    """Test that adding padding expands bbox."""
    out = exmod.run(small_stack, prev, padding=1)
    region = out["per_track"][0]["regions"][0]
    # tight (1,1,3,3) + pad 1, clipped to 4x4 → (0,0,4,4)
    assert region["bbox_padded"] == (0, 0, 4, 4)
    assert region["bbox_tight"] == (1, 1, 3, 3)


def test_extraction_padding_crop_covers_padded_region(exmod, small_stack, prev):
    """Test that image_crop shape matches padded bbox when padding is applied."""
    out = exmod.run(small_stack, prev, padding=1)
    crop = out["per_track"][0]["regions"][0]["image_crop"]
    assert crop.shape == (4, 4)  # full 4x4 frame after padding


def test_extraction_padding_clipped_at_image_boundary(exmod, prev):
    """A particle at the image edge: padded bbox must not exceed image bounds."""
    data = np.zeros((2, 4, 4))
    data[:, 0:2, 0:2] = 5.0  # corner particle
    stack = _RegionStack(data, timestamps=[0.0, 0.5])
    lm = np.zeros((4, 4), dtype=int)
    lm[0:2, 0:2] = 1
    feat = {"label": 1, "centroid": (0.5, 0.5), "area": 4}
    corner_det = {
        "labeled_masks": [lm.copy(), lm.copy()],
        "features_per_frame": [[feat], [feat]],
    }
    out = exmod.run(
        stack,
        {
            "feature_detection": corner_det,
            "particle_tracking": prev["particle_tracking"],
        },
        padding=5,
    )
    bbox = out["per_track"][0]["regions"][0]["bbox_padded"]
    assert bbox[0] >= 0 and bbox[1] >= 0
    assert bbox[2] <= 4 and bbox[3] <= 4


# ---------------------------------------------------------------------------
# Square and fixed box size
# ---------------------------------------------------------------------------


def test_extraction_square_produces_square_bbox(exmod):
    """A non-square particle bbox should become square when square=True."""
    data = np.zeros((1, 6, 6))
    data[0, 2, 2:5] = 5.0  # 1-row x 3-col particle
    stack = _RegionStack(data, timestamps=[0.0])
    lm = np.zeros((6, 6), dtype=int)
    lm[2, 2:5] = 1
    det = {
        "labeled_masks": [lm],
        "features_per_frame": [[{"label": 1, "centroid": (2, 3), "area": 3}]],
    }
    track = {
        "tracks": [{"id": 0, "frames": [0], "point_indices": [0], "coords": [(2, 3)]}]
    }
    out = exmod.run(
        stack, {"feature_detection": det, "particle_tracking": track}, square=True
    )
    bbox = out["per_track"][0]["regions"][0]["bbox_padded"]
    assert (bbox[2] - bbox[0]) == (bbox[3] - bbox[1])


def test_extraction_fixed_box_size_global(exmod, small_stack, prev):
    """Test fixed_box_size in global mode: bbox size = fixed_box_size."""
    out = exmod.run(small_stack, prev, fixed_box_size=4)
    bbox = out["per_track"][0]["regions"][0]["bbox_padded"]
    assert (bbox[2] - bbox[0]) == 4
    assert (bbox[3] - bbox[1]) == 4


def test_extraction_fixed_box_size_per_track_uses_max_detection(
    exmod, small_stack, prev
):
    """Test per_track mode: fixed size = max detection dim + 2*padding."""
    out = exmod.run(
        small_stack, prev, fixed_box_size=100, fixed_size_mode="per_track", padding=0
    )
    bbox = out["per_track"][0]["regions"][0]["bbox_padded"]
    # max detection dim across both frames = 2, padding=0 → size should be 2
    assert (bbox[2] - bbox[0]) == 2
    assert (bbox[3] - bbox[1]) == 2


def test_extraction_fixed_box_size_per_track_with_padding(exmod, small_stack, prev):
    """Test per_track with padding: size = max_dim + 2*padding."""
    out = exmod.run(
        small_stack, prev, fixed_box_size=100, fixed_size_mode="per_track", padding=1
    )
    bbox = out["per_track"][0]["regions"][0]["bbox_padded"]
    # max_dim=2, padding=1 → size = 2 + 2*1 = 4 (clipped to 4x4 image)
    assert (bbox[2] - bbox[0]) <= 4 and (bbox[3] - bbox[1]) <= 4


# ---------------------------------------------------------------------------
# flat_table and summary
# ---------------------------------------------------------------------------


def test_extraction_flat_table_required_columns(exmod, small_stack, prev):
    """Test flat_table contains all expected required columns."""
    out = exmod.run(small_stack, prev)
    row = out["flat_table"][0]
    for col in ("track_id", "frame", "timestamp", "label"):
        assert col in row, f"Missing column: {col}"


def test_extraction_flat_table_bbox_columns_present(exmod, small_stack, prev):
    """Test flat_table contains all expected bbox columns when include_bbox=True."""
    out = exmod.run(small_stack, prev)
    row = out["flat_table"][0]
    for col in (
        "bbox_tight_minr",
        "bbox_tight_minc",
        "bbox_tight_maxr",
        "bbox_tight_maxc",
        "bbox_padded_minr",
        "bbox_padded_minc",
        "bbox_padded_maxr",
        "bbox_padded_maxc",
    ):
        assert col in row, f"Missing column: {col}"


def test_extraction_flat_table_bbox_values(exmod, small_stack, prev):
    """Test the flat_table bbox values match expected tight bbox for small stack."""
    out = exmod.run(small_stack, prev)
    row = out["flat_table"][0]
    assert row["bbox_tight_minr"] == 1
    assert row["bbox_tight_minc"] == 1
    assert row["bbox_tight_maxr"] == 3
    assert row["bbox_tight_maxc"] == 3


def test_extraction_flat_table_row_count(exmod, small_stack, prev):
    """Test that the flat_table has one row per frame in the track."""
    out = exmod.run(small_stack, prev)
    assert len(out["flat_table"]) == 2  # one row per frame


def test_extraction_summary_keys(exmod, small_stack, prev):
    """Check the summary dictionary contains all expected keys."""
    out = exmod.run(small_stack, prev)
    for key in (
        "n_tracks",
        "n_rows",
        "n_skipped_index_errors",
        "n_missing_region_measurements",
        "padding",
        "fixed_box_size",
        "fixed_size_mode",
        "square",
        "include_image",
        "include_mask",
        "include_bbox",
    ):
        assert key in out["summary"], f"Missing summary key: {key}"


def test_extraction_summary_counts(exmod, small_stack, prev):
    """Test that summary counts match the expected values for the small stack."""
    out = exmod.run(small_stack, prev)
    assert out["summary"]["n_tracks"] == 1
    assert out["summary"]["n_rows"] == 2
    assert out["summary"]["n_skipped_index_errors"] == 0
    assert out["summary"]["n_missing_region_measurements"] == 0


# ==============================================================================
# Extraction Geometry helpers
# ==============================================================================


class TestTightBboxForLabel:
    """Test the _tight_bbox_for_label geometry helper."""

    def test_absent_label_returns_none(self):
        """Label not present in mask → None."""
        lm = np.zeros((5, 5), dtype=int)
        assert _tight_bbox_for_label(lm, label_val=1) is None

    def test_all_zeros_returns_none(self):
        """Label not present in an all-zeros mask → None."""
        lm = np.zeros((4, 4), dtype=int)
        assert _tight_bbox_for_label(lm, label_val=1) is None

    def test_single_pixel(self):
        """Single foreground pixel at (2, 2) → bbox (2, 2, 3, 3)."""
        lm = np.zeros((5, 5), dtype=int)
        lm[2, 2] = 1
        assert _tight_bbox_for_label(lm, 1) == (2, 2, 3, 3)

    def test_rectangular_region(self):
        """2-row x 3-col region at rows 1-2, cols 2-4 → (1, 2, 3, 5)."""
        lm = np.zeros((6, 6), dtype=int)
        lm[1:3, 2:5] = 1
        assert _tight_bbox_for_label(lm, 1) == (1, 2, 3, 5)

    def test_full_frame_region(self):
        """Region covering the entire frame → (0, 0, H, W)."""
        H, W = 4, 6
        lm = np.ones((H, W), dtype=int)
        assert _tight_bbox_for_label(lm, 1) == (0, 0, H, W)

    def test_correct_label_chosen(self):
        """Only the requested label is returned, not a different one."""
        lm = np.zeros((6, 6), dtype=int)
        lm[0:2, 0:2] = 1  # label 1: rows 0-1, cols 0-1
        lm[4:6, 4:6] = 2  # label 2: rows 4-5, cols 4-5
        assert _tight_bbox_for_label(lm, 1) == (0, 0, 2, 2)
        assert _tight_bbox_for_label(lm, 2) == (4, 4, 6, 6)

    def test_disconnected_components_returns_largest(self):
        """Two disconnected blobs sharing a label → bbox of the largest component."""
        lm = np.zeros((6, 6), dtype=int)
        lm[0, 0] = 1  # small: 1 px
        lm[2:5, 2:5] = 1  # large: 9 px → bbox (2, 2, 5, 5)
        assert _tight_bbox_for_label(lm, 1) == (2, 2, 5, 5)

    def test_returns_integer_tuple(self):
        """All four bbox values must be plain Python ints."""
        lm = np.zeros((4, 4), dtype=int)
        lm[1:3, 1:3] = 1
        bbox = _tight_bbox_for_label(lm, 1)
        assert all(isinstance(v, int) for v in bbox)

    def test_non_integer_mask_coerced(self):
        """Float mask is coerced to bool; non-zero pixels treated as foreground."""
        lm = np.zeros((4, 4), dtype=float)
        lm[1, 1] = 3.7
        # label_val=1: binary=(lm==1) is False everywhere → None
        assert _tight_bbox_for_label(lm, 1) is None


# ---------------------------------------------------------------------------
# _pad_bbox
# ---------------------------------------------------------------------------


class TestPadBbox:
    """Test the _pad_bbox bounding-box expansion helper."""

    def test_zero_padding_unchanged(self):
        """Test that zero padding returns original bbox."""
        assert _pad_bbox((1, 1, 3, 3), 0, (4, 4)) == (1, 1, 3, 3)

    def test_padding_within_bounds(self):
        """Padding that does not hit the image boundary."""
        # (2,2,4,4) + pad 1 in 8x8 → (1,1,5,5)
        assert _pad_bbox((2, 2, 4, 4), 1, (8, 8)) == (1, 1, 5, 5)

    def test_padding_clips_to_zero(self):
        """Padding that would go negative is clipped to 0."""
        # (0,0,2,2) + pad 2 → top-left would be (-2,-2), clips to (0,0)
        result = _pad_bbox((0, 0, 2, 2), 2, (6, 6))
        assert result[0] == 0
        assert result[1] == 0

    def test_padding_clips_to_image_boundary(self):
        """Padding that would exceed image dims is clipped."""
        # (2,2,4,4) + pad 5 in 4x4 → clipped at (4,4)
        result = _pad_bbox((2, 2, 4, 4), 5, (4, 4))
        assert result[2] == 4
        assert result[3] == 4

    def test_full_clip_all_sides(self):
        """Corner bbox padded far beyond image → entire image."""
        assert _pad_bbox((1, 1, 3, 3), 10, (4, 4)) == (0, 0, 4, 4)

    def test_result_dimensions_grow_symmetrically(self):
        """With no clipping, each side expands by exactly padding."""
        bbox = (3, 3, 5, 5)
        p = 2
        r0, c0, r1, c1 = _pad_bbox(bbox, p, (10, 10))
        assert r0 == bbox[0] - p
        assert c0 == bbox[1] - p
        assert r1 == bbox[2] + p
        assert c1 == bbox[3] + p

    def test_non_square_image(self):
        """Works correctly for non-square images (H ≠ W)."""
        result = _pad_bbox((1, 1, 3, 3), 2, (5, 8))
        assert result[2] <= 5  # height clipped to H
        assert result[3] <= 8  # width clipped to W


# ---------------------------------------------------------------------------
# _centered_fixed_bbox
# ---------------------------------------------------------------------------


class TestCenteredFixedBbox:
    """Test the _centered_fixed_bbox fixed-size centring helper."""

    def test_output_dimensions_equal_size(self):
        """Returned bbox must have height == width == size when not clipped."""
        bbox = (1, 1, 3, 3)  # centre at (2, 2) in a 6x6 image
        r0, c0, r1, c1 = _centered_fixed_bbox(bbox, 4, (6, 6))
        assert r1 - r0 == 4
        assert c1 - c0 == 4

    def test_centered_on_input_bbox_center(self):
        """Output is centred on the centre of the input bbox."""
        # Input centre at row=(1+3)//2=2, col=(1+3)//2=2; size=4, half=2
        # expected: (0,0,4,4) in a 6x6 image
        assert _centered_fixed_bbox((1, 1, 3, 3), 4, (6, 6)) == (0, 0, 4, 4)

    def test_clipped_near_top_left(self):
        """Bbox near top-left corner is shifted to maintain size."""
        # Centre of (0,0,2,2) is (1,1); size=4, half=2 → would be (-1,-1,3,3);
        # clipped to (0,0,4,4)
        r0, c0, r1, c1 = _centered_fixed_bbox((0, 0, 2, 2), 4, (4, 4))
        assert r0 == 0 and c0 == 0
        assert r1 - r0 == 4 and c1 - c0 == 4

    def test_clipped_near_bottom_right(self):
        """Bbox near bottom-right corner is shifted to maintain size."""
        # Centre of (4,4,6,6) in 6x6 is (5,5); size=4 → clips, re-shifted
        r0, c0, r1, c1 = _centered_fixed_bbox((4, 4, 6, 6), 4, (6, 6))
        assert r1 == 6 and c1 == 6
        assert r1 - r0 == 4 and c1 - c0 == 4

    def test_odd_size(self):
        """Odd size is handled by floor division; output still has correct size."""
        r0, c0, r1, c1 = _centered_fixed_bbox((1, 1, 3, 3), 3, (6, 6))
        assert r1 - r0 == 3
        assert c1 - c0 == 3

    def test_size_equals_image(self):
        """Size equal to image dimensions → full image returned."""
        H, W = 5, 5
        r0, c0, r1, c1 = _centered_fixed_bbox((1, 1, 4, 4), H, (H, W))
        assert (r0, c0, r1, c1) == (0, 0, H, W)

    def test_output_within_image_bounds(self):
        """Output coords must always be within image boundaries."""
        for centre in [(0, 0, 1, 1), (4, 4, 6, 6), (2, 2, 4, 4)]:
            r0, c0, r1, c1 = _centered_fixed_bbox(centre, 4, (6, 6))
            assert r0 >= 0 and c0 >= 0
            assert r1 <= 6 and c1 <= 6


# ---------------------------------------------------------------------------
# _square_bbox
# ---------------------------------------------------------------------------


class TestSquareBbox:
    """Test the _square_bbox squaring helper."""

    def test_already_square_unchanged(self):
        """A square bbox is returned unchanged (same coords)."""
        bbox = (1, 1, 3, 3)
        assert _square_bbox(bbox, (6, 6)) == bbox

    def test_wide_bbox_becomes_square(self):
        """A wider-than-tall bbox: height is extended to match width."""
        # (1,1,2,4) → height=1, width=3 → size=3
        r0, c0, r1, c1 = _square_bbox((1, 1, 2, 4), (8, 8))
        assert r1 - r0 == c1 - c0

    def test_tall_bbox_becomes_square(self):
        """A taller-than-wide bbox: width is extended to match height."""
        # (1,1,4,2) → height=3, width=1 → size=3
        r0, c0, r1, c1 = _square_bbox((1, 1, 4, 2), (8, 8))
        assert r1 - r0 == c1 - c0

    def test_square_side_equals_max_dim(self):
        """Side of the output square equals max(height, width) of the input."""
        r0, c0, r1, c1 = _square_bbox((0, 0, 2, 5), (10, 10))
        assert r1 - r0 == 5
        assert c1 - c0 == 5

    def test_output_centred_on_input(self):
        """Output is centred on the centre of the input bbox (when not clipped)."""
        # (2,2,4,6) → centre row=(2+4)//2=3, col=(2+6)//2=4; size=4, half=2
        # expected row: (3-2, 3+2) = (1,5); col: (4-2, 4+2) = (2,6)
        r0, c0, r1, c1 = _square_bbox((2, 2, 4, 6), (10, 10))
        assert r0 == 1 and r1 == 5
        assert c0 == 2 and c1 == 6

    def test_clips_to_image_boundary(self):
        """Expansion near the image edge is clipped and re-shifted."""
        # wide bbox near top edge: (0,0,1,6) in 8x8 → size=6, clips top
        r0, c0, r1, c1 = _square_bbox((0, 0, 1, 6), (8, 8))
        assert r0 >= 0 and c0 >= 0
        assert r1 <= 8 and c1 <= 8
        # output is still square (may be slightly smaller if double-clipped)
        assert r1 - r0 == c1 - c0

    def test_single_pixel_becomes_1x1_square(self):
        """A 1x1 bbox (single pixel) is already square and stays 1x1."""
        r0, c0, r1, c1 = _square_bbox((2, 3, 3, 4), (6, 6))
        assert r1 - r0 == 1
        assert c1 - c0 == 1

    def test_output_within_image_bounds(self):
        """Output coords must always lie within image boundaries."""
        for bbox in [(0, 0, 1, 4), (3, 0, 4, 6), (0, 3, 6, 4)]:
            r0, c0, r1, c1 = _square_bbox(bbox, (6, 6))
            assert r0 >= 0 and c0 >= 0, f"underflow for {bbox}"
            assert r1 <= 6 and c1 <= 6, f"overflow for {bbox}"


# ==============================================================================
# BoundarySizeModule
# ==============================================================================

# ---------------------------------------------------------------------------
# Fixtures : BoundarySizeModule
# ---------------------------------------------------------------------------


@pytest.fixture
def bsmod():
    """Create and return a BoundarySizeModule instance."""
    return BoundarySizeModule()


def _make_extraction_out(
    bbox_tight: tuple = (1, 1, 3, 3),
    n_frames: int = 2,
    include_none: bool = False,
) -> dict:
    """Build a minimal particle_region_extraction output for BoundarySizeModule."""
    regions = []
    for i in range(n_frames):
        if include_none and i == 0:
            regions.append(None)
        else:
            regions.append(
                {
                    "frame": i,
                    "timestamp": float(i) * 0.5,
                    "label": 1,
                    "bbox_tight": bbox_tight,
                    "bbox_padded": bbox_tight,
                }
            )
    return {
        "summary": {"include_bbox": True},
        "per_track": [
            {
                "track_id": 0,
                "frames": list(range(n_frames)),
                "timestamps": [float(i) * 0.5 for i in range(n_frames)],
                "regions": regions,
            }
        ],
    }


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


def test_boundary_name(bsmod):
    """Check that the name string is correct and matches the module's __name__."""
    assert bsmod.name == "tracked_particle_boundary_size"


def test_boundary_version(bsmod):
    """Check that the version string is correct and matches the module's __version__."""
    assert bsmod.version == "0.1.0"


def test_boundary_requires_particle_region_extraction(bsmod):
    """Test boundary_size requires particle_region_extraction in previous results."""
    assert "particle_region_extraction" in bsmod.requires


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_boundary_none_previous_results_raises(bsmod, small_stack):
    """Test that None previous_results raises RuntimeError."""
    with pytest.raises(RuntimeError, match="requires previous results"):
        bsmod.run(small_stack, previous_results=None)


def test_boundary_include_bbox_false_raises(bsmod, small_stack):
    """Test that include_bbox=False in previous results raises RuntimeError."""
    ext_out = {"summary": {"include_bbox": False}, "per_track": []}
    with pytest.raises(RuntimeError, match="include_bbox=True"):
        bsmod.run(small_stack, {"particle_region_extraction": ext_out})


def test_boundary_unsupported_measure_raises(bsmod, small_stack):
    """Test that an unsupported measure argument raises ValueError."""
    ext_out = _make_extraction_out()
    with pytest.raises(ValueError, match="bbox_max_dim"):
        bsmod.run(
            small_stack, {"particle_region_extraction": ext_out}, measure="perimeter"
        )


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


def test_boundary_empty_per_track(bsmod, small_stack):
    """Test an empty per_track list produces empty flat_table and summary counts."""
    ext_out = {"summary": {"include_bbox": True}, "per_track": []}
    out = bsmod.run(small_stack, {"particle_region_extraction": ext_out})
    assert out["per_track"] == []
    assert out["flat_table"] == []
    assert out["summary"]["n_tracks"] == 0
    assert out["summary"]["n_rows"] == 0


# ---------------------------------------------------------------------------
# max_dim computation
# ---------------------------------------------------------------------------


def test_boundary_square_particle_max_dim(bsmod, small_stack):
    """Test that a square particle's max_dim is equal to its height (or width)."""
    # bbox (1,1,3,3) → height=2, width=2 → max_dim=2.0
    ext_out = _make_extraction_out(bbox_tight=(1, 1, 3, 3))
    out = bsmod.run(small_stack, {"particle_region_extraction": ext_out})
    assert out["per_track"][0]["max_dim"] == pytest.approx([2.0, 2.0])


def test_boundary_non_square_particle_uses_larger_dim(bsmod, small_stack):
    """Test that a non-square particle's max_dim is the larger of height or width."""
    # bbox (0,0,3,5) → height=3, width=5 → max_dim=5.0
    ext_out = _make_extraction_out(bbox_tight=(0, 0, 3, 5))
    out = bsmod.run(small_stack, {"particle_region_extraction": ext_out})
    assert out["per_track"][0]["max_dim"] == pytest.approx([5.0, 5.0])


def test_boundary_tall_particle_uses_height(bsmod, small_stack):
    """Test that a tall particle's max_dim is the height of the bbox."""
    # bbox (0,0,6,2) → height=6, width=2 → max_dim=6.0
    ext_out = _make_extraction_out(bbox_tight=(0, 0, 6, 2))
    out = bsmod.run(small_stack, {"particle_region_extraction": ext_out})
    assert out["per_track"][0]["max_dim"] == pytest.approx([6.0, 6.0])


def test_boundary_none_region_produces_nan_max_dim(bsmod, small_stack):
    """Test that max_dim is NaN when the region is None (missing)."""
    ext_out = _make_extraction_out(include_none=True)  # frame 0 is None
    out = bsmod.run(small_stack, {"particle_region_extraction": ext_out})
    max_dims = out["per_track"][0]["max_dim"]
    assert np.isnan(max_dims[0])
    assert not np.isnan(max_dims[1])


def test_boundary_none_region_increments_missing(bsmod, small_stack):
    """Test that summary counts missing region measurements when a region is None."""
    ext_out = _make_extraction_out(include_none=True)
    out = bsmod.run(small_stack, {"particle_region_extraction": ext_out})
    assert out["summary"]["n_missing_region_measurements"] == 1


# ---------------------------------------------------------------------------
# Threshold and state classification
# ---------------------------------------------------------------------------


def test_boundary_no_threshold_no_state_key(bsmod, small_stack):
    """Test that state is not included when threshold=None."""
    ext_out = _make_extraction_out()
    out = bsmod.run(
        small_stack, {"particle_region_extraction": ext_out}, threshold=None
    )
    assert "state" not in out["per_track"][0]
    assert "state" not in out["flat_table"][0]


def test_boundary_threshold_extended_state(bsmod, small_stack):
    """Test that max_dim > threshold produces state=1."""
    # max_dim=2.0 > threshold=1.5 → state=1 (extended)
    ext_out = _make_extraction_out(bbox_tight=(1, 1, 3, 3))
    out = bsmod.run(small_stack, {"particle_region_extraction": ext_out}, threshold=1.5)
    assert out["per_track"][0]["state"] == [1, 1]


def test_boundary_threshold_compact_state(bsmod, small_stack):
    """Test that max_dim < threshold produces state=0."""
    # max_dim=2.0 < threshold=3.0 → state=0 (compact)
    ext_out = _make_extraction_out(bbox_tight=(1, 1, 3, 3))
    out = bsmod.run(small_stack, {"particle_region_extraction": ext_out}, threshold=3.0)
    assert out["per_track"][0]["state"] == [0, 0]


def test_boundary_threshold_nan_region_state_is_nan(bsmod, small_stack):
    """Test that boundary state is NaN when the region is None (missing)."""
    ext_out = _make_extraction_out(include_none=True)
    out = bsmod.run(small_stack, {"particle_region_extraction": ext_out}, threshold=1.5)
    states = out["per_track"][0]["state"]
    assert np.isnan(states[0])
    assert states[1] == 1


def test_boundary_threshold_state_in_flat_table(bsmod, small_stack):
    """Test that flat_table contains state column when threshold is set."""
    ext_out = _make_extraction_out()
    out = bsmod.run(small_stack, {"particle_region_extraction": ext_out}, threshold=1.5)
    assert "state" in out["flat_table"][0]


# ---------------------------------------------------------------------------
# flat_table and summary
# ---------------------------------------------------------------------------


def test_boundary_flat_table_required_columns(bsmod, small_stack):
    """Test that flat_table contains required columns."""
    ext_out = _make_extraction_out()
    out = bsmod.run(small_stack, {"particle_region_extraction": ext_out})
    row = out["flat_table"][0]
    for col in ("track_id", "label", "frame", "timestamp", "max_dim"):
        assert col in row, f"Missing column: {col}"


def test_boundary_flat_table_max_dim_values(bsmod, small_stack):
    """Test that flat_table contains correct max_dim values."""
    ext_out = _make_extraction_out(bbox_tight=(1, 1, 3, 3))
    out = bsmod.run(small_stack, {"particle_region_extraction": ext_out})
    assert out["flat_table"][0]["max_dim"] == pytest.approx(2.0)


def test_boundary_flat_table_row_count(bsmod, small_stack):
    """Test that flat_table has one row per frame in the track."""
    ext_out = _make_extraction_out(n_frames=3)
    out = bsmod.run(small_stack, {"particle_region_extraction": ext_out})
    assert len(out["flat_table"]) == 3


def test_boundary_summary_keys(bsmod, small_stack):
    """Test that boundary summary contains expected keys."""
    ext_out = _make_extraction_out()
    out = bsmod.run(small_stack, {"particle_region_extraction": ext_out})
    for key in (
        "n_tracks",
        "n_rows",
        "n_missing_region_measurements",
        "state_included",
    ):
        assert key in out["summary"], f"Missing summary key: {key}"


def test_boundary_summary_state_included_false(bsmod, small_stack):
    """Test boundary summary reports state_included=False when threshold is None."""
    ext_out = _make_extraction_out()
    out = bsmod.run(small_stack, {"particle_region_extraction": ext_out})
    assert not out["summary"]["state_included"]


def test_boundary_summary_state_included_true(bsmod, small_stack):
    """Test that boundary summary reports state_included=True when threshold is set."""
    ext_out = _make_extraction_out()
    out = bsmod.run(small_stack, {"particle_region_extraction": ext_out}, threshold=1.5)
    assert out["summary"]["state_included"]


def test_boundary_summary_counts(bsmod, small_stack):
    """Test that boundary summary counds tracks and rows."""
    ext_out = _make_extraction_out(n_frames=2)
    out = bsmod.run(small_stack, {"particle_region_extraction": ext_out})
    assert out["summary"]["n_tracks"] == 1
    assert out["summary"]["n_rows"] == 2


# ==============================================================================
# Integration tests — ParticleRegionExtractionModule → BoundarySizeModule
# ==============================================================================


def test_extraction_output_structure_for_boundary(exmod, bsmod, small_stack, prev):
    """Extraction output directly feeds BoundarySizeModule without transformation."""
    ext_out = exmod.run(small_stack, prev)
    bound_out = bsmod.run(small_stack, {"particle_region_extraction": ext_out})

    assert bound_out["per_track"][0]["max_dim"] == pytest.approx([2.0, 2.0])
    assert bound_out["summary"]["n_rows"] == 2
    assert bound_out["summary"]["n_tracks"] == 1


def test_integration_extraction_boundary_with_threshold(
    exmod, bsmod, small_stack, prev
):
    """End-to-end: extraction → boundary with threshold classifies correctly."""
    ext_out = exmod.run(small_stack, prev)
    # max_dim=2.0; threshold=1.5 → state=1; threshold=3.0 → state=0
    out_extended = bsmod.run(
        small_stack, {"particle_region_extraction": ext_out}, threshold=1.5
    )
    out_compact = bsmod.run(
        small_stack, {"particle_region_extraction": ext_out}, threshold=3.0
    )

    assert all(s == 1 for s in out_extended["per_track"][0]["state"])
    assert all(s == 0 for s in out_compact["per_track"][0]["state"])


def test_integration_extraction_boundary_with_padding(exmod, bsmod, small_stack, prev):
    """
    Padding expands the crop window but not bbox_tight.

    BoundarySizeModule reads bbox_tight, so max_dim reflects the particle's
    actual extent (2.0) regardless of the padding used during extraction.
    """
    ext_out = exmod.run(small_stack, prev, padding=1)
    # padded bbox is (0,0,4,4) but bbox_tight is still (1,1,3,3) → max_dim=2.0
    bound_out = bsmod.run(small_stack, {"particle_region_extraction": ext_out})
    assert bound_out["per_track"][0]["max_dim"] == pytest.approx([2.0, 2.0])


def test_integration_extraction_boundary_with_missing_detection(
    exmod, bsmod, small_stack, detection_out
):
    """Missing detection in extraction produces NaN max_dim in boundary not an error."""
    track_with_gap = {
        "tracks": [
            {
                "id": 0,
                "frames": [0, 1],
                "point_indices": [None, 0],
                "coords": [None, (1.5, 1.5)],
            }
        ]
    }
    ext_out = exmod.run(
        small_stack,
        {"feature_detection": detection_out, "particle_tracking": track_with_gap},
    )
    bound_out = bsmod.run(small_stack, {"particle_region_extraction": ext_out})

    max_dims = bound_out["per_track"][0]["max_dim"]
    assert np.isnan(max_dims[0])
    assert max_dims[1] == pytest.approx(2.0)
    assert bound_out["summary"]["n_missing_region_measurements"] == 1


# ==============================================================================
# ParticleRegionMediaExportModule
# ==============================================================================

# ---------------------------------------------------------------------------
# Shared stubs and builders
# ---------------------------------------------------------------------------

_MODULE = "playnano.analysis.modules.particle_region_media_export"


class _ExportStack:
    """Provide a minimal stack stub for export module tests."""

    pixel_size_nm = 1.0
    file_path = "my_stack.nhf"


def _region(frame_idx: int, crop_shape=(2, 2)) -> dict:
    """Build a region dict with a float32 image_crop."""
    return {
        "frame": frame_idx,
        "timestamp": float(frame_idx) * 0.5,
        "label": 1,
        "image_crop": np.full(crop_shape, float(frame_idx + 1), dtype=np.float32),
    }


def _make_track(
    track_id: int,
    n_frames: int = 2,
    crop_shape=(2, 2),
    with_none: bool = False,
    no_crops: bool = False,
) -> dict:
    """Build a per_track entry as produced by ParticleRegionExtractionModule."""
    regions = []
    for i in range(n_frames):
        if no_crops:
            regions.append({"frame": i, "timestamp": float(i) * 0.5, "label": 1})
        elif with_none and i == 0:
            regions.append(None)
        else:
            regions.append(_region(i, crop_shape=crop_shape))
    return {
        "track_id": track_id,
        "frames": list(range(n_frames)),
        "timestamps": [float(i) * 0.5 for i in range(n_frames)],
        "regions": regions,
    }


def _extraction_out(
    n_tracks: int = 1,
    n_frames: int = 2,
    include_image: bool = True,
    **track_kwargs,
) -> dict:
    """Build a minimal particle_region_extraction output for export module tests."""
    return {
        "summary": {"include_image": include_image},
        "per_track": [
            _make_track(i, n_frames=n_frames, **track_kwargs) for i in range(n_tracks)
        ],
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def parexmod():
    """Return a bare ParticleRegionMediaExportModule instance."""
    return ParticleRegionMediaExportModule()


@pytest.fixture
def stack():
    """Return a minimal stack stub with pixel_size_nm and file_path."""
    return _ExportStack()


@pytest.fixture
def mock_exports(monkeypatch, tmp_path):
    """Patch all file-writing functions and path helpers in the export module.

    Returns a dict with 'tmp_path' and MagicMocks for 'gif', 'video', 'seq'.
    """
    monkeypatch.setattr(
        f"{_MODULE}.prepare_output_directory", lambda *a, **kw: tmp_path
    )
    monkeypatch.setattr(f"{_MODULE}.sanitize_output_name", lambda *a, **kw: "test")

    gif_mock = MagicMock()
    video_mock = MagicMock()
    seq_mock = MagicMock()
    monkeypatch.setattr(f"{_MODULE}.create_gif_with_scale_and_timestamp", gif_mock)
    monkeypatch.setattr(f"{_MODULE}.create_video_with_scale_and_timestamp", video_mock)
    monkeypatch.setattr(f"{_MODULE}.create_image_sequence", seq_mock)

    return {"tmp_path": tmp_path, "gif": gif_mock, "video": video_mock, "seq": seq_mock}


# ---------------------------------------------------------------------------
# _pad_crop_to_size
# ---------------------------------------------------------------------------


class TestPadCropToSize:
    """Test the _pad_crop_to_size centre-padding helper."""

    def test_already_target_size_unchanged(self):
        """Crop matching target dimensions is returned without modification."""
        crop = np.ones((3, 3))
        result = _pad_crop_to_size(crop, 3, 3)
        np.testing.assert_array_equal(result, crop)

    def test_output_shape_correct(self):
        """Output array has exactly the requested (target_h, target_w) shape."""
        crop = np.ones((2, 2))
        result = _pad_crop_to_size(crop, 4, 4)
        assert result.shape == (4, 4)

    def test_original_data_centred(self):
        """The original pixel values appear in the centre of the padded array."""
        crop = np.ones((2, 2)) * 5.0
        result = _pad_crop_to_size(crop, 4, 4)
        np.testing.assert_array_equal(result[1:3, 1:3], 5.0)
        assert result[0, 0] == 0.0

    def test_fill_value_used_for_padding(self):
        """Padding pixels take the specified fill_value, not zero."""
        crop = np.ones((2, 2))
        result = _pad_crop_to_size(crop, 4, 4, fill_value=-1.0)
        assert result[0, 0] == pytest.approx(-1.0)

    def test_odd_size_difference_distributes_correctly(self):
        """(1,3)→(3,3): top pad=1, bottom pad=1 via floor-then-remainder split."""
        crop = np.ones((1, 3)) * 9.0
        result = _pad_crop_to_size(crop, 3, 3)
        assert result.shape == (3, 3)
        assert result[1, 0] == pytest.approx(9.0)
        assert result[0, 0] == pytest.approx(0.0)

    def test_3d_array_channel_dim_preserved(self):
        """A (H, W, C) crop is padded in the spatial dims only."""
        crop = np.ones((2, 2, 3))
        result = _pad_crop_to_size(crop, 4, 4)
        assert result.shape == (4, 4, 3)

    def test_asymmetric_target(self):
        """Different target height and width are both applied correctly."""
        crop = np.ones((1, 1))
        result = _pad_crop_to_size(crop, 3, 5)
        assert result.shape == (3, 5)

    def test_output_dtype_matches_input(self):
        """Output array dtype matches the input crop dtype."""
        crop = np.ones((2, 2), dtype=np.float64)
        result = _pad_crop_to_size(crop, 4, 4)
        assert result.dtype == np.float64


# ---------------------------------------------------------------------------
# _assemble_track_stack
# ---------------------------------------------------------------------------


class TestAssembleTrackStack:
    """Test the _assemble_track_stack track assembly helper."""

    def test_all_none_regions_returns_none(self):
        """All-None region list yields None (no valid crops to stack)."""
        regions = [None, None, None]
        assert _assemble_track_stack(regions, fill_value=0.0) is None

    def test_regions_with_no_image_crop_returns_none(self):
        """Region dicts without 'image_crop' key are treated as absent."""
        regions = [{"frame": 0, "label": 1}]
        assert _assemble_track_stack(regions, fill_value=0.0) is None

    def test_single_valid_crop_shape(self):
        """A single crop produces a (1, H, W) output array."""
        regions = [_region(0, crop_shape=(3, 4))]
        result = _assemble_track_stack(regions, fill_value=0.0)
        assert result is not None
        assert result.shape == (1, 3, 4)

    def test_multiple_same_size_crops_stacked(self):
        """Multiple same-size crops are stacked into a (T, H, W) array."""
        regions = [_region(i) for i in range(3)]
        result = _assemble_track_stack(regions, fill_value=0.0)
        assert result.shape == (3, 2, 2)

    def test_frame_values_in_correct_temporal_order(self):
        """Frame i has pixel value (i+1); verify the stack preserves order."""
        regions = [_region(i) for i in range(3)]
        result = _assemble_track_stack(regions, fill_value=0.0)
        for i in range(3):
            assert result[i, 0, 0] == pytest.approx(float(i + 1))

    def test_none_region_filled_with_fill_value(self):
        """A None region produces a fill_value plane in the output stack."""
        regions = [None, _region(1)]
        result = _assemble_track_stack(regions, fill_value=-99.0)
        assert result is not None
        assert result[0, 0, 0] == pytest.approx(-99.0)

    def test_mixed_sizes_padded_to_largest(self):
        """Smaller crops are centre-padded to the largest crop in the track."""
        regions = [
            _region(0, crop_shape=(2, 2)),
            _region(1, crop_shape=(4, 4)),
        ]
        result = _assemble_track_stack(regions, fill_value=0.0)
        assert result.shape == (2, 4, 4)
        np.testing.assert_array_equal(result[0, 1:3, 1:3], 1.0)
        assert result[0, 0, 0] == 0.0

    def test_output_dtype_is_float32(self):
        """Assembled stack is always float32 regardless of input dtype."""
        regions = [_region(0)]
        result = _assemble_track_stack(regions, fill_value=0.0)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# _filter_and_sort_tracks
# ---------------------------------------------------------------------------


class TestFilterAndSortTracks:
    """Test the _filter_and_sort_tracks filter, sort, and cap helper."""

    def _tracks(self, ids, n_regions_each=None, n_none_each=None):
        """Build minimal track dicts suitable for filter/sort tests."""
        tracks = []
        for i, tid in enumerate(ids):
            n = n_regions_each[i] if n_regions_each else 3
            n_none = n_none_each[i] if n_none_each else 0
            regions = [None] * n_none + [{"frame": j} for j in range(n - n_none)]
            tracks.append({"track_id": tid, "regions": regions})
        return tracks

    def test_invalid_sort_by_raises(self):
        """Unrecognised sort_by value raises ValueError."""
        with pytest.raises(ValueError, match="sort_by"):
            _filter_and_sort_tracks(
                [], track_ids=None, sort_by="bad_key", max_tracks=None
            )

    def test_empty_input_returns_empty(self):
        """Empty track list passes through all stages and returns empty list."""
        assert _filter_and_sort_tracks([], None, "track_id", None) == []

    def test_no_filter_returns_all(self):
        """track_ids=None and max_tracks=None preserve all input tracks."""
        tracks = self._tracks([2, 0, 1])
        result = _filter_and_sort_tracks(
            tracks, track_ids=None, sort_by="track_id", max_tracks=None
        )
        assert len(result) == 3

    def test_track_ids_whitelist(self):
        """Only tracks whose IDs appear in track_ids are retained."""
        tracks = self._tracks([0, 1, 2, 3])
        result = _filter_and_sort_tracks(
            tracks, track_ids=[1, 3], sort_by="track_id", max_tracks=None
        )
        assert [t["track_id"] for t in result] == [1, 3]

    def test_unknown_track_ids_silently_ignored(self):
        """Test IDs in track_ids that don't exist in tracks are silently skipped."""
        tracks = self._tracks([0, 1])
        result = _filter_and_sort_tracks(
            tracks, track_ids=[1, 99], sort_by="track_id", max_tracks=None
        )
        assert len(result) == 1
        assert result[0]["track_id"] == 1

    def test_sort_by_track_id_ascending(self):
        """sort_by='track_id' returns tracks in ascending ID order."""
        tracks = self._tracks([3, 1, 2])
        result = _filter_and_sort_tracks(
            tracks, track_ids=None, sort_by="track_id", max_tracks=None
        )
        assert [t["track_id"] for t in result] == [1, 2, 3]

    def test_sort_by_n_frames_descending(self):
        """sort_by='n_frames' returns longest tracks first."""
        tracks = self._tracks([0, 1, 2], n_regions_each=[1, 3, 2])
        result = _filter_and_sort_tracks(
            tracks, track_ids=None, sort_by="n_frames", max_tracks=None
        )
        assert [t["track_id"] for t in result] == [1, 2, 0]

    def test_sort_by_n_detections_descending(self):
        """sort_by='n_detections' ranks by non-None region count, descending."""
        tracks = self._tracks(
            [0, 1, 2], n_regions_each=[3, 3, 3], n_none_each=[1, 0, 2]
        )
        result = _filter_and_sort_tracks(
            tracks, track_ids=None, sort_by="n_detections", max_tracks=None
        )
        assert [t["track_id"] for t in result] == [1, 0, 2]

    def test_max_tracks_caps_output(self):
        """max_tracks limits the output to at most N tracks."""
        tracks = self._tracks([0, 1, 2, 3])
        result = _filter_and_sort_tracks(
            tracks, track_ids=None, sort_by="track_id", max_tracks=2
        )
        assert len(result) == 2

    def test_max_tracks_applied_after_sort(self):
        """max_tracks takes the first N after sorting → the N with most frames."""
        tracks = self._tracks([0, 1, 2], n_regions_each=[1, 5, 3])
        result = _filter_and_sort_tracks(
            tracks, track_ids=None, sort_by="n_frames", max_tracks=2
        )
        assert [t["track_id"] for t in result] == [1, 2]

    def test_filter_then_sort_then_cap(self):
        """All three stages run in order: whitelist → sort → cap."""
        tracks = self._tracks([0, 1, 2, 3], n_regions_each=[1, 4, 2, 3])
        result = _filter_and_sort_tracks(
            tracks, track_ids=[1, 2, 3], sort_by="n_frames", max_tracks=2
        )
        assert [t["track_id"] for t in result] == [1, 3]

    def test_max_tracks_larger_than_available(self):
        """max_tracks exceeding available track count returns all tracks."""
        tracks = self._tracks([0, 1])
        result = _filter_and_sort_tracks(
            tracks, track_ids=None, sort_by="track_id", max_tracks=100
        )
        assert len(result) == 2


# ---------------------------------------------------------------------------
# ParticleRegionMediaExportModule — properties and errors
# ---------------------------------------------------------------------------


def test_export_module_name(parexmod):
    """Module name is 'particle_region_media_export'."""
    assert parexmod.name == "particle_region_media_export"


def test_export_module_version(parexmod):
    """Module version string is present and non-empty."""
    assert parexmod.version == "0.1.0"


def test_export_module_requires(parexmod):
    """Module declares particle_region_extraction as a dependency."""
    assert "particle_region_extraction" in parexmod.requires


def test_export_none_previous_results_raises(parexmod, stack):
    """None previous_results raises RuntimeError before any processing."""
    with pytest.raises(RuntimeError, match="requires previous results"):
        parexmod.run(stack, previous_results=None)


def test_export_all_flags_false_raises(parexmod, stack):
    """At least one export flag must be True or a RuntimeError is raised."""
    ext_out = _extraction_out()
    with pytest.raises(RuntimeError, match="at least one"):
        parexmod.run(
            stack,
            {"particle_region_extraction": ext_out},
            export_gif=False,
            export_video=False,
            export_sequence=False,
        )


def test_export_missing_extraction_module_raises(parexmod, stack):
    """Missing extraction module key in previous_results raises RuntimeError."""
    with pytest.raises(RuntimeError, match="extraction_module"):
        parexmod.run(stack, previous_results={}, export_video=True)


def test_export_include_image_false_raises(parexmod, stack):
    """Upstream extraction run without include_image=True raises RuntimeError."""
    ext_out = _extraction_out(include_image=False)
    with pytest.raises(RuntimeError, match="include_image=True"):
        parexmod.run(stack, {"particle_region_extraction": ext_out})


def test_export_invalid_sort_by_raises(parexmod, stack, mock_exports):
    """Unrecognised sort_by value propagates as ValueError."""
    ext_out = _extraction_out()
    with pytest.raises(ValueError, match="sort_by"):
        parexmod.run(
            stack,
            {"particle_region_extraction": ext_out},
            sort_by="invalid_key",
            export_video=True,
        )


# ---------------------------------------------------------------------------
# run() — export function routing
# ---------------------------------------------------------------------------


def test_export_gif_calls_gif_function(parexmod, stack, mock_exports):
    """export_gif=True calls create_gif_with_scale_and_timestamp exactly once."""
    ext_out = _extraction_out()
    parexmod.run(
        stack,
        {"particle_region_extraction": ext_out},
        export_gif=True,
        export_video=False,
    )
    assert mock_exports["gif"].call_count == 1
    assert mock_exports["video"].call_count == 0
    assert mock_exports["seq"].call_count == 0


def test_export_video_calls_video_function(parexmod, stack, mock_exports):
    """export_video=True calls create_video_with_scale_and_timestamp exactly once."""
    ext_out = _extraction_out()
    parexmod.run(stack, {"particle_region_extraction": ext_out}, export_video=True)
    assert mock_exports["video"].call_count == 1
    assert mock_exports["gif"].call_count == 0


def test_export_sequence_calls_sequence_function(parexmod, stack, mock_exports):
    """export_sequence=True calls create_image_sequence exactly once."""
    ext_out = _extraction_out()
    parexmod.run(
        stack,
        {"particle_region_extraction": ext_out},
        export_video=False,
        export_sequence=True,
    )
    assert mock_exports["seq"].call_count == 1


def test_all_three_exports_all_called(parexmod, stack, mock_exports):
    """All three export functions are each called once when all flags are True."""
    ext_out = _extraction_out()
    parexmod.run(
        stack,
        {"particle_region_extraction": ext_out},
        export_gif=True,
        export_video=True,
        export_sequence=True,
    )
    assert mock_exports["gif"].call_count == 1
    assert mock_exports["video"].call_count == 1
    assert mock_exports["seq"].call_count == 1


def test_multiple_tracks_each_gets_own_export_call(parexmod, stack, mock_exports):
    """Each track produces its own export call (one call per track)."""
    ext_out = _extraction_out(n_tracks=3)
    parexmod.run(stack, {"particle_region_extraction": ext_out}, export_video=True)
    assert mock_exports["video"].call_count == 3


def test_video_export_path_contains_track_id(parexmod, stack, mock_exports):
    """Output path passed to video export includes the track ID stem."""
    ext_out = _extraction_out(n_tracks=1)
    parexmod.run(stack, {"particle_region_extraction": ext_out}, export_video=True)
    call_kwargs = mock_exports["video"].call_args
    output_path = call_kwargs[1].get("output_path") or call_kwargs[0][3]
    assert "track_0" in str(output_path)


def test_gif_export_path_contains_track_id(parexmod, stack, mock_exports):
    """Output path passed to GIF export includes the track ID stem."""
    ext_out = _extraction_out(n_tracks=1)
    parexmod.run(
        stack,
        {"particle_region_extraction": ext_out},
        export_gif=True,
        export_video=False,
    )
    call_kwargs = mock_exports["gif"].call_args
    output_path = call_kwargs[1].get("output_path") or call_kwargs[0][3]
    assert "track_0" in str(output_path)


def test_sequence_folder_contains_track_id(parexmod, stack, mock_exports):
    """Output folder passed to image-sequence export includes the track ID stem."""
    ext_out = _extraction_out(n_tracks=1)
    parexmod.run(
        stack,
        {"particle_region_extraction": ext_out},
        export_video=False,
        export_sequence=True,
    )
    call_kwargs = mock_exports["seq"].call_args
    folder = call_kwargs[1].get("output_folder") or call_kwargs[0][3]
    assert "track_0" in str(folder)


# ---------------------------------------------------------------------------
# run() — skipped tracks
# ---------------------------------------------------------------------------


def test_track_with_no_image_crops_skipped(parexmod, stack, mock_exports):
    """A track with no image_crop on any region is skipped, not exported."""
    ext_out = _extraction_out(no_crops=True)
    out = parexmod.run(
        stack, {"particle_region_extraction": ext_out}, export_video=True
    )
    assert mock_exports["video"].call_count == 0
    assert out["summary"]["n_tracks_skipped"] == 1


def test_track_with_no_image_crops_not_in_manifest(parexmod, stack, mock_exports):
    """A skipped track does not appear in per_track manifest or video_paths."""
    ext_out = _extraction_out(no_crops=True)
    out = parexmod.run(
        stack, {"particle_region_extraction": ext_out}, export_video=True
    )
    assert out["per_track"] == []
    assert out["video_paths"] == []


def test_none_regions_do_not_skip_track(parexmod, stack, mock_exports):
    """A track with some None regions but at least one valid crop is NOT skipped."""
    ext_out = _extraction_out(with_none=True)
    out = parexmod.run(
        stack, {"particle_region_extraction": ext_out}, export_video=True
    )
    assert mock_exports["video"].call_count == 1
    assert out["summary"]["n_tracks_skipped"] == 0


# ---------------------------------------------------------------------------
# run() — track filtering
# ---------------------------------------------------------------------------


def test_track_ids_filters_exports(parexmod, stack, mock_exports):
    """Only the track IDs in track_ids are exported; others are skipped."""
    ext_out = _extraction_out(n_tracks=3)
    out = parexmod.run(
        stack, {"particle_region_extraction": ext_out}, export_video=True, track_ids=[1]
    )
    assert mock_exports["video"].call_count == 1
    assert out["per_track"][0]["track_id"] == 1


def test_max_tracks_limits_exports(parexmod, stack, mock_exports):
    """max_tracks caps the number of exported tracks after sorting."""
    ext_out = _extraction_out(n_tracks=5)
    out = parexmod.run(
        stack, {"particle_region_extraction": ext_out}, export_video=True, max_tracks=2
    )
    assert mock_exports["video"].call_count == 2
    assert len(out["per_track"]) == 2


def test_sort_by_n_frames_exports_longest_first(parexmod, stack, mock_exports):
    """With sort_by='n_frames' + max_tracks=1, the longest track is exported."""
    tracks = [
        _make_track(0, n_frames=1),
        _make_track(1, n_frames=4),
        _make_track(2, n_frames=2),
    ]
    ext_out = {"summary": {"include_image": True}, "per_track": tracks}
    out = parexmod.run(
        stack,
        {"particle_region_extraction": ext_out},
        export_video=True,
        sort_by="n_frames",
        max_tracks=1,
    )
    assert out["per_track"][0]["track_id"] == 1


# ---------------------------------------------------------------------------
# run() — manifest and summary
# ---------------------------------------------------------------------------


def test_manifest_gif_path_present(parexmod, stack, mock_exports):
    """Per-track manifest contains 'gif_path' and gif_paths list is populated."""
    ext_out = _extraction_out()
    out = parexmod.run(
        stack,
        {"particle_region_extraction": ext_out},
        export_gif=True,
        export_video=False,
    )
    assert "gif_path" in out["per_track"][0]
    assert len(out["gif_paths"]) == 1


def test_manifest_video_path_present(parexmod, stack, mock_exports):
    """Per-track manifest contains 'video_path' and video_paths list is populated."""
    ext_out = _extraction_out()
    out = parexmod.run(
        stack, {"particle_region_extraction": ext_out}, export_video=True
    )
    assert "video_path" in out["per_track"][0]
    assert len(out["video_paths"]) == 1


def test_manifest_sequence_folder_present(parexmod, stack, mock_exports):
    """Per-track manifest contains 'sequence_folder' and folders list is populated."""
    ext_out = _extraction_out()
    out = parexmod.run(
        stack,
        {"particle_region_extraction": ext_out},
        export_video=False,
        export_sequence=True,
    )
    assert "sequence_folder" in out["per_track"][0]
    assert len(out["sequence_folders"]) == 1


def test_manifest_keys_absent_when_not_exported(parexmod, stack, mock_exports):
    """Keys for disabled export types must not appear in the per_track entry."""
    ext_out = _extraction_out()
    out = parexmod.run(
        stack, {"particle_region_extraction": ext_out}, export_video=True
    )
    record = out["per_track"][0]
    assert "gif_path" not in record
    assert "sequence_folder" not in record


def test_manifest_frame_size(parexmod, stack, mock_exports):
    """Per-track manifest records the (height, width) of the assembled crop."""
    ext_out = _extraction_out()
    out = parexmod.run(
        stack, {"particle_region_extraction": ext_out}, export_video=True
    )
    assert out["per_track"][0]["frame_size"] == (2, 2)


def test_summary_keys_present(parexmod, stack, mock_exports):
    """Summary dict contains all expected bookkeeping and configuration keys."""
    ext_out = _extraction_out()
    out = parexmod.run(
        stack, {"particle_region_extraction": ext_out}, export_video=True
    )
    for key in (
        "n_tracks_total",
        "n_tracks_selected",
        "n_tracks_written",
        "n_tracks_skipped",
        "n_tracks_filtered_out",
        "track_ids",
        "max_tracks",
        "sort_by",
        "export_gif",
        "export_video",
        "export_sequence",
    ):
        assert key in out["summary"], f"Missing summary key: {key}"


def test_summary_counts_all_exported(parexmod, stack, mock_exports):
    """All track counters are correct when every track is exported successfully."""
    ext_out = _extraction_out(n_tracks=3)
    out = parexmod.run(
        stack, {"particle_region_extraction": ext_out}, export_video=True
    )
    s = out["summary"]
    assert s["n_tracks_total"] == 3
    assert s["n_tracks_selected"] == 3
    assert s["n_tracks_written"] == 3
    assert s["n_tracks_skipped"] == 0
    assert s["n_tracks_filtered_out"] == 0


def test_summary_filtered_out_count(parexmod, stack, mock_exports):
    """n_tracks_filtered_out reflects tracks excluded by max_tracks."""
    ext_out = _extraction_out(n_tracks=4)
    out = parexmod.run(
        stack, {"particle_region_extraction": ext_out}, export_video=True, max_tracks=2
    )
    assert out["summary"]["n_tracks_filtered_out"] == 2
    assert out["summary"]["n_tracks_selected"] == 2


def test_summary_export_flags_echoed(parexmod, stack, mock_exports):
    """Summary echoes the export_gif, export_video, export_sequence flags."""
    ext_out = _extraction_out()
    out = parexmod.run(
        stack,
        {"particle_region_extraction": ext_out},
        export_gif=True,
        export_video=False,
        export_sequence=True,
    )
    assert out["summary"]["export_gif"] is True
    assert out["summary"]["export_video"] is False
    assert out["summary"]["export_sequence"] is True


def test_empty_extraction_output_returns_empty_manifest(parexmod, stack, mock_exports):
    """Empty per_track input produces empty manifest with n_tracks_total=0."""
    ext_out = {"summary": {"include_image": True}, "per_track": []}
    out = parexmod.run(
        stack, {"particle_region_extraction": ext_out}, export_video=True
    )
    assert out["per_track"] == []
    assert out["video_paths"] == []
    assert out["summary"]["n_tracks_total"] == 0
