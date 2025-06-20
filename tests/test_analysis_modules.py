"""Tests for built in analysis modules."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from playNano.afm_stack import AFMImageStack
from playNano.analysis.modules.feature_detection import FeatureDetectionModule

# --- Tests for feature_detection ---


@pytest.fixture
def stack_1frame_with_timestamps():
    """
    AFMImageStack with 1 frame of 3x3 data and an explicit timestamp.
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
    AFMImageStack with 2 frames of 3×3 data, but missing timestamps in metadata.
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
    """
    Mask only the center pixel of a 3x3 frame.
    """
    H, W = frame.shape
    mask = np.zeros((H, W), dtype=bool)
    # center at index (1,1)
    mask[1, 1] = True
    return mask


def full_mask(frame: np.ndarray, **kwargs) -> np.ndarray:
    """
    Mask all pixels True.
    """
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


def test_2frames_default_timestamps(stack_2frames_no_timestamps):
    """Test that or stacks without timestamps, module should default frame_timestamp to frame index."""  # noqa
    module = FeatureDetectionModule()
    stack = stack_2frames_no_timestamps
    # Use full_mask and remove_edge=False so each frame yields one feature
    out = module.run(stack, mask_fn=full_mask, min_size=1, remove_edge=False)
    fpf = out["features_per_frame"]
    # Two frames, each with one feature
    assert len(fpf) == 2
    # frame_timestamp should be float(i) for i=0,1
    assert fpf[0][0]["frame_timestamp"] == pytest.approx(0.0)
    assert fpf[1][0]["frame_timestamp"] == pytest.approx(1.0)


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
