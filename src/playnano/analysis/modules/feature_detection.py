"""
Threshold-based feature detection for AFM image stacks.

This is a playNano analysis module for analyzing processed high- speed AFM
data. This implements a concrete subclass of the abstract `AnalysisModule`
base class.

This module detects contiguous features in each frame of an AFM image stack
using user-provided or precomputed masks. It performs optional hole filling,
morphological separation, size and edge filtering, and extracts per-feature
statistics for each frame.

Author: Daniel E. Rollins (d.e.rollins@leed.ac.uk) / Github: derollins
"""

import inspect
from typing import Any, Optional

import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.measure import label, regionprops
from skimage.morphology import disk, opening, remove_small_holes

from playnano.analysis.base import AnalysisModule
from playnano.processing.mask_generators import register_masking
from playnano.utils.param_utils import param_conditions

MASK_MAP = register_masking()


class FeatureDetectionModule(AnalysisModule):
    """
    Detect contiguous features in each frame of an AFM image stack.

    This module accepts either a callable mask-generating function or a reference
    to a boolean mask array stored in `previous_results`. Connected components are
    identified per frame, optionally morphologically separated, filtered by size
    and edge contact, and returned along with feature statistics and labeled masks.

    Parameters
    ----------
    mask_fn : callable, optional
        Function of the form `frame -> bool_2D_array` used to generate a mask
        for each frame. Required if `mask_key` is not provided.

    mask_key : str, optional
        Key referencing a boolean mask array stored in `previous_results`.
        Required if `mask_fn` is not provided.

    morph_opening : bool, default False
        If True, apply morphological opening to each labeled object to separate
        touching features.

    sep_radius : int, default 6
        Radius of the disk structuring element used for morphological opening.

    min_size : int, default 10
        Minimum pixel area for a region to be retained.

    remove_edge : bool, default True
        If True, discard any region touching the frame boundary.

    fill_holes : bool, default False
        If True, fill holes within each mask before labeling.

    hole_area : int or None, optional
        Maximum size of holes to fill. If None, fill all holes.

    **mask_kwargs : Any
        Additional keyword arguments passed to `mask_fn(frame, **mask_kwargs)`.

    Returns
    -------
    dict[str, Any]
        - **features_per_frame** : list[list[dict]]
          Per-frame list of feature stats dicts, each with:

            - `"frame_timestamp"` : float
            - `"label"`           : int
            - `"area"`            : int
            - `"min"`, `"max"`, `"mean"` : float
            - `"bbox"`            : (min_row, min_col, max_row, max_col)
            - `"centroid"`        : (row, col)

        - **labeled_masks** : list[np.ndarray]
              Integer-labeled masks for each frame.
        - **summary** : dict
              Aggregate values: total frames, total features, average features
              per frame.

    Raises
    ------
    ValueError
        If mask configuration is invalid or mask has incorrect shape/dtype.

    KeyError
        If `mask_key` is supplied but missing from `previous_results`.

    Version
    -------
    0.2.0

    Version 0.2.0 adds morphological opening for seperating close or touching particles.

    Examples
    --------
    >>> pipeline.add("feature_detection", mask_fn=mask_mean_offset,
    ...              min_size=20, fill_holes=True, hole_area=50)
    >>> result = pipeline.run(stack)
    >>> result["summary"]["total_features"]
    123
    """

    version = "0.2.0"

    @property
    def name(self) -> str:
        """
        Return the name of the analysis module.

        Returns
        -------
        str
            The identifier for this module: ``"feature_detection"``.
        """
        return "feature_detection"

    def _get_mask_array(
        self,
        data: np.ndarray,
        previous_results: Optional[dict[str, Any]],
        mask_fn: Optional[callable],
        mask_key: Optional[str],
        **mask_kwargs,
    ) -> np.ndarray:
        """
        Resolve the boolean mask array for the stack.

        Either retrieves a precomputed mask from `previous_results` using
        `mask_key` or computes a mask for each frame using `mask_fn`.

        Parameters
        ----------
        data : np.ndarray
            3D array of shape (n_frames, H, W) representing the stack data.

        previous_results : dict[str, Any] or None
            Mapping of previously computed analysis outputs.

        mask_fn : callable or None
            Function that produces a mask for a single frame.

        mask_key : str or None
            Key referencing a boolean array in `previous_results`.

        **mask_kwargs : Any
            Passed directly to `mask_fn`.

        Returns
        -------
        np.ndarray
            Boolean array of shape matching `data`.

        Raises
        ------
        ValueError
            If mask configuration is invalid.

        KeyError
            If `mask_key` is provided but missing in `previous_results`.
        """
        n_frames, H, W = data.shape

        if mask_key is not None:
            if not previous_results or mask_key not in previous_results:
                raise KeyError(f"mask_key '{mask_key}' not found in previous_results")
            mask_arr = previous_results[mask_key]
            if not (
                isinstance(mask_arr, np.ndarray)
                and mask_arr.dtype == bool
                and mask_arr.shape == data.shape
            ):
                raise ValueError(
                    f"previous_results[{mask_key}] must be a boolean ndarray of shape {data.shape}"  # noqa
                )
            return mask_arr

        if mask_fn is None:
            raise ValueError("Either mask_fn or mask_key must be provided")

        # Resolve mask_fn if it's a registered string
        if isinstance(mask_fn, str):
            if mask_fn not in MASK_MAP:
                raise ValueError(
                    f"mask_fn '{mask_fn}' is not a known registered mask. "
                    f"Available: {list(MASK_MAP.keys())}"
                )
            mask_fn = MASK_MAP[mask_fn]

        # Compute mask frame-by-frame
        mask_arr = np.zeros_like(data, dtype=bool)
        for i in range(n_frames):
            try:
                mf = mask_fn(data[i], **mask_kwargs)
            except TypeError:
                mf = mask_fn(data[i])
            if not (
                isinstance(mf, np.ndarray) and mf.dtype == bool and mf.shape == (H, W)
            ):
                raise ValueError(f"mask_fn returned invalid mask for frame {i}")
            mask_arr[i] = mf
        return mask_arr

    def _separate_touching_by_opening(
        self,
        labeled_mask: np.ndarray,
        selem_radius: int = 6,
        max_area_loss: float = 0.6,
    ) -> np.ndarray:
        """
        Apply morphological opening to each labeled object separately, then relabel.

        If opening removes too much of an object (area loss > max_area_loss), keep
        the original object to avoid over-erosion of thin rectangles.

        Parameters
        ----------
        labeled_mask : np.ndarray
            Integer-labeled mask (0 = background)
        selem_radius : int
            Radius for disk structuring element used in opening
        max_area_loss : float
            If opened area < (1 - max_area_loss) * original area, fallback to original

        Returns
        -------
        np.ndarray
            New labeled mask with objects possibly split into multiple labels.
        """
        if selem_radius <= 0:
            return labeled_mask

        selem = disk(selem_radius)
        out = np.zeros_like(labeled_mask, dtype=np.int32)
        next_label = 1

        # Iterate each object independently
        for obj_id in np.unique(labeled_mask):
            if obj_id == 0:
                continue
            binary = labeled_mask == obj_id

            orig_area = int(binary.sum())
            if orig_area == 0:
                continue

            opened = opening(binary, selem)
            opened_area = int(opened.sum())

            # Safety: if opening removed too much, retain original shape
            if opened_area == 0 or opened_area < (1.0 - max_area_loss) * orig_area:
                opened = binary

            relabeled = label(opened)
            if relabeled.max() == 0:
                continue

            for sub_id in range(1, relabeled.max() + 1):
                out[relabeled == sub_id] = next_label
                next_label += 1

        return out

    def _process_frame(
        self,
        frame: np.ndarray,
        mask_frame: np.ndarray,
        frame_ts: float,
        *,
        morph_opening: bool,
        sep_radius: int,
        min_size: int,
        remove_edge: bool,
        fill_holes: bool,
        hole_area: Optional[int],
    ) -> tuple[list[dict[str, Any]], np.ndarray]:
        """Process a single frame: hole fill, labeling, filtering, stats."""
        H, W = frame.shape

        # Optionally fill holes
        if fill_holes:
            if hole_area is not None:
                # Normalize semantics to "fill holes with area < hole_area"
                # across skimage versions.
                # New API (0.26+): prefers `max_size` which fills holes with
                # area <= max_size.
                # Old API: `area_threshold` fills holes with area < area_threshold.
                sig = inspect.signature(remove_small_holes)
                if "max_size" in sig.parameters:
                    # Emulate strict "< hole_area" by using <= hole_area-1
                    max_size = max(hole_area - 1, 0)  # guard against negatives
                    mask_frame = remove_small_holes(
                        mask_frame, max_size=max_size, connectivity=1
                    )
                else:
                    mask_frame = remove_small_holes(
                        mask_frame, area_threshold=hole_area, connectivity=1
                    )
            else:
                mask_frame = binary_fill_holes(mask_frame)
            mask_frame = mask_frame.astype(bool)

        # Label connected regions
        initial_labeled = label(mask_frame)

        # Morphological separation (per object), before filtering
        if morph_opening and sep_radius > 0:
            initial_labeled = self._separate_touching_by_opening(
                initial_labeled, selem_radius=sep_radius
            )

        filtered_mask = np.zeros_like(mask_frame, dtype=bool)

        for prop in regionprops(initial_labeled):
            if prop.area < min_size:
                continue
            minr, minc, maxr, maxc = prop.bbox
            if remove_edge and (minr == 0 or minc == 0 or maxr == H or maxc == W):
                continue
            filtered_mask[initial_labeled == prop.label] = True

        # Relabel after filtering
        labeled = label(filtered_mask)
        props = regionprops(labeled, intensity_image=frame)

        # Collect stats
        features: list[dict[str, Any]] = []
        for prop in props:
            mask_pixels = labeled == prop.label
            vals = frame[mask_pixels]
            if vals.size == 0:
                continue
            features.append(
                {
                    "frame_timestamp": frame_ts,
                    "label": int(prop.label),
                    "area": float(prop.area),
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                    "mean": float(vals.mean()),
                    "bbox": tuple(map(int, prop.bbox)),  # (minr, minc, maxr, maxc)
                    "centroid": tuple(map(float, prop.centroid)),
                }
            )
        return features, labeled

    def _summarize(self, n_frames: int, total_features: int) -> dict[str, Any]:
        """Summarize results across frames."""
        return {
            "total_frames": n_frames,
            "total_features": total_features,
            "avg_features_per_frame": (
                total_features / n_frames if n_frames > 0 else 0.0
            ),
        }

    @param_conditions(
        mask_fn=lambda p: not p.get("mask_key"),
        mask_key=lambda p: not p.get("mask_fn"),
        hole_area=lambda p: p.get("fill_holes", False),
    )
    def run(
        self,
        stack,
        previous_results: Optional[dict[str, Any]] = None,
        *,
        # Mask input: either supply a mask function or refer to
        # existing mask in previous_results
        mask_fn: Optional[callable] = None,
        mask_key: Optional[str] = None,
        # Morphological opening
        morph_opening: bool = False,
        sep_radius: int = 6,
        # Filtering criteria:
        min_size: int = 10,
        remove_edge: bool = True,
        # Hole-filling options:
        fill_holes: bool = False,
        hole_area: Optional[int] = None,
        # kwargs for mask_fn(frame, **mask_kwargs)
        **mask_kwargs,
    ) -> dict[str, Any]:
        """
        Detect contiguous features on each frame of stack.data.

        Parameters
        ----------
        stack : AFMImageStack
            The AFM stack whose `.data` (3D array) and `.time_for_frame()` are used.

        previous_results : dict[str, Any], optional
            Mapping of earlier analysis outputs. If `mask_key` is given,
            must contain a boolean mask array under that key.

        mask_fn : callable, optional
            Function frame->bool array for masking.
            Required if `mask_key` is None.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:

            - features_per_frame : list of lists of dict
            - labeled_masks : list of np.ndarray
            - summary : dict with total_features, total_frames, avg_features_per_frame

        Raises
        ------
        ValueError
            If `stack.data` is None or not 3D, or mask array invalid,
            or neither `mask_fn` nor `mask_key` provided.
        KeyError
            If `mask_key` not found in `previous_results`.

        Examples
        --------
        >>> pipeline.add("feature_detection", mask_fn=mask_mean_offset, min_size=20)
        >>> result = pipeline.run(stack)
        """
        data = stack.data
        if data is None:
            raise ValueError("AFMImageStack has no data")
        if not isinstance(data, np.ndarray) or data.ndim != 3:
            raise ValueError("stack.data must be a 3D numpy array (n_frames, H, W)")
        n_frames, _, _ = data.shape

        mask_arr = self._get_mask_array(
            data, previous_results, mask_fn, mask_key, **mask_kwargs
        )

        features_per_frame: list[list[dict[str, Any]]] = []
        labeled_masks: list[np.ndarray] = []
        total_features = 0

        for i in range(n_frames):
            try:
                frame_ts = float(stack.time_for_frame(i))
            except Exception:
                frame_ts = float(i)

            feats, labeled = self._process_frame(
                data[i],
                mask_arr[i].copy(),
                frame_ts,
                morph_opening=morph_opening,
                sep_radius=sep_radius,
                min_size=min_size,
                remove_edge=remove_edge,
                fill_holes=fill_holes,
                hole_area=hole_area,
            )
            features_per_frame.append(feats)
            labeled_masks.append(labeled)
            total_features += len(feats)

        return {
            "features_per_frame": features_per_frame,
            "labeled_masks": labeled_masks,
            "summary": self._summarize(n_frames, total_features),
        }
