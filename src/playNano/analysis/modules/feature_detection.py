"""
Threshold-based feature detection for AFM image stacks.

Detect features in each frame of an AFM image stack through thresholding methods.
"""

from typing import Any, Optional

import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_holes

from playNano.analysis.base import AnalysisModule
from playNano.processing.mask_generators import register_masking

MASK_MAP = register_masking()


class FeatureDetectionModule(AnalysisModule):
    """
    Detect contiguous features in each frame of an AFM image stack.

    This module takes either a user-supplied mask function or a pre-computed boolean
    mask array, labels connected regions in each frame, filters them by size and edge
    contact, optionally fills holes, and returns per-frame feature statistics and
    labeled masks.

    Parameters
    ----------
    mask_fn : callable, optional
        A function `frame -> bool_2D_array` used to generate a mask for each frame.
        Required if `mask_key` is not provided.

    mask_key : str, optional
        Name of a boolean mask array from a previous analysis (e.g.
        `previous_results["your_mask_key"]`). Required if `mask_fn` is not provided.

    min_size : int
        Minimum area (in pixels) for a region to be kept. Default is 10.

    remove_edge : bool
        If True, discard any region that touches the frame boundary. Default is True.

    fill_holes : bool
        If True, fill holes in each mask before labeling. Default is False.

    hole_area : int or None
        If set, fills only holes smaller than this area. Default is None (all
        holes filled).

    **mask_kwargs : Any
        Additional keyword arguments forwarded to `mask_fn(frame, **mask_kwargs)`.


    Raises
    ------
    ValueError
        If neither `mask_fn` nor `mask_key` is provided, or if the mask array
        has the wrong shape/dtype.

    KeyError
        If `mask_key` is not found in `previous_results`.

    Returns
    -------
    dict[str, Any]
        Dictionary with the following keys:

        - features_per_frame : list of list of dict
          Per-frame list of feature stats dicts, each with:

            - `"frame_timestamp"` : float
            - `"label"`           : int
            - `"area"`            : int
            - `"min"`, `"max"`, `"mean"` : float
            - `"bbox"`            : (min_row, min_col, max_row, max_col)
            - `"centroid"`        : (row, col)

        - labeled_masks : list of np.ndarray
          The final labeled mask (integer labels) for each frame.

        - summary : dict
          Aggregate metrics:

            - `"total_frames"` : int
            - `"total_features"` : int
            - `"avg_features_per_frame"` : float

    Version
    -------
    0.1.0

    Examples
    --------
    >>> pipeline.add("feature_detection", mask_fn=mask_mean_offset, min_size=20,
    ...              fill_holes=True, hole_area=50)
    >>> result = pipeline.run(stack)
    >>> result["summary"]["total_features"]
    123
    """

    version = "0.1.0"

    @property
    def name(self) -> str:
        """
        Name of the analysis module.

        Returns
        -------
        str
            The string identifier for this module: "feature_detection".
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
        """Resolve mask array from previous results or by computing frame-by-frame."""
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

    def _process_frame(
        self,
        frame: np.ndarray,
        mask_frame: np.ndarray,
        frame_ts: float,
        *,
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
                mask_frame = remove_small_holes(mask_frame, area_threshold=hole_area)
            else:
                mask_frame = binary_fill_holes(mask_frame)
            mask_frame = mask_frame.astype(bool)

        # Label connected regions
        initial_labeled = label(mask_frame)
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
                    "area": int(prop.area),
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

    def run(
        self,
        stack,
        previous_results: Optional[dict[str, Any]] = None,
        *,
        # Mask input: either supply a mask function or refer to
        # existing mask in previous_results
        mask_fn: Optional[callable] = None,
        mask_key: Optional[str] = None,
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
