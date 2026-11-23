"""
Analysis module that detects features using a mask of each frame.

Pixels selected by a mask (for example, everything above a height threshold)
are grouped into connected features, which can be separated, filtered by size
and edge contact, and measured.

.. versionadded:: 0.2.0

See Also
--------
playnano.analysis.modules.log_blob_detection : Detects round features without a mask.
playnano.analysis.modules.particle_tracking : Links detected features across frames.
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
    Detect connected features in each frame using a mask.

    Pipeline name: ``"feature_detection"``. For each frame:

    1. A mask is made with ``mask_fn`` (or read using ``mask_key``).
    2. If ``fill_holes``, holes in the mask are filled.
    3. Groups of mask pixels that touch, at a side or a corner, become
       features.
    4. If ``morph_opening``, touching features are separated.
    5. Features smaller than ``min_size``, or touching the frame edge if
       ``remove_edge``, are removed.
    6. The remaining features are numbered from 1 and measured.

    Version 0.2.0 added ``morph_opening``. See :meth:`run` for the parameters
    and outputs.

    Examples
    --------
    >>> import numpy as np
    >>> from playnano.afm_stack import AFMImageStack
    >>> data = np.zeros((3, 64, 64))
    >>> for t in range(3):
    ...     data[t, 10 + t : 16 + t, 10:16] = 3.0  # a 6 x 6 pixel particle
    >>> stack = AFMImageStack(
    ...     data,
    ...     pixel_size_nm=2.0,
    ...     channel="height_trace",
    ...     file_path="demo.h5-jpk",
    ...     frame_metadata=[{"timestamp": 0.5 * t} for t in range(3)],
    ... )
    >>> from playnano.analysis.modules.feature_detection import (
    ...     FeatureDetectionModule,
    ... )
    >>> result = FeatureDetectionModule().run(
    ...     stack, mask_fn="mask_threshold", threshold=1.0
    ... )
    >>> result["summary"]
    {'total_frames': 3, 'total_features': 3, 'avg_features_per_frame': 1.0}
    >>> feature = result["features_per_frame"][1][0]
    >>> feature["area"], feature["centroid"], feature["mean"]
    (36.0, (13.5, 12.5), 3.0)

    In a pipeline, the same step is added with
    ``pipeline.add("feature_detection", mask_fn="mask_threshold", threshold=1.0)``.
    """

    version = "0.2.0"

    @property
    def name(self) -> str:
        """Name of this module in a pipeline, ``"feature_detection"``."""
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
        Detect and measure features in each frame of ``stack.data``.

        Parameters
        ----------
        stack : AFMImageStack
            The stack to analyse.
        previous_results : dict, optional
            Outputs of earlier steps; used only with ``mask_key``. Default is None.
        mask_fn : callable or str, optional
            Masking function, called as ``mask_fn(frame, **mask_kwargs)`` and
            returning a boolean array the shape of the frame, or the name of one
            in :mod:`playnano.processing.mask_generators`, for example
            ``"mask_threshold"``. Give either this or ``mask_key``.
        mask_key : str, optional
            Key of a boolean array, shape (n_frames, height, width), in
            ``previous_results``. Note that ``previous_results`` is keyed by
            module name and holds dicts, so through a pipeline this cannot
            currently find a mask.
        morph_opening : bool, optional
            Separate touching features with a morphological opening (with a disk
            of radius ``sep_radius``), applied to each feature on its own. A
            feature that would lose more than 60% of its area, or all of it, is
            kept unchanged. Default is False.
        sep_radius : int, optional
            Radius of the opening disk, in pixels. Used only if ``morph_opening``.
            Default is 6.
        min_size : int, optional
            Smallest feature kept, in pixels. Default is 10.
        remove_edge : bool, optional
            Remove features that touch the edge of the frame. Default is True.
        fill_holes : bool, optional
            Fill holes in the mask before finding features. Default is False.
        hole_area : int, optional
            With ``fill_holes``, fill only holes smaller than this, in pixels.
            If None, fill all holes. Default is None.
        **mask_kwargs
            Passed to ``mask_fn``, for example ``threshold=1.0``. If ``mask_fn``
            raises ``TypeError``, it is called again without them.

        Returns
        -------
        dict
            ``"features_per_frame"``
                For each frame, a list of dicts, one per feature, with:

                - ``frame_timestamp`` (float): the frame's time, in s, or the frame
                  index if the stack has no times.
                - ``label`` (int): the feature's value in ``labeled_masks``.
                - ``area`` (float): in pixels.
                - ``min``, ``max``, ``mean`` (float): height inside the feature,
                  in the units of the data (usually nm).
                - ``bbox`` (tuple of int): ``(min_row, min_col, max_row,
                  max_col)``, with the maxima exclusive.
                - ``centroid`` (tuple of float): ``(row, col)``, in pixels.
            ``"labeled_masks"``
                For each frame, an int array the shape of the frame: 0 for
                background and each feature's ``label`` inside it.
            ``"summary"``
                ``total_frames``, ``total_features`` and
                ``avg_features_per_frame``.

        Raises
        ------
        ValueError
            If ``stack.data`` is not a 3D array, neither ``mask_fn`` nor
            ``mask_key`` is given, ``mask_fn`` names no masking function, or a
            mask is not a boolean array of the right shape.
        KeyError
            If ``mask_key`` is not in ``previous_results``.
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
