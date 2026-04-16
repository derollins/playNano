# mypy: disable-error-code=type-arg
"""
Particle region extraction module for the playNano analysis pipeline.

This module extracts per-particle image regions over time using labeled masks
from a detection module (e.g. ``feature_detection``) and particle associations
from a tracking module (e.g. ``particle_tracking``).

For each tracked particle and each frame, the module locates the labeled region
associated with that particle and extracts:

- the tight bounding-box coordinates of the region,
- an optionally padded bounding box (clipped to image bounds),
- a crop of the raw image stack frame (optional),
- a crop of the binary object mask (optional).

All outputs are ``None`` / ``np.nan`` for frames where the particle is absent
(dense tracks with missing detections) or where indices are out of range.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from playnano.analysis.base import AnalysisModule
from skimage.measure import label as sk_label, regionprops


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _tight_bbox_for_label(
    labeled_mask: np.ndarray,
    label_val: int,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Return the tight bounding box ``(minr, minc, maxr, maxc)`` for a labeled
    region, or ``None`` if the label is absent from the mask.

    The bounding box is derived from the largest connected component that
    carries ``label_val`` in ``labeled_mask``.

    Parameters
    ----------
    labeled_mask : ndarray of int
        Integer-labeled connected-component image (one label per region).
    label_val : int
        Integer label of the region of interest.

    Returns
    -------
    tuple of int or None
        ``(minr, minc, maxr, maxc)`` for the largest matching component, or
        ``None`` if no pixels carry ``label_val``.
    """
    binary = labeled_mask == label_val
    if not np.any(binary):
        return None

    relabeled = sk_label(binary)
    props = regionprops(relabeled)
    if not props:
        return None

    region = max(props, key=lambda r: r.area)
    minr, minc, maxr, maxc = region.bbox
    return int(minr), int(minc), int(maxr), int(maxc)


def _pad_bbox(
    bbox: Tuple[int, int, int, int],
    padding: int,
    image_shape: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    """
    Expand a bounding box by ``padding`` pixels on each side, clipped to
    ``image_shape``.

    Parameters
    ----------
    bbox : (minr, minc, maxr, maxc)
        Tight bounding box to expand.
    padding : int
        Number of pixels to add on each side.
    image_shape : (height, width)
        Shape of the image used for clipping.

    Returns
    -------
    (minr, minc, maxr, maxc)
        Padded and clipped bounding box.
    """
    minr, minc, maxr, maxc = bbox
    h, w = image_shape
    return (
        max(0, minr - padding),
        max(0, minc - padding),
        min(h, maxr + padding),
        min(w, maxc + padding),
    )


def _centered_fixed_bbox(
    bbox: Tuple[int, int, int, int],
    size: int,
    image_shape: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    """
    Return a fixed-size, centered bounding box clipped to image bounds.
    """
    minr, minc, maxr, maxc = bbox
    h, w = image_shape

    cr = (minr + maxr) // 2
    cc = (minc + maxc) // 2
    half = size // 2

    minr = max(0, cr - half)
    minc = max(0, cc - half)
    maxr = min(h, minr + size)
    maxc = min(w, minc + size)

    # Re-shift if clipping occurred
    minr = max(0, maxr - size)
    minc = max(0, maxc - size)

    return minr, minc, maxr, maxc


def _square_bbox(
    bbox: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    minr, minc, maxr, maxc = bbox
    h, w = image_shape

    height = maxr - minr
    width = maxc - minc
    size = max(height, width)

    cr = (minr + maxr) // 2
    cc = (minc + maxc) // 2
    half = size // 2

    minr = max(0, cr - half)
    minc = max(0, cc - half)
    maxr = min(h, minr + size)
    maxc = min(w, minc + size)

    minr = max(0, maxr - size)
    minc = max(0, maxc - size)

    return minr, minc, maxr, maxc


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


class ParticleRegionExtractionModule(AnalysisModule):
    """
    Extract per-particle image regions over time from tracked, labeled detections.

    For each tracked particle and frame the module resolves the labeled region
    from the detection output, computes a tight bounding box, optionally pads
    it, and (optionally) crops the raw image frame and/or binary mask.

    The module requires:
    - output from a particle tracking module (e.g. ``particle_tracking``)
    - output from a labeled detection module providing:
        * ``features_per_frame`` — list of per-frame feature dicts, each
          containing a region label under ``label_key``.
        * ``labeled_masks`` — list of 2-D integer-labeled NumPy arrays.

    Parameters accepted in :meth:`run`
    ------------------------------------
    padding : int, default 0
        Pixels to add around the tight bounding box (clipped to image bounds).
    include_image : bool, default True
        Whether to include a crop of the raw image frame.
    include_mask : bool, default True
        Whether to include a crop of the binary object mask.
    include_bbox : bool, default True
        Whether to include tight and padded bounding-box coordinates.

    Notes
    -----
    - Out-of-range frame or point indices produce ``None`` region records and
      increment ``n_skipped_index_errors`` in the summary.
    - Missing detections in dense tracks (``point_index is None``) produce
      ``None`` region records without counting as errors.
    - If a feature dict is missing the ``label_key``, a ``RuntimeError`` is
      raised.
    - Raw image frames are retrieved via ``stack[frame_idx]``; if ``stack``
      does not support integer indexing a ``RuntimeError`` is raised.
    """

    version = "0.1.0"

    @property
    def name(self) -> str:
        return "particle_region_extraction"

    requires = ["particle_tracking"]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_timestamp(self, stack: Any, frame_idx: int) -> float:
        """Return the timestamp for *frame_idx*, falling back to the index."""
        try:
            return float(stack.time_for_frame(frame_idx))
        except Exception:
            return float(frame_idx)

    def _get_frame_image(self, stack: Any, frame_idx: int) -> np.ndarray:
        """
        Retrieve the raw 2-D image array for *frame_idx* from *stack*.

        Raises
        ------
        RuntimeError
            If the stack does not support integer indexing.
        """
        try:
            frame = stack[frame_idx]
        except (TypeError, KeyError) as exc:
            raise RuntimeError(
                f"[{self.name}] Cannot retrieve frame {frame_idx} from stack via "
                f"stack[frame_idx]. Ensure the stack supports integer indexing."
            ) from exc
        return np.asarray(frame)

    def _compute_track_fixed_size(
        self,
        frames: List[int],
        pt_indices: List[Optional[int]],
        features_per_frame: List[List[Dict[str, Any]]],
        labeled_masks: List[np.ndarray],
        *,
        label_key: str,
        padding: int,
        fixed_box_size: Optional[int],
        fixed_size_mode: str,
        n_frames: int,
    ) -> Optional[int]:
        """
        Determine fixed crop size for a track.
        """
        if fixed_box_size is None or fixed_size_mode != "per_track":
            return fixed_box_size

        sizes: List[int] = []

        for frame_idx, pt_idx in zip(frames, pt_indices):
            if pt_idx is None or frame_idx < 0 or frame_idx >= n_frames:
                continue

            feat = features_per_frame[frame_idx][pt_idx]
            label_val = feat[label_key]
            bbox = _tight_bbox_for_label(labeled_masks[frame_idx], int(label_val))
            if bbox is not None:
                h = bbox[2] - bbox[0]
                w = bbox[3] - bbox[1]
                sizes.append(max(h, w))

        if not sizes:
            return fixed_box_size

        return max(sizes) + 2 * padding

    def _resolve_bbox(
        self,
        bbox_tight: Tuple[int, int, int, int],
        *,
        image_shape: Tuple[int, int],
        padding: int,
        square: bool,
        track_fixed_size: Optional[int],
    ) -> Tuple[int, int, int, int]:
        """
        Resolve the final bounding box according to padding, square and fixed-size rules.
        """
        bbox = (
            _pad_bbox(bbox_tight, padding, image_shape) if padding > 0 else bbox_tight
        )

        has_fixed_size = track_fixed_size is not None

        if square and not has_fixed_size:
            bbox = _square_bbox(bbox, image_shape)

        if has_fixed_size:
            bbox = _centered_fixed_bbox(
                bbox,
                track_fixed_size,
                image_shape,
            )

        return bbox

    def _extract_crops(
        self,
        *,
        stack: Any,
        lm: np.ndarray,
        label_val: int,
        bbox: Tuple[int, int, int, int],
        frame_idx: int,
        include_image: bool,
        include_mask: bool,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        pr0, pc0, pr1, pc1 = bbox

        mask_crop = lm[pr0:pr1, pc0:pc1] == label_val if include_mask else None

        image_crop = (
            self._get_frame_image(stack, frame_idx)[pr0:pr1, pc0:pc1]
            if include_image
            else None
        )

        return image_crop, mask_crop

    def _extract_detection_outputs(
        self,
        previous_results: Dict[str, Any],
        detection_module: str,
    ) -> Tuple[List[np.ndarray], List[List[Dict[str, Any]]]]:
        """
        Extract ``labeled_masks`` and ``features_per_frame`` from
        ``previous_results[detection_module]``.

        Raises
        ------
        RuntimeError
            If the detection module output is absent or missing required keys.
        """
        if detection_module not in previous_results:
            raise RuntimeError(
                f"{self.name!r} requires detection_module={detection_module!r} "
                f"to be present in previous_results."
            )
        det_out = previous_results[detection_module]
        missing = [
            k for k in ("labeled_masks", "features_per_frame") if k not in det_out
        ]
        if missing:
            raise RuntimeError(
                f"{self.name!r} requires {detection_module!r} output to contain "
                f"{missing}. Ensure you are using a detection module that produces "
                f"labeled regions (e.g. feature_detection)."
            )
        return det_out["labeled_masks"], det_out["features_per_frame"]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        stack: Any,
        previous_results: Optional[Dict[str, Any]] = None,
        *,
        tracking_module: str = "particle_tracking",
        detection_module: str = "feature_detection",
        padding: int = 0,
        fixed_box_size: Optional[int] = None,
        square: bool = False,
        fixed_size_mode: str = "global",  # "global" | "per_track"
        include_image: bool = True,
        include_mask: bool = True,
        include_bbox: bool = True,
        label_key: str = "label",
    ) -> Dict[str, Any]:
        """
        Execute the region extraction analysis.

        Parameters
        ----------
        stack : AFMImageStack
            Input image stack.  Must support ``stack[frame_idx]`` to retrieve
            raw 2-D image arrays when ``include_image=True``.
        previous_results : dict, optional
            Outputs from earlier pipeline modules.  Must contain results for
            both ``tracking_module`` and ``detection_module``.
        tracking_module : str, default ``"particle_tracking"``
            Key identifying the tracking module output in ``previous_results``.
        detection_module : str, default ``"feature_detection"``
            Key identifying the labeled detection module output in
            ``previous_results``.
        padding : int, default 0
            Number of pixels to add around the tight bounding box on each
            side.  The padded box is clipped to the image boundary.
        fixed_box_size : int or None, default None
            If given, enforce a fixed-size crop (in pixels). The box is centered on
            the detected region and clipped to image bounds.
        square : bool, default False
            If True and no fixed box size is used, force the bounding box to be square.
        fixed_size_mode : {"global", "per_track"}, default "global"
            If "global", use fixed_box_size directly.
            If "per_track", compute one fixed size per track (max over frames).
        include_image : bool, default True
            If ``True``, the output region dict contains ``"image_crop"`` — a
            NumPy array cropped from the raw frame using the padded bounding box.
        include_mask : bool, default True
            If ``True``, the output region dict contains ``"mask_crop"`` — a
            boolean NumPy array cropped from the binary object mask using the
            padded bounding box.
        include_bbox : bool, default True
            If ``True``, the output region dict contains ``"bbox_tight"`` and
            ``"bbox_padded"`` — tuples ``(minr, minc, maxr, maxc)``.
        label_key : str, default ``"label"``
            Key used to look up the region integer label in each feature dict.

        Returns
        -------
        dict
            Keys:

            ``per_track`` : list of dict
                One entry per track.  Each entry has:

                - ``"track_id"`` : int
                - ``"frames"`` : list of int
                - ``"timestamps"`` : list of float
                - ``"regions"`` : list of dict or None
                    One entry per frame.  ``None`` indicates a missing
                    detection.  Otherwise a dict with any subset of:

                    - ``"frame"`` : int
                    - ``"timestamp"`` : float
                    - ``"label"`` : int or nan
                    - ``"bbox_tight"`` : (minr, minc, maxr, maxc) or None
                    - ``"bbox_padded"`` : (minr, minc, maxr, maxc) or None
                    - ``"image_crop"`` : ndarray or None
                    - ``"mask_crop"`` : ndarray or None

            ``flat_table`` : list of dict
                Row-per-frame representation (excludes array crops) suitable
                for conversion to a ``pandas.DataFrame``.

            ``summary`` : dict
                Bookkeeping counters and configuration echo.
        """
        if previous_results is None:
            raise RuntimeError(f"{self.name!r} requires previous results to run.")

        if tracking_module not in previous_results:
            raise RuntimeError(
                f"{self.name!r} requires tracking_module={tracking_module!r} "
                f"to be present in previous_results."
            )

        track_out = previous_results[tracking_module]
        if "tracks" not in track_out:
            raise RuntimeError(
                f"{self.name!r} expected '{tracking_module}' output to contain 'tracks'."
            )

        labeled_masks, features_per_frame = self._extract_detection_outputs(
            previous_results, detection_module
        )

        n_frames = min(len(labeled_masks), len(features_per_frame))
        if n_frames == 0:
            return self._empty_result(
                padding, include_image, include_mask, include_bbox
            )

        if fixed_size_mode not in {"global", "per_track"}:
            raise ValueError(
                f"{self.name}: fixed_size_mode must be 'global' or 'per_track', "
                f"got {fixed_size_mode!r}"
            )

        per_track: List[Dict[str, Any]] = []
        flat_table: List[Dict[str, Any]] = []
        n_skipped = 0
        n_missing = 0

        for trk in track_out["tracks"]:
            track_id = int(trk["id"])
            frames = list(trk.get("frames", []))
            pt_indices = list(trk.get("point_indices", []))

            track_fixed_size = self._compute_track_fixed_size(
                frames,
                pt_indices,
                features_per_frame,
                labeled_masks,
                label_key=label_key,
                padding=padding,
                fixed_box_size=fixed_box_size,
                fixed_size_mode=fixed_size_mode,
                n_frames=n_frames,
            )

            if len(frames) != len(pt_indices):
                warnings.warn(
                    f"[{self.name}] track_id={track_id}: frames length ({len(frames)}) "
                    f"!= point_indices length ({len(pt_indices)}). Truncating to shortest.",
                    stacklevel=2,
                )

            track_frames: List[int] = []
            track_timestamps: List[float] = []
            track_regions: List[Optional[Dict[str, Any]]] = []

            for frame_idx, pt_idx in zip(frames, pt_indices, strict=True):
                frame_idx = int(frame_idx)
                ts = self._get_timestamp(stack, frame_idx)

                track_frames.append(frame_idx)
                track_timestamps.append(ts)

                # ---- missing detection in dense track --------------------
                if pt_idx is None:
                    track_regions.append(None)
                    flat_table.append(
                        self._flat_row(
                            track_id,
                            frame_idx,
                            ts,
                            label_val=np.nan,
                            bbox_tight=None,
                            bbox_padded=None,
                            include_bbox=include_bbox,
                        )
                    )
                    n_missing += 1
                    continue

                pt_idx = int(pt_idx)

                # ---- out-of-range frame ----------------------------------
                if frame_idx < 0 or frame_idx >= n_frames:
                    warnings.warn(
                        f"[{self.name}] track_id={track_id}: frame {frame_idx} "
                        f"out of range [0, {n_frames}). Writing None.",
                        stacklevel=2,
                    )
                    track_regions.append(None)
                    flat_table.append(
                        self._flat_row(
                            track_id,
                            frame_idx,
                            ts,
                            label_val=np.nan,
                            bbox_tight=None,
                            bbox_padded=None,
                            include_bbox=include_bbox,
                        )
                    )
                    n_skipped += 1
                    continue

                feats_this = features_per_frame[frame_idx]

                # ---- out-of-range point index ----------------------------
                if pt_idx < 0 or pt_idx >= len(feats_this):
                    warnings.warn(
                        f"[{self.name}] track_id={track_id}, frame={frame_idx}: "
                        f"point_index {pt_idx} out of range. Writing None.",
                        stacklevel=2,
                    )
                    track_regions.append(None)
                    flat_table.append(
                        self._flat_row(
                            track_id,
                            frame_idx,
                            ts,
                            label_val=np.nan,
                            bbox_tight=None,
                            bbox_padded=None,
                            include_bbox=include_bbox,
                        )
                    )
                    n_skipped += 1
                    continue

                # ---- valid detection ------------------------------------
                feat = feats_this[pt_idx]
                if label_key not in feat or feat[label_key] is None:
                    raise RuntimeError(
                        f"[{self.name}] Feature at frame={frame_idx}, "
                        f"point_index={pt_idx} is missing '{label_key}'. "
                        f"This module requires labeled regions (e.g. from "
                        f"feature_detection)."
                    )

                label_val = int(feat[label_key])
                lm = np.asarray(labeled_masks[frame_idx])
                image_shape = lm.shape[:2]

                bbox_tight = _tight_bbox_for_label(lm, label_val)

                if bbox_tight is None:
                    n_missing += 1
                    bbox_padded = None
                    image_crop = None
                    mask_crop = None
                else:
                    bbox_padded = self._resolve_bbox(
                        bbox_tight,
                        image_shape=image_shape,
                        padding=padding,
                        square=square,
                        track_fixed_size=track_fixed_size,
                    )

                    image_crop, mask_crop = self._extract_crops(
                        stack=stack,
                        lm=lm,
                        label_val=label_val,
                        bbox=bbox_padded,
                        frame_idx=frame_idx,
                        include_image=include_image,
                        include_mask=include_mask,
                    )

                region: Dict[str, Any] = {
                    "frame": frame_idx,
                    "timestamp": ts,
                    "label": label_val,
                }

                if include_bbox:
                    region["bbox_tight"] = bbox_tight
                    region["bbox_padded"] = bbox_padded

                if include_image:
                    region["image_crop"] = image_crop

                if include_mask:
                    region["mask_crop"] = mask_crop

                track_regions.append(region)
                flat_table.append(
                    self._flat_row(
                        track_id,
                        frame_idx,
                        ts,
                        label_val=label_val,
                        bbox_tight=bbox_tight,
                        bbox_padded=bbox_padded,
                        include_bbox=include_bbox,
                    )
                )

            per_track.append(
                {
                    "track_id": track_id,
                    "frames": track_frames,
                    "timestamps": track_timestamps,
                    "regions": track_regions,
                    "fixed_box_size": track_fixed_size,
                }
            )

        return {
            "per_track": per_track,
            "flat_table": flat_table,
            "summary": {
                "n_tracks": len(track_out["tracks"]),
                "n_rows": len(flat_table),
                "n_skipped_index_errors": n_skipped,
                "n_missing_region_measurements": n_missing,
                "padding": padding,
                "fixed_box_size": fixed_box_size,
                "fixed_size_mode": fixed_size_mode,
                "square": square,
                "include_image": include_image,
                "include_mask": include_mask,
                "include_bbox": include_bbox,
            },
        }

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flat_row(
        track_id: int,
        frame_idx: int,
        ts: float,
        label_val: Any,
        bbox_tight: Optional[Tuple[int, int, int, int]],
        bbox_padded: Optional[Tuple[int, int, int, int]],
        include_bbox: bool,
    ) -> Dict[str, Any]:
        """Build a flat table row (array crops are excluded by design)."""
        row: Dict[str, Any] = {
            "track_id": track_id,
            "frame": frame_idx,
            "timestamp": ts,
            "label": label_val,
        }
        if include_bbox:
            if bbox_tight is not None:
                minr, minc, maxr, maxc = bbox_tight
                row["bbox_tight_minr"] = minr
                row["bbox_tight_minc"] = minc
                row["bbox_tight_maxr"] = maxr
                row["bbox_tight_maxc"] = maxc
            else:
                for k in (
                    "bbox_tight_minr",
                    "bbox_tight_minc",
                    "bbox_tight_maxr",
                    "bbox_tight_maxc",
                ):
                    row[k] = np.nan

            if bbox_padded is not None:
                minr, minc, maxr, maxc = bbox_padded
                row["bbox_padded_minr"] = minr
                row["bbox_padded_minc"] = minc
                row["bbox_padded_maxr"] = maxr
                row["bbox_padded_maxc"] = maxc
            else:
                for k in (
                    "bbox_padded_minr",
                    "bbox_padded_minc",
                    "bbox_padded_maxr",
                    "bbox_padded_maxc",
                ):
                    row[k] = np.nan

        return row

    @staticmethod
    def _empty_result(
        padding: int,
        include_image: bool,
        include_mask: bool,
        include_bbox: bool,
    ) -> Dict[str, Any]:
        return {
            "per_track": [],
            "flat_table": [],
            "summary": {
                "n_tracks": 0,
                "n_rows": 0,
                "n_skipped_index_errors": 0,
                "n_missing_region_measurements": 0,
                "padding": padding,
                "include_image": include_image,
                "include_mask": include_mask,
                "include_bbox": include_bbox,
            },
        }
