# mypy: disable-error-code=type-arg
"""
Analysis module that cuts out each tracked particle's region, frame by frame.

For each track from ``particle_tracking`` and each frame it spans, the module
finds the particle's labelled region in the ``feature_detection`` output and
records:

- its tight bounding box,
- a bounding box enlarged by padding, made square or set to a fixed size,
- a crop of the frame to that box (optional),
- a crop of the particle's mask to that box (optional).

It only organises data: per-frame measurements come from the detection step,
and later modules such as ``tracked_particle_boundary_size`` and
``particle_region_media_export`` use its output.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from skimage.measure import label as sk_label
from skimage.measure import regionprops

from playnano.analysis.base import AnalysisModule

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _tight_bbox_for_label(
    labeled_mask: np.ndarray,
    label_val: int,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Compute the tight bounding box for a labeled region.

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
    """Expand bounding box by ``padding`` pixels on each side, clipped to bounds."""
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
    """Return a fixed-size bounding box centred on ``bbox``, clipped to image bounds."""
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
    """Expand a bounding box to be square, centred on the original box."""
    minr, minc, maxr, maxc = bbox
    h, w = image_shape
    size = max(maxr - minr, maxc - minc)
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
    Cut out each tracked particle's region in every frame of its track.

    Pipeline name: ``"particle_region_extraction"``. It needs the outputs of
    ``particle_tracking`` and of a detection step that gives labelled regions
    (``feature_detection``). See :meth:`run` for the parameters and outputs.

    Bounding boxes are ``(min_row, min_col, max_row, max_col)`` in pixels, with
    the maxima exclusive, so ``frame[min_row:max_row, min_col:max_col]`` is the
    crop. Image crops are taken from ``stack.data``, so from processed data if the
    stack was processed.

    Examples
    --------
    >>> import numpy as np
    >>> from playnano.afm_stack import AFMImageStack
    >>> from playnano.analysis.pipeline import AnalysisPipeline
    >>> data = np.zeros((3, 64, 64))
    >>> for t in range(3):
    ...     data[t, 10 + t : 16 + t, 10:16] = 3.0
    >>> stack = AFMImageStack(
    ...     data,
    ...     pixel_size_nm=2.0,
    ...     channel="height_trace",
    ...     file_path="demo.h5-jpk",
    ...     frame_metadata=[{"timestamp": 0.5 * t} for t in range(3)],
    ... )
    >>> pipeline = AnalysisPipeline()
    >>> pipeline.add("feature_detection", mask_fn="mask_threshold", threshold=1.0)
    >>> pipeline.add("particle_tracking", max_distance=5)
    >>> pipeline.add("particle_region_extraction", padding=2)
    >>> record = pipeline.run(stack)
    >>> output = record["analysis"]["step_3_particle_region_extraction"]
    >>> region = output["per_track"][0]["regions"][1]
    >>> region["bbox_tight"], region["bbox_padded"], region["image_crop"].shape
    ((11, 10, 17, 16), (9, 8, 19, 18), (10, 10))
    """

    version = "0.1.0"

    @property
    def name(self) -> str:
        """Name of this module in a pipeline, ``"particle_region_extraction"``."""
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
        """Return fixed crop size for this track (per_track mode only)."""
        if fixed_box_size is None or fixed_size_mode != "per_track":
            return fixed_box_size

        sizes: List[int] = []

        for frame_idx, pt_idx in zip(frames, pt_indices, strict=False):
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
        """Apply padding, square, and fixed-size rules to produce the final bbox."""
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
        """Crop image and binary mask to ``bbox`` for ``frame_idx``."""
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
        Extract detection outputs.

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
        Find, box and crop each tracked particle's region in each frame.

        Parameters
        ----------
        stack : AFMImageStack
            The stack: crops are taken from ``stack.data`` and times from its frame
            metadata (the frame index if a frame has no time).
        previous_results : dict
            Outputs of earlier steps. Must hold the outputs of ``tracking_module``
            and ``detection_module``.
        tracking_module : str, optional
            Name of the tracking step. Default is ``"particle_tracking"``.
        detection_module : str, optional
            Name of the detection step. Its output must have ``"labeled_masks"`` and
            ``"features_per_frame"``. Default is ``"feature_detection"``.
        padding : int, optional
            Pixels added on each side of the tight box, limited by the frame edges.
            Default is 0.
        fixed_box_size : int, optional
            Make every box this many pixels square, centred on the padded box and
            shifted to stay inside the frame. Overrides ``square``. Default is None.
        square : bool, optional
            Make the padded box square, with sides equal to its longer side. Not
            used if a fixed size is set. Default is False.
        fixed_size_mode : {"global", "per_track"}, optional
            ``"global"`` uses ``fixed_box_size`` for every track. ``"per_track"``
            gives each track its own size: its largest tight box side plus
            ``2 * padding``. ``"per_track"`` only takes effect if ``fixed_box_size``
            is also set (to any value, which is then not used). Default is
            ``"global"``.
        include_image : bool, optional
            Add ``"image_crop"`` to each region. Default is True.
        include_mask : bool, optional
            Add ``"mask_crop"`` to each region. Default is True.
        include_bbox : bool, optional
            Add the boxes to each region and to ``flat_table``. Default is True.
        label_key : str, optional
            Key of each feature's label in ``labeled_masks``. Default is
            ``"label"``.

        Returns
        -------
        dict
            ``"per_track"``
                One dict per track, with ``track_id``, ``frames``, ``timestamps``
                (s), ``fixed_box_size`` (the size used for this track, or None) and
                ``regions``: for each frame, None if the particle was not detected,
                otherwise a dict with:

                - ``frame``, ``timestamp`` and ``label``;
                - ``bbox_tight``: the box around the largest connected piece of the
                  label, or None if the label is not in the mask;
                - ``bbox_padded``: the box after ``padding``, ``square`` and
                  ``fixed_box_size``;
                - ``image_crop``: the frame cropped to ``bbox_padded`` (array);
                - ``mask_crop``: boolean mask of the particle in that crop.
            ``"flat_table"``
                One dict per track and frame, with ``track_id``, ``frame``,
                ``timestamp``, ``label`` and, if ``include_bbox``, the box corners as
                ``bbox_tight_minr``, ``bbox_tight_minc``, ``bbox_tight_maxr``,
                ``bbox_tight_maxc`` and the same for ``bbox_padded``. Missing values
                are NaN. Crops are not included. Ready for :class:`pandas.DataFrame`.
            ``"summary"``
                Counts (``n_tracks``, ``n_rows``, ``n_skipped_index_errors``,
                ``n_missing_region_measurements``) and the settings used.

            Frames or feature indices out of range give a warning and a None region.

        Raises
        ------
        RuntimeError
            If ``previous_results`` is None or lacks the tracking or detection
            output (or their ``"tracks"``, ``"labeled_masks"`` or
            ``"features_per_frame"``), or a tracked feature has no ``label_key``.
        ValueError
            If ``fixed_size_mode`` is not recognised.
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
                f"{self.name!r} expected '{tracking_module}' output to contain 'tracks'."  # noqa: E501
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
                    f"!= point_indices length ({len(pt_indices)}). Truncating to shortest.",  # noqa: E501
                    stacklevel=2,
                )

            track_frames: List[int] = []
            track_timestamps: List[float] = []
            track_regions: List[Optional[Dict[str, Any]]] = []

            for frame_idx, pt_idx in zip(frames, pt_indices, strict=False):
                frame_idx = int(frame_idx)
                ts = self._get_timestamp(stack, frame_idx)

                track_frames.append(frame_idx)
                track_timestamps.append(ts)

                # ---- missing detection ----------------------------------
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

                # ---- out-of-range frame ---------------------------------
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

                # ---- out-of-range point index ---------------------------
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
        """Build a flat table row (array crops excluded by design)."""
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
