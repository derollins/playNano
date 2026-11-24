# mypy: disable-error-code=type-arg
"""
Analysis module that saves tracked particle regions as GIFs, videos or image sequences.

It uses the crops made by ``particle_region_extraction``. For each track, the
crops are padded to the size of the track's largest crop and stacked, then
written with the same rendering as the whole-stack exports:

- :func:`playnano.io.gif_export.create_gif_with_scale_and_timestamp`,
- :func:`playnano.io.video_export.create_video_with_scale_and_timestamp`,
- :func:`playnano.io.image_sequence_export.create_image_sequence`.

Any combination of the three can be written in one step.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from playnano.analysis.base import AnalysisModule
from playnano.io.gif_export import create_gif_with_scale_and_timestamp
from playnano.io.image_sequence_export import create_image_sequence
from playnano.io.render_utils import DEFAULT_FONT_SCALE
from playnano.io.video_export import create_video_with_scale_and_timestamp
from playnano.utils.colormaps import DEFAULT_CMAP
from playnano.utils.io_utils import prepare_output_directory, sanitize_output_name

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _pad_crop_to_size(
    crop: np.ndarray,
    target_h: int,
    target_w: int,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Centre-pad *crop* to ``(target_h, target_w)``.

    Parameters
    ----------
    crop : ndarray, shape (h, w)
        Source image crop.
    target_h, target_w : int
        Target spatial dimensions (must be >= crop dimensions).
    fill_value : float
        Value used to fill the padding region.

    Returns
    -------
    ndarray, shape (target_h, target_w)
    """
    h, w = crop.shape[:2]
    pad_top = (target_h - h) // 2
    pad_bottom = target_h - h - pad_top
    pad_left = (target_w - w) // 2
    pad_right = target_w - w - pad_left

    if crop.ndim == 2:
        return np.pad(
            crop,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=fill_value,
        )
    # (h, w, c)
    return np.pad(
        crop,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",
        constant_values=fill_value,
    )


def _assemble_track_stack(
    regions: List[Optional[Dict[str, Any]]],
    fill_value: float,
) -> Optional[np.ndarray]:
    """
    Assemble a ``(T, H, W)`` float32 array from per-frame region crops.

    All frames are centre-padded to the largest crop seen in the track.
    Frames with ``None`` region or missing ``image_crop`` are filled with
    ``fill_value``.

    Returns ``None`` if no valid crops are present.
    """
    max_h = max_w = 0
    for region in regions:
        if region is None:
            continue
        crop = region.get("image_crop")
        if crop is None:
            continue
        h, w = np.asarray(crop).shape[:2]
        max_h = max(max_h, h)
        max_w = max(max_w, w)

    if max_h == 0 or max_w == 0:
        return None

    frames: List[np.ndarray] = []
    for region in regions:
        raw = region.get("image_crop") if region is not None else None
        if raw is None:
            frame = np.full((max_h, max_w), fill_value, dtype=np.float32)
        else:
            frame = np.asarray(raw, dtype=np.float32)
            if frame.shape[:2] != (max_h, max_w):
                frame = _pad_crop_to_size(frame, max_h, max_w, fill_value)
        frames.append(frame)

    return np.stack(frames, axis=0)  # (T, H, W)


def _filter_and_sort_tracks(
    tracks: List[Dict[str, Any]],
    track_ids: Optional[List[int]],
    sort_by: str,
    max_tracks: Optional[int],
) -> List[Dict[str, Any]]:
    """
    Filter and sort a list of per-track dicts from ``particle_region_extraction``.

    Parameters
    ----------
    tracks : list of dict
        The ``per_track`` list from the extraction module output.
    track_ids : list of int or None
        Explicit whitelist of track IDs to retain.  ``None`` keeps all tracks.
    sort_by : {"track_id", "n_frames", "n_detections"}
        Sort key applied after ID filtering and before the ``max_tracks`` cap.

        - ``"track_id"`` : ascending by track ID (default, stable ordering).
        - ``"n_frames"`` : descending by total frame count (longest tracks
          first).
        - ``"n_detections"`` : descending by the number of frames that have a
          non-``None`` region (most consistently detected tracks first).

    max_tracks : int or None
        Maximum number of tracks to retain after sorting.  ``None`` keeps all.

    Returns
    -------
    list of dict
        Filtered, sorted, and optionally capped subset of *tracks*.

    Raises
    ------
    ValueError
        If ``sort_by`` is not one of the recognised sort keys.
    """
    _VALID_SORT_KEYS = {"track_id", "n_frames", "n_detections"}
    if sort_by not in _VALID_SORT_KEYS:
        raise ValueError(
            f"sort_by={sort_by!r} is not recognised. "
            f"Choose from {sorted(_VALID_SORT_KEYS)}."
        )

    # --- ID whitelist filter ---------------------------------------------
    if track_ids is not None:
        id_set = set(track_ids)
        tracks = [t for t in tracks if int(t["track_id"]) in id_set]

    # --- Sort ------------------------------------------------------------
    if sort_by == "track_id":
        tracks = sorted(tracks, key=lambda t: int(t["track_id"]))
    elif sort_by == "n_frames":
        tracks = sorted(
            tracks,
            key=lambda t: len(t.get("regions", [])),
            reverse=True,
        )
    elif sort_by == "n_detections":
        tracks = sorted(
            tracks,
            key=lambda t: sum(1 for r in t.get("regions", []) if r is not None),
            reverse=True,
        )

    # --- Cap -------------------------------------------------------------
    if max_tracks is not None:
        tracks = tracks[:max_tracks]

    return tracks


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


class ParticleRegionMediaExportModule(AnalysisModule):
    """
    Save each tracked particle's crops as a GIF, video and/or image sequence.

    Pipeline name: ``"particle_region_media_export"``. It needs the output of
    ``particle_region_extraction``, run with ``include_image=True`` (the
    default). Files are written to ``output_folder``, overwriting any with the
    same name:

    - GIF: ``<name>_track_<id>.gif``;
    - video: ``<name>_track_<id>.<video_fmt>``;
    - image sequence: a folder ``<name>_track_<id>/`` of ``frame_0000.png``,
      ``frame_0001.png`` and so on.

    Frames where the particle was not detected are filled with ``fill_value``.
    See :meth:`run` for the parameters and outputs.

    Examples
    --------
    After detection, tracking and region extraction, write a GIF of the ten
    tracks with the most detections:

    >>> pipeline.add(
    ...     "particle_region_media_export",
    ...     export_gif=True,
    ...     export_video=False,
    ...     sort_by="n_detections",
    ...     max_tracks=10,
    ... )  # doctest: +SKIP
    """

    version = "0.1.0"

    @property
    def name(self) -> str:
        """Name of this module in a pipeline, ``"particle_region_media_export"``."""
        return "particle_region_media_export"

    requires = ["particle_region_extraction"]

    def run(
        self,
        stack: Any,
        previous_results: Optional[Dict[str, Any]] = None,
        *,
        extraction_module: str = "particle_region_extraction",
        output_folder: Optional[str] = None,
        output_name: Optional[str] = None,
        track_ids: Optional[List[int]] = None,
        max_tracks: Optional[int] = None,
        sort_by: str = "track_id",
        export_gif: bool = False,
        export_video: bool = True,
        video_fmt: str = "mp4",
        export_sequence: bool = False,
        sequence_fmt: str = "png",
        fps: float = 5.0,
        cmap_name: str = DEFAULT_CMAP,
        zmin: Optional[float | str] = None,
        zmax: Optional[float | str] = None,
        draw_ts: bool = True,
        draw_scale: bool = True,
        scale_bar_length_nm: int = 100,
        font_scale: float = DEFAULT_FONT_SCALE,
        fill_value: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Write the GIFs, videos and image sequences for the selected tracks.

        Parameters
        ----------
        stack : AFMImageStack
            The stack: its ``pixel_size_nm`` sets the scale bar and its file name
            the default output name.
        previous_results : dict
            Outputs of earlier steps. Must hold the output of ``extraction_module``.
        extraction_module : str, optional
            Name of the region extraction step. Default is
            ``"particle_region_extraction"``.
        output_folder : str, optional
            Folder for the output; created if missing. If None, ``./output``.
        output_name : str, optional
            Start of each file name. If None, the name of the file the stack was
            loaded from.
        track_ids : list of int, optional
            Export only these tracks. IDs not found are ignored. If None, all
            tracks. Default is None.
        max_tracks : int, optional
            Export at most this many tracks, after ``track_ids`` and ``sort_by``.
            Default is None (no limit).
        sort_by : {"track_id", "n_frames", "n_detections"}, optional
            Order of the tracks before ``max_tracks`` is applied: by ID
            (ascending), by number of frames spanned, or by number of frames with a
            detection (both descending). Default is ``"track_id"``.
        export_gif : bool, optional
            Write a GIF per track. Default is False.
        export_video : bool, optional
            Write a video per track. Default is True.
        video_fmt : {"mp4", "avi", "mov", "mkv"}, optional
            Video format. Default is ``"mp4"``.
        export_sequence : bool, optional
            Write a folder of images per track. Default is False.
        sequence_fmt : {"png", "jpg", "jpeg"}, optional
            Image format of the sequences. Default is ``"png"``.
        fps : float, optional
            Frames per second of the GIFs and videos. Default is 5.0.
        cmap_name : str, optional
            Name of a registered matplotlib colormap. Default is ``"afm_brown"``.
        zmin, zmax : float or str, optional
            Colour scale limits, as for :func:`playnano.io.gif_export.export_gif`:
            both None scales each frame separately; ``"auto"`` uses the 1st or
            99th percentile of the track. Padding and missing frames
            (``fill_value``) count towards both, so they can change the colours.
            Default is None.
        draw_ts : bool, optional
            Draw each frame's time. Default is True.
        draw_scale : bool, optional
            Draw a scale bar. Default is True.
        scale_bar_length_nm : int, optional
            Length of the scale bar, in nm; 0 hides it. Default is 100.
        font_scale : float, optional
            Size of the annotation text for a frame 512 pixels high. Default is 2.0.
        fill_value : float, optional
            Value used for padding and for frames where the particle was not
            detected. Default is 0.0.

        Returns
        -------
        dict
            ``"gif_paths"``, ``"video_paths"``, ``"sequence_folders"``
                Absolute paths of what was written, as str.
            ``"per_track"``
                One dict per track written, with ``track_id``, ``n_frames``,
                ``frame_size`` (height, width, in pixels) and its ``gif_path``,
                ``video_path`` and ``sequence_folder``.
            ``"summary"``
                Counts of tracks found, selected, written, skipped (no crops) and
                filtered out, and the settings used.

        Raises
        ------
        RuntimeError
            If no export is enabled, ``previous_results`` is None or lacks the
            extraction output, or the extraction ran with ``include_image=False``.
        ValueError
            If ``sort_by`` is not recognised, ``sequence_fmt`` is not supported, or
            ``zmin`` is greater than ``zmax``.

        Notes
        -----
        Formats are not checked before writing starts, so with an unsupported
        format some files may already be written when the error is raised. An
        unsupported ``video_fmt`` raises ``FileNotFoundError`` rather than
        ``ValueError``. The scale bar uses the stack's pixel size, not each frame's.
        """
        if previous_results is None:
            raise RuntimeError(f"{self.name!r} requires previous results to run.")

        if not export_gif and not export_video and not export_sequence:
            raise RuntimeError(
                f"{self.name!r}: at least one of export_gif, export_video, or "
                f"export_sequence must be True."
            )

        if extraction_module not in previous_results:
            raise RuntimeError(
                f"{self.name!r} requires extraction_module={extraction_module!r} "
                f"to be present in previous_results."
            )

        ext_out = previous_results[extraction_module]
        if not ext_out.get("summary", {}).get("include_image", True):
            raise RuntimeError(
                f"{self.name!r} requires the upstream {extraction_module!r} to have "
                f"been run with include_image=True."
            )

        # Normalise and validate format strings early so we fail fast.
        video_fmt = video_fmt.lower().lstrip(".")
        sequence_fmt = sequence_fmt.lower().lstrip(".")

        out_dir = prepare_output_directory(output_folder, default="output")
        base = sanitize_output_name(output_name, Path(stack.file_path).stem)
        pixel_size_nm: float = float(getattr(stack, "pixel_size_nm", 1.0))

        all_tracks = ext_out.get("per_track", [])
        selected_tracks = _filter_and_sort_tracks(
            all_tracks,
            track_ids=track_ids,
            sort_by=sort_by,
            max_tracks=max_tracks,
        )
        n_filtered_out = len(all_tracks) - len(selected_tracks)
        if n_filtered_out:
            logger.info(
                f"[{self.name}] {n_filtered_out} track(s) excluded by "
                f"track_ids/max_tracks filter; exporting {len(selected_tracks)}."
            )

        gif_paths: List[str] = []
        video_paths: List[str] = []
        sequence_folders: List[str] = []
        per_track_manifest: List[Dict[str, Any]] = []
        n_skipped_tracks = 0

        for trk in selected_tracks:
            track_id = int(trk["track_id"])
            regions: List[Optional[Dict[str, Any]]] = trk.get("regions", [])
            timestamps: List[float] = trk.get("timestamps", [])

            # --- Assemble padded (T, H, W) array once for this track -----
            track_stack = _assemble_track_stack(regions, fill_value)

            if track_stack is None:
                logger.warning(
                    f"[{self.name}] track_id={track_id}: no valid image crops found. "
                    f"Skipping export for this track."
                )
                n_skipped_tracks += 1
                continue

            n_t, target_h, target_w = track_stack.shape
            pixel_sizes_nm = [pixel_size_nm] * n_t
            track_stem = f"{base}_track_{track_id}"

            trk_record: Dict[str, Any] = {
                "track_id": track_id,
                "n_frames": n_t,
                "frame_size": (target_h, target_w),
            }

            # --- GIF export ----------------------------------------------
            if export_gif:
                gif_path = out_dir / f"{track_stem}.gif"
                create_gif_with_scale_and_timestamp(
                    track_stack,
                    pixel_sizes_nm=pixel_sizes_nm,
                    timestamps=timestamps[:n_t],
                    scale_bar_length_nm=scale_bar_length_nm,
                    output_path=gif_path,
                    fps=fps,
                    cmap_name=cmap_name,
                    zmin=zmin,
                    zmax=zmax,
                    draw_ts=draw_ts,
                    draw_scale=draw_scale,
                    font_scale=font_scale,
                )
                logger.info(f"[{self.name}] GIF written → {gif_path}")
                gif_paths.append(str(gif_path))
                trk_record["gif_path"] = str(gif_path)

            # --- Video export --------------------------------------------
            if export_video:
                video_path = out_dir / f"{track_stem}.{video_fmt}"
                create_video_with_scale_and_timestamp(
                    track_stack,
                    pixel_sizes_nm=pixel_sizes_nm,
                    timestamps=timestamps[:n_t],
                    scale_bar_length_nm=scale_bar_length_nm,
                    output_path=video_path,
                    fps=fps,
                    cmap_name=cmap_name,
                    zmin=zmin,
                    zmax=zmax,
                    draw_ts=draw_ts,
                    draw_scale=draw_scale,
                    font_scale=font_scale,
                )
                logger.info(f"[{self.name}] Video written → {video_path}")
                video_paths.append(str(video_path))
                trk_record["video_path"] = str(video_path)

            # --- Image sequence export -----------------------------------
            if export_sequence:
                seq_folder = out_dir / track_stem
                create_image_sequence(
                    track_stack,
                    pixel_sizes_nm=pixel_sizes_nm,
                    timestamps=timestamps[:n_t],
                    scale_bar_length_nm=scale_bar_length_nm,
                    output_folder=seq_folder,
                    base_name="frame",
                    fmt=sequence_fmt,
                    cmap_name=cmap_name,
                    zmin=zmin,
                    zmax=zmax,
                    draw_ts=draw_ts,
                    draw_scale=draw_scale,
                    font_scale=font_scale,
                )
                logger.info(f"[{self.name}] Image sequence written → {seq_folder}")
                sequence_folders.append(str(seq_folder))
                trk_record["sequence_folder"] = str(seq_folder)

            per_track_manifest.append(trk_record)

        return {
            "gif_paths": gif_paths,
            "video_paths": video_paths,
            "sequence_folders": sequence_folders,
            "per_track": per_track_manifest,
            "summary": {
                "n_tracks_total": len(all_tracks),
                "n_tracks_selected": len(selected_tracks),
                "n_tracks_written": len(per_track_manifest),
                "n_tracks_skipped": n_skipped_tracks,
                "n_tracks_filtered_out": n_filtered_out,
                "track_ids": track_ids,
                "max_tracks": max_tracks,
                "sort_by": sort_by,
                "output_folder": str(out_dir),
                "export_gif": export_gif,
                "export_video": export_video,
                "video_fmt": video_fmt if export_video else None,
                "export_sequence": export_sequence,
                "sequence_fmt": sequence_fmt if export_sequence else None,
                "fps": fps,
                "cmap_name": cmap_name,
                "scale_bar_length_nm": scale_bar_length_nm,
                "draw_ts": draw_ts,
                "draw_scale": draw_scale,
            },
        }
