# mypy: disable-error-code=type-arg
"""
Particle region media export module for the playNano analysis pipeline.

This module exports per-particle GIFs, video files, and/or image sequences
using the cropped image regions produced by ``ParticleRegionExtractionModule``.

For each tracked particle the per-frame crops are centre-padded to a
consistent spatial size, assembled once into a ``(T, H, W)`` NumPy array, and
passed to whichever of the following rendering functions are enabled:

- :func:`playnano.io.gif_export.create_gif_with_scale_and_timestamp` for
  animated GIF output.
- :func:`playnano.io.video_export.create_video_with_scale_and_timestamp` for
  MP4/AVI/MOV/MKV output.
- :func:`playnano.io.image_sequence_export.create_image_sequence` for
  per-frame PNG/JPEG output.

Any combination of the three export types can be produced in a single pipeline
step from the same assembled stack.  At least one must be enabled.

Output paths follow the pattern ``<base>_track_<id>.<ext>``; image sequences
are written into per-track sub-directories ``<base>_track_<id>/`` with frames
named ``frame_NNNN.<fmt>``.

The module is designed to integrate with the playNano ``AnalysisPipeline``.
Its primary side-effects are writing files to disk; the returned results dict
records a manifest of everything written, plus a summary.
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
# Internal helpers  (shared with particle_region_gif and particle_region_tiff)
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


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


class ParticleRegionMediaExportModule(AnalysisModule):
    """
    Export per-track videos and/or image sequences from pre-extracted particle
    image regions.

    One output is produced per tracked particle per enabled export type.
    Both types share the same padded ``(T, H, W)`` stack assembly step, so
    enabling both incurs no redundant work beyond the two separate write calls.

    The module requires:
    - output from ``ParticleRegionExtractionModule`` (``"particle_region_extraction"``)
      run with ``include_image=True``.

    Parameters accepted in :meth:`run`
    ------------------------------------
    export_gif : bool
        Whether to write an animated GIF per track.  Default ``False``.
    export_video : bool
        Whether to write a video file per track.  Default ``True``.
    video_fmt : str
        Video container format: ``"mp4"``, ``"avi"``, ``"mov"``, or
        ``"mkv"``.  Default ``"mp4"``.
    export_sequence : bool
        Whether to write a per-frame image sequence per track.  Default
        ``False``.
    sequence_fmt : str
        Image format for sequences: ``"png"``, ``"jpg"``, or ``"jpeg"``.
        Default ``"png"``.
    output_folder : str or None
        Directory to write output into.  Defaults to ``"output"``.
    output_name : str or None
        Base name stem, derived from the stack file name when ``None``.
        Videos are written as ``<base>_track_<id>.<video_fmt>``.
        Sequence sub-directories are named ``<base>_track_<id>/``.
    fps : float
        Playback frame rate in frames per second.  Default ``5.0``.
    cmap_name : str
        Matplotlib colourmap name.  Default ``DEFAULT_CMAP``.
    zmin, zmax : float, str, or None
        Z-range for normalisation forwarded to the rendering functions.
        ``"auto"`` uses the 1st/99th percentile across all crops for the
        track; ``None`` normalises each frame independently.
    draw_ts : bool
        Whether to annotate frames with a timestamp.  Default ``True``.
    draw_scale : bool
        Whether to draw a scale bar.  Default ``True``.
    scale_bar_length_nm : int
        Scale bar length in nm.  Default ``100``.
    font_scale : float
        Font scale for annotations.  Default ``DEFAULT_FONT_SCALE``.
    fill_value : float
        Fill value for padding and missing-frame placeholders.  Default ``0.0``.
    extraction_module : str
        Key of the upstream extraction module in ``previous_results``.
        Default ``"particle_region_extraction"``.
    """

    version = "0.1.0"

    @property
    def name(self) -> str:
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
        Execute the video and/or image-sequence export.

        Parameters
        ----------
        stack : AFMImageStack
            Input stack.  ``stack.pixel_size_nm`` and ``stack.file_path`` are
            used for scale bar calibration and default output naming.
        previous_results : dict, optional
            Outputs from earlier pipeline modules.  Must contain output from
            ``extraction_module`` with ``include_image=True``.
        extraction_module : str, default ``"particle_region_extraction"``
            Key identifying the region extraction module output.
        output_folder : str or None
            Directory to write output.  Defaults to ``"output"``.
        output_name : str or None
            Base file name stem.  Track ID is appended automatically.
        export_gif : bool, default False
            Write an animated GIF (``<base>_track_<id>.gif``) per track.
        export_video : bool, default True
            Write a video file (``<base>_track_<id>.<video_fmt>``) per track.
        video_fmt : str, default ``"mp4"``
            Video container format.  One of ``"mp4"``, ``"avi"``, ``"mov"``,
            ``"mkv"``.
        export_sequence : bool, default False
            Write a folder of per-frame images
            (``<base>_track_<id>/frame_NNNN.<sequence_fmt>``) per track.
        sequence_fmt : str, default ``"png"``
            Image format for sequence frames.  One of ``"png"``, ``"jpg"``,
            ``"jpeg"``.
        fps : float, default 5.0
            Playback frame rate in frames per second (video only).
        cmap_name : str, default ``DEFAULT_CMAP``
            Matplotlib colourmap for false-colour rendering.
        zmin : float, str, or None
            Lower bound for z-normalisation, forwarded to the rendering
            functions.
        zmax : float, str, or None
            Upper bound for z-normalisation, forwarded to the rendering
            functions.
        draw_ts : bool, default True
            Whether to overlay timestamps on frames.
        draw_scale : bool, default True
            Whether to overlay a scale bar on frames.
        scale_bar_length_nm : int, default 100
            Scale bar length in nanometres.
        font_scale : float, default ``DEFAULT_FONT_SCALE``
            Font scale for timestamp and scale bar annotations.
        fill_value : float, default 0.0
            Value used to fill padded pixels and missing-frame placeholders.

        Returns
        -------
        dict
            Keys:

            ``gif_paths`` : list of str
                Absolute paths of written GIF files.  Empty when
                ``export_gif=False``.
            ``video_paths`` : list of str
                Absolute paths of written video files.  Empty when
                ``export_video=False``.
            ``sequence_folders`` : list of str
                Absolute paths of written sequence folders.  Empty when
                ``export_sequence=False``.
            ``per_track`` : list of dict
                One entry per track containing ``track_id``, ``n_frames``,
                ``frame_size``, and whichever of ``gif_path`` /
                ``video_path`` / ``sequence_folder`` were produced.
            ``summary`` : dict
                Bookkeeping counters and configuration echo.

        Raises
        ------
        RuntimeError
            If neither ``export_video`` nor ``export_sequence`` is enabled,
            or if required upstream outputs are missing or misconfigured.
        ValueError
            If ``video_fmt`` or ``sequence_fmt`` is not a supported format.
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

        gif_paths: List[str] = []
        video_paths: List[str] = []
        sequence_folders: List[str] = []
        per_track_manifest: List[Dict[str, Any]] = []
        n_skipped_tracks = 0

        for trk in ext_out.get("per_track", []):
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
                "n_tracks_written": len(per_track_manifest),
                "n_tracks_skipped": n_skipped_tracks,
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
