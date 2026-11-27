"""
Core logic for CLI actions in playNano. What the ``playnano`` subcommands do.

``process``, ``analyze``, ``play``, ``wizard`` and ``env-info``.

Called by :mod:`playnano.cli.handlers`; the functions can also be called
from Python.
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from playnano.afm_stack import AFMImageStack
from playnano.analysis.pipeline import AnalysisPipeline
from playnano.analysis.utils.common import export_to_hdf5, make_json_safe
from playnano.analysis.utils.loader import load_analysis_module
from playnano.cli.utils import (
    _sanitize_for_dump,
    ask_for_analysis_params,
    ask_for_processing_params,
    get_processing_step_type,
    is_valid_analysis_step,
    is_valid_step,
    parse_analysis_file,
    parse_analysis_string,
    parse_processing_file,
    parse_processing_string,
)
from playnano.errors import LoadError
from playnano.gui.main import gui_entry
from playnano.io.export_data import export_bundles
from playnano.io.gif_export import export_gif
from playnano.io.image_sequence_export import export_image_sequence
from playnano.io.video_export import export_video
from playnano.processing.core import process_stack
from playnano.utils.colormaps import DEFAULT_CMAP, resolve_cmap
from playnano.utils.io_utils import (
    ensure_not_input_path,
    prepare_output_directory,
    sanitize_output_name,
)
from playnano.utils.param_utils import prune_kwargs

logger = logging.getLogger(__name__)


def process_pipeline_mode(
    input_file: str,
    channel: str,
    processing_str: str | None,
    processing_file: str | None,
    export: str | None,
    make_gif: bool,
    make_video: str | None,
    make_sequence: bool,
    output_folder: str | None,
    output_name: str | None,
    scale_bar_nm: int | None,
    zmin: str = "auto",
    zmax: str = "auto",
    draw_ts: bool = False,
    cmap: str = DEFAULT_CMAP,
    fps: float = 5.0,
) -> None:
    """
    Process a stack, then export it as files, a GIF, videos and/or images.

    This is ``playnano process``. Output files go in ``output_folder`` and are
    named from ``output_name``; see :func:`playnano.io.export_data.export_bundles`,
    :func:`playnano.io.gif_export.export_gif`,
    :func:`playnano.io.video_export.export_video` and
    :func:`playnano.io.image_sequence_export.export_image_sequence`.

    Parameters
    ----------
    input_file : str
        AFM file or folder to load.
    channel : str
        Channel to load, for example ``"height_trace"``.
    processing_str : str or None
        Processing steps as text, for example
        ``"remove_plane;gaussian_filter:sigma=2"`` (see
        :func:`playnano.cli.utils.parse_processing_string`).
    processing_file : str or None
        YAML or JSON processing file (see
        :func:`playnano.cli.utils.parse_processing_file`). Used instead of
        ``processing_str`` if both are given. If neither is given, the data is
        exported unprocessed.
    export : str or None
        Comma-separated data formats: ``"tif"``, ``"npz"`` and/or ``"h5"``.
        Surrounding quotes are removed. None exports no data files.
    make_gif : bool
        Write an animated GIF.
    make_video : str or None
        Comma-separated video formats, for example ``"mp4"`` or ``"mp4,avi"``.
        None writes no video.
    make_sequence : bool
        Write a folder of PNG images, one per frame.
    output_folder : str or None
        Folder for all outputs; created if missing. If None, ``./output``.
    output_name : str or None
        Start of the output file names. If None, the input file's name.
    scale_bar_nm : int or None
        Scale bar length on the GIF, videos and images, in nm; 0 hides it. If
        None, 100.
    zmin, zmax : float or str, optional
        Colour scale limits for the GIF, videos and images: a number,
        ``"auto"`` (1st and 99th percentiles of the stack) or None (each frame
        scaled separately). Default is ``"auto"``.
    draw_ts : bool, optional
        Draw each frame's time on the GIF, videos and images. Default is False.
    cmap : str, optional
        Colormap name; an unknown name gives a warning and ``"afm_brown"``.
        Default is ``"afm_brown"``.
    fps : float, optional
        Frames per second of the GIF and videos. Default is 5.0.

    Raises
    ------
    SystemExit
        With status 1, if loading or processing fails (the error is logged).
    ValueError
        If ``export`` names an unsupported format, or an output name or folder
        is invalid.
    """
    logger.debug("Entering process_pipeline_mode: %r", locals())

    cmap = resolve_cmap(cmap)  # validate or apply fallback

    # 1) Build steps_with_kwargs for processing
    if processing_file:
        steps_with_kwargs = parse_processing_file(processing_file)
    elif processing_str:
        steps_with_kwargs = parse_processing_string(processing_str)
    else:
        steps_with_kwargs = []

    # 2) Process stack with the steps
    try:
        afm_stack = process_stack(Path(input_file), channel, steps_with_kwargs)
    except LoadError as e:
        # The message already says why (for example, the channels available);
        # a traceback would only hide it.
        logger.error("%s", e)
        sys.exit(1)
    except Exception as e:
        # Unexpected errors from processing: keep the traceback.
        logger.exception("Failed to process AFM stack: %s", e)
        sys.exit(1)

    # 3) Exports
    if export:
        export_raw = export
        if (export_raw.startswith("'") and export_raw.endswith("'")) or (
            export_raw.startswith('"') and export_raw.endswith('"')
        ):
            export_raw = export_raw[1:-1]
        formats = [tok.strip() for tok in export_raw.split(",") if tok.strip()]
        export_bundles(afm_stack, output_folder, output_name, formats)

    # 4) GIF
    export_gif(
        afm_stack=afm_stack,
        make_gif=make_gif,
        output_folder=output_folder,
        output_name=output_name,
        scale_bar_nm=scale_bar_nm,
        raw=False,
        zmin=zmin,
        zmax=zmax,
        draw_ts=draw_ts,
        cmap_name=cmap,
        fps=fps,
    )

    # 5) Video Export
    if make_video is not None:
        for fmt in make_video.split(","):
            fmt = fmt.strip()
            export_video(
                afm_stack=afm_stack,
                make_video=True,
                output_folder=output_folder,
                output_name=output_name,
                scale_bar_nm=scale_bar_nm,
                fmt=fmt,
                raw=False,
                zmin=zmin,
                zmax=zmax,
                draw_ts=draw_ts,
                cmap_name=cmap,
                fps=fps,
            )
    # 6) Image Sequence Export
    export_image_sequence(
        afm_stack=afm_stack,
        make_sequence=make_sequence,
        output_folder=output_folder,
        output_name=output_name,
        scale_bar_nm=scale_bar_nm,
        raw=False,
        zmin=zmin,
        zmax=zmax,
        draw_ts=draw_ts,
        cmap_name=cmap,
    )


def warn_if_unprocessed(stack: AFMImageStack) -> None:
    """
    Log a warning if a stack has not been through a processing pipeline.

    Checks for a ``"raw"`` entry in ``stack.processed``, which processing adds.

    Parameters
    ----------
    stack : AFMImageStack
        The stack to check.
    """
    processed = getattr(stack, "processed", None)
    if not (isinstance(processed, dict) and "raw" in processed):
        logger.warning(
            "This AFMImageStack has not been run through a playNano processing "
            "pipeline yet. No `.processed` dictionary (with a 'raw' key) was found. "
            "Ensure this data is appropriately processed for analysis. "
        )


def analyze_pipeline_mode(
    input_file: str,
    channel: str,
    analysis_str: str | None,
    analysis_file: str | None,
    output_folder: str | None,
    output_name: str | None,
) -> None:
    """
    Analyse a stack and save the results as JSON and HDF5.

    This is ``playnano analyze``. The stack is loaded as it is, so load a file
    saved after processing (or one that needs none); a warning is logged if it
    has not been processed.

    The JSON file holds the full analysis record, with arrays as lists; the
    HDF5 file holds the same record (see
    :func:`playnano.analysis.utils.common.export_to_hdf5`). Both are written
    even if one fails.

    Parameters
    ----------
    input_file : str
        File or folder to load.
    channel : str
        Channel to load, for example ``"height_trace"``.
    analysis_str : str or None
        Analysis steps as text, for example
        ``"feature_detection:threshold=5;particle_tracking"`` (see
        :func:`playnano.cli.utils.parse_analysis_string`).
    analysis_file : str or None
        YAML or JSON analysis file (see
        :func:`playnano.cli.utils.parse_analysis_file`). Used instead of
        ``analysis_str`` if both are given.
    output_folder : str or None
        Folder for the two files; created if missing. If None, ``./output``.
    output_name : str or None
        Name of the two files, ``<name>.json`` and ``<name>.h5``. Any folder
        part and a known extension are removed. If None, the input file's name.

    Raises
    ------
    ValueError
        If an output file would overwrite ``input_file``, or ``output_name``
        contains invalid characters. Both are checked before the data is loaded.
    RuntimeError
        If the JSON or HDF5 file could not be written.
    """
    # 0) decide output paths first, so nothing runs if they would overwrite the input
    out_dir = prepare_output_directory(output_folder, default="output")
    base_name = sanitize_output_name(output_name, Path(input_file).stem)
    json_path = out_dir / f"{base_name}.json"
    h5_path = out_dir / f"{base_name}.h5"
    for out_path in (json_path, h5_path):
        ensure_not_input_path(out_path, input_file)

    # 1) load data
    stack = AFMImageStack.load_data(input_file, channel=channel)
    warn_if_unprocessed(stack)

    # 2) parse steps
    if analysis_file:
        steps = parse_analysis_file(analysis_file)
    elif analysis_str:
        steps = parse_analysis_string(analysis_str)
    else:
        steps = []

    # 3) build & run pipeline
    pipeline = AnalysisPipeline()
    for name, kwargs in steps:
        pipeline.add(name, **kwargs)
    raw_record = pipeline.run(stack, log_to=None)

    # 4) write JSON and 5) HDF5. Try both, so one failure does not lose the
    # other file, then report every failure.
    failed: list[Path] = []
    logger.debug("Writing analysis JSON to %s", json_path)
    try:
        safe_record = make_json_safe(raw_record)
        with json_path.open("w") as jf:
            json.dump(safe_record, jf, indent=2)
        logger.info("Wrote analysis JSON to %s", json_path)
    except Exception as e:
        logger.error("Failed to write analysis JSON to %s: %s", json_path, e)
        failed.append(json_path)

    logger.debug("Writing analysis HDF5 to %s", h5_path)
    try:
        export_to_hdf5(raw_record, out_path=h5_path)
        logger.info("Wrote analysis HDF5 to %s", h5_path)
    except Exception as e:
        logger.error("Failed to write analysis HDF5 to %s: %s", h5_path, e)
        failed.append(h5_path)

    if failed:
        raise RuntimeError(
            "Analysis ran, but these files could not be written: "
            + ", ".join(str(p) for p in failed)
        )


#: Playback rate used when a stack has no usable frame timing, or when its
#: real-time rate is below ``MIN_REALTIME_FPS``.
DEFAULT_PLAYBACK_FPS = 5.0

#: Slowest real-time rate that is played back as recorded. Slower series (e.g.
#: slow-scan AFM with seconds or minutes per frame) play at
#: ``DEFAULT_PLAYBACK_FPS`` instead, which is watchable.
MIN_REALTIME_FPS = 1.0


def _measured_fps(afm_stack) -> float | None:
    """
    Work out the real-time frame rate of a stack from its frame metadata.

    Sources are tried in this order:

    1. The median interval between frame timestamps, when every frame has one.
    2. The median ``frame_duration_s``.
    3. ``line_rate`` divided by the number of rows. This ignores bidirectional
       (interlaced) scanning, so it is only a fallback.

    Parameters
    ----------
    afm_stack : AFMImageStack
        The loaded stack.

    Returns
    -------
    float or None
        Frames per second, or None if the metadata has no usable timing.
    """
    frame_metadata = getattr(afm_stack, "frame_metadata", None)
    if not isinstance(frame_metadata, (list, tuple)):
        frame_metadata = []
    frames = [m for m in frame_metadata if isinstance(m, dict)]

    timestamps = [m.get("timestamp") for m in frames]
    if len(timestamps) >= 2 and all(t is not None for t in timestamps):
        intervals = np.diff(np.asarray(timestamps, dtype=float))
        intervals = intervals[np.isfinite(intervals) & (intervals > 0)]
        if intervals.size:
            fps = 1.0 / float(np.median(intervals))
            logger.debug(f"Computed fps from frame timestamps: {fps:.3g}")
            return fps

    durations = [m.get("frame_duration_s") for m in frames]
    durations = [
        float(d) for d in durations if d is not None and np.isfinite(d) and d > 0
    ]
    if durations:
        fps = 1.0 / float(np.median(durations))
        logger.debug(f"Computed fps from frame_duration_s: {fps:.3g}")
        return fps

    line_rate = frames[0].get("line_rate") if frames else None
    if line_rate:
        fps = line_rate / afm_stack.image_shape[0]
        logger.debug(
            f"Computed fps from line_rate: {fps:.3g} (line_rate={line_rate}, image_shape={afm_stack.image_shape})"  # noqa
        )
        return fps

    return None


def _default_playback_fps(afm_stack) -> float:
    """
    Choose the default playback rate for a stack.

    The stack plays in real time (see :func:`_measured_fps`) when it was
    recorded at ``MIN_REALTIME_FPS`` or faster. Slower stacks, and stacks with
    no usable frame timing, play at ``DEFAULT_PLAYBACK_FPS``.

    Parameters
    ----------
    afm_stack : AFMImageStack
        The loaded stack.

    Returns
    -------
    float
        Frames per second.
    """
    fps = _measured_fps(afm_stack)
    if fps is None:
        logger.warning(
            "No frame timing in metadata; playing at %.1f fps.", DEFAULT_PLAYBACK_FPS
        )
        return DEFAULT_PLAYBACK_FPS
    if fps < MIN_REALTIME_FPS:
        logger.info(
            "Recorded at %.3g fps, slower than %.1f fps; playing at %.1f fps.",
            fps,
            MIN_REALTIME_FPS,
            DEFAULT_PLAYBACK_FPS,
        )
        return DEFAULT_PLAYBACK_FPS
    return fps


def play_pipeline_mode(
    input_file: str,
    channel: str,
    processing_str: str | None,
    processing_file: str | None,
    output_folder: str | None,
    output_name: str | None,
    scale_bar_nm: int | None,
    fps: float | None,
    zmin: str = "auto",
    zmax: str = "auto",
    cmap: str = DEFAULT_CMAP,
) -> None:
    """
    Open a stack in the playNano GUI.

    This is ``playnano play``. The processing steps are not applied when the
    window opens: press F in the window to apply them (or, if none were given,
    a default set of steps). The window can also export a GIF (G) and data
    files (E).

    Parameters
    ----------
    input_file : str
        File or folder to load.
    channel : str
        Channel to load, for example ``"height_trace"``.
    processing_str : str or None
        Processing steps as text, as for :func:`process_pipeline_mode`.
    processing_file : str or None
        YAML or JSON processing file. Used instead of ``processing_str`` if both
        are given.
    output_folder : str or None
        Folder for exports from the window.
    output_name : str or None
        Start of the exported file names.
    scale_bar_nm : int or None
        Scale bar length, in nm; 0 hides it. If None, 100.
    fps : float or None
        Playback frame rate. If None, the rate the data was recorded at, or
        5 frames per second if that is unknown or slower than 1 frame per
        second.
    zmin, zmax : float or str, optional
        Colour scale limits: a number, or ``"auto"`` (1st and 99th
        percentiles). A value that is neither is logged as an error and passed
        on unchanged. Default is ``"auto"``.
    cmap : str, optional
        Colormap the window opens with. Default is ``"afm_brown"``.

    Raises
    ------
    LoadError
        If the file cannot be loaded.
    """
    try:
        afm_stack = AFMImageStack.load_data(input_file, channel=channel)
    except Exception as e:
        # Not logged here: the caller reports the LoadError, which keeps the reason.
        raise LoadError(f"Failed to load {input_file}: {e}") from e

    if fps is None:
        fps = _default_playback_fps(afm_stack)
    else:
        logger.debug(f"Overriding data frame rate. FPS = {fps:.2f}.")

    if processing_file:
        steps_with_kwargs = parse_processing_file(processing_file)
    elif processing_str:
        steps_with_kwargs = parse_processing_string(processing_str)
    else:
        steps_with_kwargs = []

    if zmin != "auto":
        try:
            zmin = float(zmin)
        except (TypeError, ValueError):
            logger.error(
                "The value of zmin must be either a number or the string 'auto'."
            )

    if zmax != "auto":
        try:
            zmax = float(zmax)
        except (TypeError, ValueError):
            logger.error(
                "The value of zmax must be either a number or the string 'auto'."
            )

    gui_entry(
        afm_stack,
        output_dir=output_folder,
        output_name=output_name,
        steps_with_kwargs=steps_with_kwargs,
        scale_bar_nm=100 if scale_bar_nm is None else scale_bar_nm,
        zmin=zmin,
        zmax=zmax,
        cmap=cmap,
        fps=fps,
    )


class Wizard:
    """
    Interactive prompt for building and running processing and analysis.

    This is ``playnano wizard``. The stack is loaded when the wizard is made;
    :meth:`run` then reads commands until ``quit``. Type ``help`` for the list.

    Processing commands:

    - ``add <step>``: add a processing step, asking for its parameters;
    - ``clear``: add a step that removes the current mask;
    - ``remove <n>``, ``move <from> <to>``, ``list``: edit and show the steps;
    - ``save <file>``, ``load <file>``: write or read a YAML processing file;
    - ``run``: process the stack, then offer to export data files, a GIF and a
      video.

    Analysis commands:

    - ``aadd <module>`` (asks for parameters) or
      ``aadd <module>:key=value,...``: add an analysis step;
    - ``aremove <n>``, ``amove <from> <to>``, ``alist``: edit and show the steps;
    - ``asave <file>``, ``aload <file>``: write or read an analysis file;
    - ``arun``: run the processing (if there are steps and they changed since
      the last run) and then the analysis, and save the results as
      ``<name>.json`` and ``<name>.h5``.

    Step numbers start at 1. ``help`` lists most commands; ``load`` and
    ``clear`` are missing from it.

    Parameters
    ----------
    input_file : str
        File or folder to load.
    channel : str
        Channel to load, for example ``"height_trace"``.
    output_folder : str or None
        Folder for exports and analysis results. If None, ``./output``.
    output_name : str or None
        Start of the output file names. If None, the input file's name.
    scale_bar_nm : int or None
        Scale bar length used when a length typed at the ``run`` prompts is not
        a whole number. If None, 100.
    io : IO, optional
        Reads commands and prints messages. Default is None (the terminal).
        Mainly for tests.
    cmap : str, optional
        Colormap for GIF and video exports. Default is ``"afm_brown"``.

    Attributes
    ----------
    afm_stack : AFMImageStack
        The stack as loaded.
    input_path : pathlib.Path
        ``input_file`` as a path.
    process_steps : list of tuple
        Processing steps, as ``(name, parameters)``.
    analysis_steps : list of tuple
        Analysis steps, as ``(name, parameters)``.
    processed_stack : AFMImageStack or None
        Result of the last processing run, reused by ``arun`` while the steps
        are unchanged.
    processed_steps_snapshot : list of tuple
        The processing steps ``processed_stack`` was made with.

    Raises
    ------
    FileNotFoundError
        If ``input_file`` does not exist.
    LoadError
        If it cannot be loaded.

    Examples
    --------
    >>> wizard = Wizard(
    ...     "sample.h5-jpk", "height_trace", None, None, None
    ... )  # doctest: +SKIP
    >>> wizard.run()  # doctest: +SKIP
    """

    def __init__(
        self,
        input_file: str,
        channel: str,
        output_folder: Optional[str],
        output_name: Optional[str],
        scale_bar_nm: Optional[int],
        io: Optional[IO] = None,
        cmap: str = DEFAULT_CMAP,
    ) -> None:
        """Load the stack. The parameters are described in the class docstring."""
        self.input_file = input_file
        self.input_path = Path(input_file)
        self.channel = channel
        self.output_folder = output_folder
        self.output_name = output_name
        self.scale_bar_nm = scale_bar_nm
        self.cmap = cmap
        self.io = io or IO()

        if not self.input_path.exists():
            raise FileNotFoundError(f"File not found: {self.input_path}")

        try:
            self.afm_stack = AFMImageStack.load_data(self.input_path, channel=channel)
        except Exception as e:
            raise LoadError(f"Failed to load {input_file}: {e}") from e

        # wizard state
        self.process_steps: List[Tuple[str, Dict[str, Any]]] = []
        self.analysis_steps: List[Tuple[str, Dict[str, Any]]] = []
        self.processed_stack = None
        self.processed_steps_snapshot: List[Tuple[str, Dict[str, Any]]] = []

    # -------------------------
    # Utilities / UX
    # -------------------------
    def print_help(self) -> None:
        """Print the wizard commands."""
        help_text = (
            "Commands:\n"
            "  add <filter_name>   - Add processing step\n"
            "  remove <index>      - Remove processing step\n"
            "  move <old> <new>    - Reorder processing steps\n"
            "  list                - List processing steps\n"
            "  save <path>         - Save processing to YAML file\n"
            "  run                 - Execute processing now (returns to prompt)\n"
            "  aadd <spec|name>    - Add analysis step inline or interactive (aadd name)\n"  # noqa
            "  aremove <index>     - Remove analysis step\n"
            "  amove <old> <new>   - Reorder analysis steps\n"
            "  alist               - List analysis steps\n"
            "  asave <path>        - Save analysis to YAML/JSON\n"
            "  aload <path>        - Load analysis file YAML/JSON\n"
            "  arun                - Execute analysis now (runs processing first if needed)\n"  # noqa
            "  help                - Show this message\n"
            "  quit                - Exit without running\n"
        )
        self.io.say(help_text)

    def list_steps(self) -> None:
        """Print the processing steps, numbered from 1."""
        if not self.process_steps:
            self.io.say("  [no processing steps]")
            return
        for i, (name, kw) in enumerate(self.process_steps, start=1):
            if kw:
                params = ", ".join(f"{k}={v}" for k, v in kw.items())
                self.io.say(f"  {i}) {name} ({params})")
            else:
                self.io.say(f"  {i}) {name}")
        self.io.say("")

    def list_analysis(self) -> None:
        """Print the analysis steps, numbered from 1."""
        if not self.analysis_steps:
            self.io.say("  [no analysis steps]")
            return
        for i, (name, kw) in enumerate(self.analysis_steps, start=1):
            if kw:
                params = ", ".join(f"{k}={v}" for k, v in kw.items())
                self.io.say(f"  {i}) {name} ({params})")
            else:
                self.io.say(f"  {i}) {name}")
        self.io.say("")

    # -------------------------
    # Processing handlers
    # -------------------------
    def prompt_for_processing_params(self, step_name: str) -> Dict[str, Any]:
        """
        Ask for a few common parameters of a processing step.

        Used by ``add`` when the step's parameters cannot be read from its
        function. Only these are asked for: ``sigma`` (``gaussian_filter``),
        ``order`` (``polynomial_flatten``), ``factor`` (``mask_mean_offset``) and
        ``threshold`` (``mask_threshold``, ``mask_below_threshold``). Pressing
        Enter uses the default shown; an invalid value is asked for again.

        Parameters
        ----------
        step_name : str
            Name of the processing step.

        Returns
        -------
        dict
            Parameter names and values; empty for other steps.
        """
        params_to_ask = []
        if step_name == "gaussian_filter":
            params_to_ask = [("sigma", float, 1.0)]
        elif step_name == "polynomial_flatten":
            params_to_ask = [("order", int, 2)]
        elif step_name in ("mask_mean_offset",):
            params_to_ask = [("factor", float, 1.0)]
        elif step_name in ("mask_threshold", "mask_below_threshold"):
            params_to_ask = [("threshold", float, 1.0)]

        kwargs: Dict[str, Any] = {}
        for param_name, param_type, default in params_to_ask:
            while True:
                val_str = self.io.ask(
                    f"  Enter {param_name} (default={default}): "
                ).strip()
                if val_str == "":
                    kwargs[param_name] = default
                    break
                try:
                    if param_type is int:
                        val = int(val_str)
                    elif param_type is float:
                        val = float(val_str)
                    else:
                        val = val_str
                    kwargs[param_name] = val
                    break
                except ValueError:
                    self.io.say(
                        f"  Invalid {param_name}! Expecting a {param_type.__name__}. Try again."  # noqa
                    )
        return kwargs

    def handle_add(self, parts: List[str]) -> None:
        """
        ``add <step>``: add a processing step, asking for each parameter.

        Parameters
        ----------
        parts : list of str
            The command line typed at the prompt, split on spaces; ``parts[0]`` is
            the command.
        """
        if len(parts) < 2:
            self.io.say("Usage: add <processing_step>")
            return
        step_name = parts[1]
        if not is_valid_step(step_name):
            self.io.say(f"Unknown processing step: '{step_name}'")
            return
        step_type = get_processing_step_type(step_name)
        self.io.say(f"Adding {step_type}: {step_name}")
        # prefer the richer introspection but fallback to built-in prompts
        try:
            kwargs = ask_for_processing_params(step_name)
            if kwargs is None:
                kwargs = self.prompt_for_processing_params(step_name)
        except Exception:
            kwargs = self.prompt_for_processing_params(step_name)
        self.process_steps.append((step_name, kwargs))
        self.io.say(f"Added {step_type}: {step_name} {kwargs}")

    def handle_clear(self, parts: List[str]) -> None:
        """
        ``clear``: add a step that removes the current mask.

        Parameters
        ----------
        parts : list of str
            The command line typed at the prompt, split on spaces; ``parts[0]`` is
            the command.
        """
        self.process_steps.append(("clear", {}))
        self.io.say("Added clear-mask step.")

    def handle_remove(self, parts: List[str]) -> None:
        """
        ``remove <n>``: remove processing step ``n`` (from 1).

        Parameters
        ----------
        parts : list of str
            The command line typed at the prompt, split on spaces; ``parts[0]`` is
            the command.
        """
        if len(self.process_steps) == 0:
            self.io.say("No processing steps to remove.")
            return
        if len(parts) != 2 or not parts[1].isdigit():
            self.io.say("Usage: remove <index>")
            return
        idx = int(parts[1])
        if idx < 1 or idx > len(self.process_steps):
            self.io.say(f"Index out of range (1-{len(self.process_steps)}).")
            return
        removed = self.process_steps.pop(idx - 1)
        self.io.say(f"Removed step {idx}: {removed[0]}")

    def handle_move(self, parts: List[str]) -> None:
        """
        ``move <from> <to>``: move a processing step to a new position (from 1).

        Parameters
        ----------
        parts : list of str
            The command line typed at the prompt, split on spaces; ``parts[0]`` is
            the command.
        """
        if len(parts) != 3 or not parts[1].isdigit() or not parts[2].isdigit():
            self.io.say("Usage: move <old_index> <new_index>")
            return
        old_i = int(parts[1]) - 1
        new_i = int(parts[2]) - 1
        if (
            old_i < 0
            or old_i >= len(self.process_steps)
            or new_i < 0
            or new_i > len(self.process_steps)
        ):
            self.io.say("Indices out of range.")
            return
        item = self.process_steps.pop(old_i)
        self.process_steps.insert(new_i, item)
        self.io.say(f"Moved step from position {old_i+1} to {new_i+1}.")

    def handle_save(self, parts: list[str]) -> None:
        """
        ``save <file>``: write the processing steps to a YAML file.

        Parameters that are None, or that do not apply (see
        :func:`playnano.utils.param_utils.prune_kwargs`), are left out. Values are
        converted to plain YAML types (tuples to lists, NumPy numbers to Python
        numbers, paths to text), so ``load`` and ``--processing-file`` can read the
        file. Missing folders are created.

        Parameters
        ----------
        parts : list of str
            The command line typed at the prompt, split on spaces; ``parts[0]`` is
            the command.
        """
        if len(parts) != 2:
            self.io.say("Usage: save <path/to/output.yaml>")
            return

        save_path = Path(parts[1])
        processing_dict = {"filters": []}

        for name, kw in self.process_steps:
            entry = {"name": name}

            # Get the registered processing function (if available)
            try:
                name, func = self.afm_stack._resolve_step(name)
            except Exception:
                func = None

            # Apply param pruning (removes inactive or None params)
            if func is not None:
                filtered = prune_kwargs(func, kw)
            else:
                # fallback: simple None-filtering
                filtered = {k: v for k, v in kw.items() if v is not None}

            entry.update(filtered)
            processing_dict["filters"].append(entry)

        try:
            safe_dict = _sanitize_for_dump(processing_dict)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w", encoding="utf8") as f:
                yaml.safe_dump(safe_dict, f, sort_keys=False)
            self.io.say(f"Processing saved to {save_path}")
        except Exception as e:
            self.io.say(f"Error saving processing: {e}")

    def handle_load(self, parts: List[str]) -> None:
        """
        ``load <file>``: replace the processing steps with those in a YAML file.

        Reads a ``filters:`` list as written by ``save``; entries without a
        ``name`` are skipped.

        Parameters
        ----------
        parts : list of str
            The command line typed at the prompt, split on spaces; ``parts[0]`` is
            the command.
        """
        if len(parts) != 2:
            self.io.say("Usage: load <path/to/processing.yaml>")
            return

        path = Path(parts[1])
        if not path.exists():
            self.io.say(f"File not found: {path}")
            return
        try:
            with path.open("r", encoding="utf8") as f:
                data = yaml.safe_load(f)

            if not isinstance(data, dict):
                self.io.say(
                    "Invalid processing file: expected a mapping with a 'filters' key."
                )
                return
            if "filters" not in data or not isinstance(data["filters"], list):
                self.io.say("Invalid processing file format: missing 'filters' list.")
                return

            new_steps: List[Tuple[str, Dict[str, Any]]] = []
            for entry in data["filters"]:
                if not isinstance(entry, dict):
                    continue
                name = entry.get("name")
                if not name:
                    continue
                # create kwargs without mutating original entry
                kwargs = {k: v for k, v in entry.items() if k != "name"}
                new_steps.append((name, kwargs))
            self.process_steps = new_steps
            self.processed_stack = None
            self.processed_steps_snapshot = []
            self.io.say(
                f"Loaded {len(self.process_steps)} processing steps from {path}"
            )
        except Exception as e:
            self.io.say(f"Error loading processing file: {e}")

    def run_processing_and_cache(self):
        """
        Load the file again, process it with the current steps, and keep the result.

        Returns
        -------
        AFMImageStack
            The processed stack, also stored in :attr:`processed_stack`.

        Raises
        ------
        LoadError
            If loading or processing fails.
        """
        try:
            processed = process_stack(
                Path(self.input_file), self.channel, self.process_steps
            )
        except Exception as e:
            # wrap with LoadError to be consistent with __init__ behaviour
            raise LoadError(f"Failed to process stack: {e}") from e
        self.processed_stack = processed
        self.processed_steps_snapshot = list(self.process_steps)
        return self.processed_stack

    def _parse_scale_value(self, val_str: str) -> int | float | str:
        """Parse the z scale values for Gif Export."""
        v = val_str.strip().lower()
        if v in ("auto", ""):
            return "auto"
        try:
            if "." in v:
                return float(v)
            return int(v)
        except ValueError:
            return v  # fall back to raw; export_gif should validate

    def handle_run(self, parts: List[str]) -> None:
        """
        ``run``: process the stack, then offer to export it.

        Asks in turn whether to export data files (and in which formats), a GIF and
        a video, and for each image export the colour scale limits, timestamps and
        scale bar. The processed stack is kept for ``arun``.

        Parameters
        ----------
        parts : list of str
            The command line typed at the prompt, split on spaces; ``parts[0]`` is
            the command.
        """
        if not self.process_steps:
            self.io.say("No steps to run. Use `add <filter_name>` first.")
            return
        self.io.say("Executing processing…")
        try:
            afm_stack_local = self.run_processing_and_cache()
        except LoadError as e:
            self.io.say(f"Error: {e}")
            return
        self.io.say("Processing execution complete.")
        export_choice = self.io.ask("Export results? (y/n): ").strip().lower()
        if export_choice in ("y", "yes"):
            fmt_str = self.io.ask(
                "Enter formats (comma-separated, e.g. tif,npz,h5): "
            ).strip()
            formats = [fmt.strip().lower() for fmt in fmt_str.split(",") if fmt.strip()]
            try:
                export_bundles(
                    afm_stack_local, self.output_folder, self.output_name, formats
                )
            except ValueError as e:
                self.io.say(f"Export failed: {e}")
        gif_choice = self.io.ask("Create a GIF? (y/n): ").strip().lower()
        if gif_choice in ("y", "yes"):
            zmin_raw = (
                self.io.ask("Enter a minimum value for the Z scale (or 'auto'): ")
                .strip()
                .lower()
            )
            zmax_raw = (
                self.io.ask("Enter a maximum value for the Z scale (or 'auto'): ")
                .strip()
                .lower()
            )
            draw_ts_choice = self.io.ask("Add timestamps? (y/n): ").strip().lower() in (
                "y",
                "yes",
            )
            scale_bar_choice = self.io.ask("Add a scale bar? (y/n): ").strip().lower()
            if scale_bar_choice in ("y", "yes"):
                gif_scale_bar_length_raw = self.io.ask(
                    "Enter scale bar length in nm: "
                ).strip()
                try:
                    gif_scale_bar_nm = int(gif_scale_bar_length_raw)
                except ValueError:
                    self.io.say(
                        f"Invalid scale bar length '{gif_scale_bar_length_raw}' (must be an integer).; "  # noqa: E501
                        "using session default."
                    )
                    gif_scale_bar_nm = self.scale_bar_nm
            else:
                gif_scale_bar_nm = 0

            zmin_choice = self._parse_scale_value(zmin_raw)
            zmax_choice = self._parse_scale_value(zmax_raw)
            export_gif(
                afm_stack_local,
                True,
                self.output_folder,
                self.output_name,
                gif_scale_bar_nm,
                zmin=zmin_choice,
                zmax=zmax_choice,
                cmap_name=self.cmap,
                draw_ts=draw_ts_choice,
            )
        export_video_choice = self.io.ask("Create a video? (y/n): ").strip().lower()
        if export_video_choice in ("y", "yes"):
            zmin_raw = (
                self.io.ask("Enter a minimum value for the Z scale (or 'auto'): ")
                .strip()
                .lower()
            )
            zmax_raw = (
                self.io.ask("Enter a maximum value for the Z scale (or 'auto'): ")
                .strip()
                .lower()
            )
            ts_choice = self.io.ask("Add timestamps? (y/n): ").strip().lower() in (
                "y",
                "yes",
            )
            scale_bar_choice = self.io.ask("Add a scale bar? (y/n): ").strip().lower()
            if scale_bar_choice in ("y", "yes"):
                scale_bar_length_raw = self.io.ask(
                    "Enter scale bar length in nm: "
                ).strip()
                try:
                    video_scale_bar_nm = int(scale_bar_length_raw)
                except ValueError:
                    self.io.say(
                        f"Invalid scale bar length '{scale_bar_length_raw}' (must be an integer); "  # noqa: E501
                        "using session default."
                    )
                    video_scale_bar_nm = self.scale_bar_nm
            else:
                video_scale_bar_nm = 0
            zmin_choice = self._parse_scale_value(zmin_raw)
            zmax_choice = self._parse_scale_value(zmax_raw)
            export_video(
                afm_stack_local,
                True,
                self.output_folder,
                self.output_name,
                video_scale_bar_nm,
                zmin=zmin_choice,
                zmax=zmax_choice,
                cmap_name=self.cmap,
                draw_ts=ts_choice,
            )
        self.io.say("Processing complete; processed stack cached.")

    # -------------------------
    # Analysis handlers
    # -------------------------
    def handle_alist(self, parts: List[str]) -> None:
        """
        ``alist``: print the analysis steps.

        Parameters
        ----------
        parts : list of str
            The command line typed at the prompt, split on spaces; ``parts[0]`` is
            the command.
        """
        self.io.say("Analysis steps:")
        self.list_analysis()

    def handle_aadd(self, parts: List[str]) -> None:
        """
        ``aadd``: add an analysis step.

        ``aadd <module>`` asks for each parameter;
        ``aadd <module>:key=value,...`` gives them directly, read as by
        :func:`playnano.cli.utils.parse_analysis_string` (several steps can be
        given, separated by ``;``).

        Parameters
        ----------
        parts : list of str
            The command line typed at the prompt, split on spaces; ``parts[0]`` is
            the command.
        """
        if len(parts) < 2:
            self.io.say(
                "Usage: aadd <step[:param=val,...]> or `aadd <step>` for interactive."
            )
            return
        spec = parts[1]
        # Inline spec case
        if ":" in spec or "=" in spec:
            try:
                new = parse_analysis_string(spec)
                if not isinstance(new, list):
                    raise ValueError(
                        "Parsed spec must be a list of (name, kwargs) tuples"
                    )
            except Exception as e:
                self.io.say(f"Invalid analysis spec: {e}")
                return
            self.analysis_steps.extend(new)
            self.io.say(f"Added analysis: {new}")
            return

        module_name = spec

        # Try introspection first (tests often patch ask_for_analysis_params).
        try:
            kwargs = ask_for_analysis_params(module_name)
            if kwargs is None:
                kwargs = {}
        except ModuleNotFoundError:
            # If module missing, validate and bail out if unknown
            try:
                if not is_valid_analysis_step(module_name):
                    self.io.say(f"Unknown analysis step: '{module_name}'")
                    return
            except Exception:
                self.io.say(f"Unknown analysis step: '{module_name}'")
                return
            kwargs = {}
        except Exception as e:
            # Introspection failed for other reasons; surface error
            self.io.say(f"Unable to introspect module '{module_name}': {e}")
            return

        self.analysis_steps.append((module_name, kwargs))
        self.io.say(f"Added: {module_name} {kwargs}")

    def handle_aremove(self, parts: List[str]) -> None:
        """
        ``aremove <n>``: remove analysis step ``n`` (from 1).

        Parameters
        ----------
        parts : list of str
            The command line typed at the prompt, split on spaces; ``parts[0]`` is
            the command.
        """
        if len(self.analysis_steps) == 0:
            self.io.say("No analysis steps to remove.")
            return
        if len(parts) != 2 or not parts[1].isdigit():
            self.io.say("Usage: aremove <index>")
            return
        idx = int(parts[1]) - 1
        if idx < 0 or idx >= len(self.analysis_steps):
            self.io.say(f"Index out of range 1-{len(self.analysis_steps)}")
            return
        removed = self.analysis_steps.pop(idx)
        self.io.say(f"Removed analysis step {idx+1}: {removed[0]}")

    def handle_amove(self, parts: List[str]) -> None:
        """
        ``amove <from> <to>``: move an analysis step to a new position (from 1).

        Parameters
        ----------
        parts : list of str
            The command line typed at the prompt, split on spaces; ``parts[0]`` is
            the command.
        """
        if len(parts) != 3 or not parts[1].isdigit() or not parts[2].isdigit():
            self.io.say("Usage: amove <old> <new>")
            return
        old, new = int(parts[1]) - 1, int(parts[2]) - 1
        if any(i < 0 or i >= len(self.analysis_steps) for i in (old, new)):
            self.io.say("Indices out of range")
            return
        item = self.analysis_steps.pop(old)
        self.analysis_steps.insert(new, item)
        self.io.say(f"Moved analysis from {old+1} to {new+1}")

    def handle_aload(self, parts: List[str]) -> None:
        """
        ``aload <file>``: replace the analysis steps with those in a YAML or JSON file.

        Accepts ``{"analysis": [...]}`` or a plain list of steps. Text values that
        are Python literals, such as ``"[1, 2]"``, are converted. Entries without a
        ``name`` are skipped.

        Parameters
        ----------
        parts : list of str
            The command line typed at the prompt, split on spaces; ``parts[0]`` is
            the command.
        """
        if len(parts) != 2:
            self.io.say("Usage: aload <path>")
            return
        path = Path(parts[1])
        if not path.exists():
            self.io.say(f"File not found: {path}")
            return
        try:
            with open(path, "r", encoding="utf8") as f:
                if path.suffix.lower() in (".yaml", ".yml"):
                    raw = yaml.safe_load(f)
                else:
                    raw = json.load(f)
            if raw is None:
                self.io.say("Empty analysis file.")
                return

            # Accept either {"analysis": [...]} or just [...]
            if isinstance(raw, dict):
                loaded = raw.get("analysis", None)
            else:
                loaded = raw

            if not isinstance(loaded, list):
                self.io.say(
                    "Invalid analysis file format: must be a list of steps or {'analysis': [...]}"  # noqa
                )
                return

            steps: List[Tuple[str, Dict[str, Any]]] = []
            import ast

            for step in loaded:
                if not isinstance(step, dict) or "name" not in step:
                    continue
                name = step["name"]
                # copy and attempt to literal_eval any stringified values
                kw = {}
                for k, v in step.items():
                    if k == "name":
                        continue
                    if isinstance(v, str):
                        try:
                            val = ast.literal_eval(v)
                            kw[k] = val
                        except Exception:
                            kw[k] = v
                    else:
                        kw[k] = v
                steps.append((name, kw))

            self.analysis_steps[:] = steps
            self.io.say(f"Loaded analysis from {path}")
        except Exception as e:
            self.io.say(f"Error loading analysis file: {e}")

    def handle_asave(self, parts: list[str]) -> None:
        """
        ``asave <file>``: write the analysis steps to a YAML file.

        Parameters that are None, or that do not apply, are left out, and values
        are converted to plain YAML types. Missing folders are created.

        Parameters
        ----------
        parts : list of str
            The command line typed at the prompt, split on spaces; ``parts[0]`` is
            the command.
        """
        if len(parts) != 2:
            self.io.say("Usage: asave <path/to/output.yaml>")
            return

        save_path = Path(parts[1])
        analysis_dict = {"analysis": []}

        for name, kw in self.analysis_steps:
            entry = {"name": name}
            try:
                module = load_analysis_module(name)
                # Analysis modules are expected to implement a `run()` method.
                func = getattr(module, "run", None)
            except Exception:
                func = None

            if func:
                filtered = prune_kwargs(func, kw)
            else:
                filtered = {k: v for k, v in kw.items() if v is not None}

            entry.update(filtered)
            analysis_dict["analysis"].append(entry)

        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            safe_dict = _sanitize_for_dump(analysis_dict)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w", encoding="utf8") as f:
                yaml.safe_dump(safe_dict, f, sort_keys=False)

            self.io.say(f"Analysis saved to {save_path}")
        except Exception as e:
            self.io.say(f"Error saving analysis: {e}")

    def _normalize_steps(
        self, steps: List[Tuple[str, Dict[str, Any]]]
    ) -> List[Tuple[str, str]]:
        """Return a stringified representation of steps for equality checks."""
        normalized = []
        for name, kwargs in steps:
            # Use json.dumps for deterministic ordering of kwargs
            try:
                kw_json = json.dumps(kwargs, sort_keys=True, default=str)
            except Exception:
                # fallback: repr
                kw_json = repr(kwargs)
            normalized.append((name, kw_json))
        return normalized

    def _steps_equal(
        self, a: List[Tuple[str, Dict[str, Any]]], b: List[Tuple[str, Dict[str, Any]]]
    ) -> bool:
        """Check deep equality of two step lists in a deterministic way."""
        return self._normalize_steps(a) == self._normalize_steps(b)

    def _ensure_processed_stack_up_to_date(self) -> None:
        """
        Ensure the cached processed stack exists and patch it if needed.

        Runs processing.
        """
        if self.processed_stack is None or not self._steps_equal(
            self.processed_steps_snapshot, self.process_steps
        ):
            self.io.say("Running processing first (to provide data for analysis)...")
            try:
                self.run_processing_and_cache()
            except LoadError:
                # propagate with a clear message; callers will handle
                raise
            except Exception as e:
                raise LoadError(f"Processing failed: {e}") from e

    def _build_pipeline(self):
        """Build an AnalysisPipeline from current analysis_steps."""
        pipeline = AnalysisPipeline()
        for name, kwargs in self.analysis_steps:
            pipeline.add(name, **kwargs)
        return pipeline

    def _atomic_write_json(self, out_path: Path, data: Any) -> None:
        """Write JSON to a temporary file in same dir and atomically replace target."""
        out_path = Path(out_path)
        out_dir = out_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        # write to a temp file in same directory to allow atomic replace
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf8", delete=False, dir=str(out_dir)
        ) as tf:
            tmp_name = Path(tf.name)
            json.dump(data, tf, indent=2)
        tmp_name.replace(out_path)

    def _atomic_write_hdf5(self, out_path: Path, raw_record: Dict[str, Any]) -> None:
        """Write HDF5 to tmp file then replace atomically (wraps export_to_hdf5)."""
        out_path = Path(out_path)
        out_dir = out_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        # use NamedTemporaryFile to create a unique tmp file in same directory
        with tempfile.NamedTemporaryFile(
            mode="wb", delete=False, dir=str(out_dir)
        ) as tf:
            tmp_path = Path(tf.name)
        try:
            # export_to_hdf5 should write to the path we provide
            export_to_hdf5(raw_record, out_path=tmp_path)
            tmp_path.replace(out_path)
        finally:
            # ensure leftover tmp is removed if something failed
            if tmp_path.exists() and tmp_path != out_path:
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

    def _run_analysis_and_export(self, stack) -> Dict[str, Any]:
        """
        Run the analysis pipeline on the provided stack and export results.

        Parameters
        ----------
        stack : Any
            The data stack to run the analysis pipeline on. This may be the raw
            AFM stack or a processed stack, depending on the context.

        Returns
        -------
        Dict[str, Any]
            The raw analysis record containing environment details, analysis
            results, and provenance information.

        Notes
        -----
        The output directory is determined by `self.output_folder`, defaulting
        to ``./output`` if not set.

        The base filename is taken from `self.output_name`, or derived from the
        stem of `self.input_file` if `output_name` is not provided. As for
        ``process``, any folder part is dropped and a known extension removed.

        Results are saved atomically to `<base_name>.json` and `<base_name>.h5`
        in the output directory.

        Raises
        ------
        ValueError
            If either output file would overwrite the input file. This is checked
            before the analysis runs.
        """
        out_dir = prepare_output_directory(self.output_folder, default="output")
        base_name = sanitize_output_name(self.output_name, Path(self.input_file).stem)
        json_path = out_dir / f"{base_name}.json"
        h5_path = out_dir / f"{base_name}.h5"
        for out_path in (json_path, h5_path):
            ensure_not_input_path(out_path, self.input_file)

        pipeline = self._build_pipeline()
        raw_record = pipeline.run(stack, log_to=None)

        safe_record = make_json_safe(raw_record)

        # atomic writes
        self._atomic_write_json(json_path, safe_record)
        self._atomic_write_hdf5(h5_path, raw_record)

        return raw_record

    def handle_arun(self, parts: List[str]) -> None:
        """
        ``arun``: run the analysis and save the results.

        If there are processing steps, the stack is processed first, unless it was
        already processed with the same steps; otherwise the analysis runs on the
        stack as loaded. Results are written to ``<name>.json`` and ``<name>.h5`` in
        the output folder, each through a temporary file so a failed write leaves
        no partial file.

        Parameters
        ----------
        parts : list of str
            The command line typed at the prompt, split on spaces; ``parts[0]`` is
            the command.
        """
        if not self.analysis_steps:
            self.io.say("No analysis steps. Use aadd first.")
            return

        try:
            # Decide stack and ensure processed stack is current if needed
            if self.process_steps:
                try:
                    self._ensure_processed_stack_up_to_date()
                except LoadError as e:
                    self.io.say(f"Processing failed: {e}")
                    return
                stack = self.processed_stack
                source = "processed stack"
            else:
                stack = self.afm_stack
                source = "loaded stack"

            # Run analysis and export results (atomic)
            self._run_analysis_and_export(stack)
            self.processed_steps_snapshot = list(
                self.process_steps
            )  # keep snapshot up-to-date
            self.io.say(f"Analysis complete (ran on {source}).")
            logger.info("Analysis successfully completed.")
        except Exception as e:
            # log full traceback for debugging while keeping CLI output friendly
            logger.exception("Analysis failed")
            self.io.say(f"Analysis failed: {e}")

    # -------------------------
    # REPL
    # -------------------------
    def run(self) -> None:
        """
        Read and carry out commands until ``quit``.

        Raises
        ------
        SystemExit
            With status 0, on ``quit``, ``exit``, Ctrl+C or end of input.
        """
        self.io.say(f"Loaded AFM stack: {self.input_path}")
        self.io.say(
            f"Channel: {self.channel}, frames={self.afm_stack.n_frames}, shape={self.afm_stack.image_shape}"  # noqa
        )
        self.io.say("Enter `help` for a list of commands.")

        while True:
            try:
                cmd = self.io.ask("playNano wizard> ").strip()
            except (EOFError, KeyboardInterrupt):
                self.io.say("Exiting wizard.")
                sys.exit(0)
            if not cmd:
                continue
            parts = cmd.split()
            verb = parts[0].lower()

            if verb in ("quit", "exit"):
                self.io.say("Exiting wizard without running.")
                sys.exit(0)
            if verb == "help":
                self.print_help()
                continue
            # processing
            if verb == "add":
                self.handle_add(parts)
                continue
            if verb == "clear":
                self.handle_clear(parts)
                continue
            if verb == "remove":
                self.handle_remove(parts)
                continue
            if verb == "move":
                self.handle_move(parts)
                continue
            if verb == "list":
                self.io.say("Processing steps:")
                self.list_steps()
                continue
            if verb == "save":
                self.handle_save(parts)
                continue
            if verb == "load":
                self.handle_load(parts)
                continue
            if verb == "run":
                self.handle_run(parts)
                continue
            # analysis
            if verb == "aadd":
                self.handle_aadd(parts)
                continue
            if verb == "alist":
                self.handle_alist(parts)
                continue
            if verb == "aremove":
                self.handle_aremove(parts)
                continue
            if verb == "amove":
                self.handle_amove(parts)
                continue
            if verb == "aload":
                self.handle_aload(parts)
                continue
            if verb == "asave":
                self.handle_asave(parts)
                continue
            if verb == "arun":
                self.handle_arun(parts)
                continue

            self.io.say(f"Unknown command: {verb}. Type help.")


def print_env_info():
    """
    Print the software environment as JSON (``playnano env-info``).

    See :func:`playnano.utils.system_info.gather_environment_info`.
    """
    import json

    from playnano.utils.system_info import gather_environment_info

    env = gather_environment_info()
    print(json.dumps(env, indent=2))


class IO:
    """Terminal input and output for :class:`Wizard`; tests replace it."""

    def ask(self, prompt: str) -> str:
        """
        Show a prompt and return the line typed.

        Parameters
        ----------
        prompt : str
            Text shown before the input.

        Returns
        -------
        str
            The line typed, without the newline.
        """
        return input(prompt)

    def say(self, msg: str) -> None:
        """
        Print a message.

        Parameters
        ----------
        msg : str
            The message.
        """
        print(msg)
