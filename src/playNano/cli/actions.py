"""Core logic for CLI actions in playNano."""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from playNano.afm_stack import AFMImageStack
from playNano.analysis.pipeline import AnalysisPipeline
from playNano.analysis.utils.common import export_to_hdf5, make_json_safe
from playNano.cli.utils import (
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
from playNano.errors import LoadError
from playNano.gui.main import gui_entry
from playNano.io.export_data import export_bundles
from playNano.io.gif_export import export_gif
from playNano.processing.core import process_stack

logger = logging.getLogger(__name__)


def process_pipeline_mode(
    input_file: str,
    channel: str,
    processing_str: str | None,
    processing_file: str | None,
    export: str | None,
    make_gif: bool,
    output_folder: str | None,
    output_name: str | None,
    scale_bar_nm: int | None,
    zmin: str = "auto",
    zmax: str = "auto",
) -> None:
    """
    Apply a processing pipeline to an AFM file, then optionally export data and GIF.

    Steps
    -----
    1. Parse processing steps from either `processing_file` (YAML/JSON)
    or `processing_str`.
    2. Run the ProcessingPipeline on the AFM stack to apply all filters.
    3. Export the processed stack to TIFF/NPZ/HDF5 formats (`export_bundles`).
    4. Generate an animated GIF of the filtered data (`export_gif`).

    Parameters
    ----------
    input_file : str
        Path to the AFM input file.
    channel : str
        Name of the data channel to extract (e.g., "height_trace").
    processing_str : str or None
        Semicolon-delimited inline pipeline string, e.g.
        `"remove_plane;gaussian_filter:sigma=2"`.
    processing_file : str or None
        Path to a YAML/JSON file defining the processing steps.
    export : str or None
        Comma-separated output formats for bundles (e.g. `"tif,npz,h5"`).
    make_gif : bool
        Whether to create an animated GIF of the filtered stack.
    output_folder : str or None
        Directory in which to write any export files.
    output_name : str or None
        Base filename (no extension) for bundles/GIF; defaults to
        the stem of `input_file`.
    scale_bar_nm : int or None
        Length (in nm) of the scale bar overlaid on each GIF frame.
    zmin : str
        Minimum Z-value for GIF color normalization (float string or `"auto"`).
    zmax : str
        Maximum Z-value for GIF color normalization (float string or `"auto"`).

    Returns
    -------
    None
    """

    logger.debug("Entering process_pipeline_mode: %r", locals())

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
    except Exception as e:
        # Catch any exception (including LoadError or
        # unexpected exceptions from AFMImageStack)
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
    )


def warn_if_unprocessed(stack: AFMImageStack) -> None:
    """
    Warn if the AFMImageStack has not been processed using the playNano pipeline.

    Emits a warning if .processed is missing or does not contain a 'raw' key.
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
    Run an analysis pipeline on an AFM stack and export both JSON and HDF5.

    Steps
    -----
    1. Load the AFMImageStack from disk using `input_file` and `channel`.
    2. Parse analysis modules from `analysis_file` or `analysis_str`.
    3. Build and execute an `AnalysisPipeline` over the stack.
    4. Sanitize the full record (`make_json_safe`) and write it to `<output>.json`.
    5. Export the raw record to HDF5 via `export_to_hdf5`.

    Parameters
    ----------
    input_file : str
        Path to the AFM input file.
    channel : str
        Name of the data channel to analyze (e.g., "height_trace").
    analysis_str : str or None
        Semicolon-delimited inline analysis string, e.g.
        `"feature_detection:threshold=5;particle_tracking"`.
    analysis_file : str or None
        Path to a YAML/JSON file defining the analysis pipeline.
    output_folder : str or None
        Directory in which to write JSON + HDF5 exports.
    output_name : str or None
        Base filename (no extension) for both `<output>.json` and `<output>.h5`;
        defaults to the stem of `input_file`.

    Returns
    -------
    None
    """
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

    # 4) write JSON
    # — determine output folder & name
    out_dir = Path(output_folder or ".")
    out_dir.mkdir(parents=True, exist_ok=True)

    base_name = output_name or Path(input_file).stem
    json_path = out_dir / f"{base_name}.json"

    # — sanitize & dump
    safe_record = make_json_safe(raw_record)
    logger.debug("Writing analysis JSON to %s", json_path)
    with json_path.open("w") as jf:
        json.dump(safe_record, jf, indent=2)
    logger.info("Wrote analysis JSON to %s", json_path)

    # 5) write HDF5
    h5_path = out_dir / f"{base_name}.h5"
    logger.debug("Writing analysis HDF5 to %s", h5_path)
    export_to_hdf5(raw_record, out_path=h5_path)
    logger.info("Wrote analysis HDF5 to %s", h5_path)


def play_pipeline_mode(
    input_file: str,
    channel: str,
    processing_str: str | None,
    processing_file: str | None,
    output_folder: str | None,
    output_name: str | None,
    scale_bar_nm: int | None,
    zmin: str = "auto",
    zmax: str = "auto",
) -> None:
    """
    Launch an interactive GUI to browse an AFM stack with optional filters.

    Steps
    -----
    1. Load the AFM stack from `input_file` and `channel`.
    2. Optionally apply a processing pipeline (inline or YAML/JSON).
    3. Display frames in a QT-based viewer with live filtering controls.
    4. Allow on-the-fly export to bundles or GIF via GUI.

    Parameters
    ----------
    input_file : str
        Path to the AFM input file.
    channel : str
        Data channel to display (e.g., "height_trace").
    processing_str : str or None
        Inline processing string as for `process_pipeline_mode`.
    processing_file : str or None
        Path to a YAML/JSON file specifying processing steps.
    output_folder : str or None
        Directory for any GUI-triggered exports.
    output_name : str or None
        Base filename (no extension) for GUI exports.
    scale_bar_nm : int or None
        Scale bar length (in nm) displayed on frames.
    zmin : str
        Minimum Z-value mapping (float or `"auto"`).
    zmax : str
        Maximum Z-value mapping (float or `"auto"`).

    Returns
    -------
    None
    """
    try:
        afm_stack = AFMImageStack.load_data(input_file, channel=channel)
    except Exception as e:
        raise LoadError(f"Failed to load {input_file}") from e
    # Determine fps from metadata
    frame_metadata = getattr(afm_stack, "frame_metadata", None)
    line_rate = None
    if (
        isinstance(frame_metadata, (list, tuple))
        and len(frame_metadata) > 0
        and isinstance(frame_metadata[0], dict)
    ):
        line_rate = frame_metadata[0].get("line_rate")
    if not line_rate:
        logger.warning("No line_rate in metadata; defaulting to 1 fps")
        fps = 1.0
    else:
        fps = line_rate / afm_stack.image_shape[0]
        logger.debug(
            f"Computed fps from line_rate: {fps:.2f} (line_rate={line_rate}, image_shape={afm_stack.image_shape})"  # noqa
        )

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
        scale_bar_nm=scale_bar_nm or 100,
        zmin=zmin,
        zmax=zmax,
    )


class Wizard:
    """
    Interactive processing + analysis wizard.

    This class provides a REPL interface for building and running processing and
    analysis pipelines on AFM data. Users can add, remove, reorder, save, and load
    both processing and analysis steps. Processing steps operate on 2D image data
    and are applied in sequence to a stack, while analysis steps are executed on the
    processed data or raw data if no processing is specified.

    The wizard can be driven interactively through text commands. It supports
    saving and loading pipelines from YAML or JSON files, exporting processed
    stacks in multiple formats, and creating animated GIFs.

    Examples
    --------
    >>> wiz = Wizard("data.afm", "height", ".", "output", 100)
    >>> wiz.run()

    Parameters
    ----------
    input_file : str
        Path to the AFM input file.
    channel : str
        Channel name to load from the AFM file (e.g. "height_trace").
    output_folder : Optional[str]
        Directory to write exported analysis/processing outputs. If `None`,
        the output directory defaults to the current working directory when
        an export is performed (see `_run_analysis_and_export`).
    output_name : Optional[str]
        Base filename (no extension) for exported files. If `None`, the stem
        of `input_file` will be used at export time.
    scale_bar_nm : Optional[int]
        Default scale bar size in nm used when creating GIFs. Use `0` to
        explicitly disable the scale bar. This value is passed to `export_gif`
        as given; that function determines how to interpret `0`.
    io : IO or None, optional
        I/O abstraction used for reading commands and printing output. If None,
        standard input/output will be used. Mainly for testing and automation.

    Attributes
    ----------
    process_steps : list[tuple[str, Dict]]
        Ordered list of processing steps (name, kwargs).
    analysis_steps : list[tuple[str, Dict]]
        Ordered list of analysis steps (name, kwargs).
    processed_stack : Optional[AFMImageStack]
        Cached processed AFM stack produced by running the processing steps.
    processed_steps_snapshot : list[tuple[str, Dict]]
        Snapshot of `process_steps` at time `processed_stack` was produced.
    afm_stack : AFMImageStack
        Raw loaded AFM stack.
    input_path : pathlib.Path
        Path object for `input_file`.

    Raises
    ------
    FileNotFoundError
        If `input_file` does not exist.
    LoadError
        If the AFM stack cannot be loaded from the given file.
    """

    def __init__(
        self,
        input_file: str,
        channel: str,
        output_folder: Optional[str],
        output_name: Optional[str],
        scale_bar_nm: Optional[int],
        io: Optional[IO] = None,
    ) -> None:
        """
        Initialize the Wizard for interactive processing and analysis of a AFM stack.

        Parameters
        ----------
        input_file : str
            Path to AFM input file.
        channel : str
            Data channel to load.
        output_folder : Optional[str]
            Directory for output files. If `None` no directory is set immediately;
            a default of the current working directory will be used when export
            functions are called.
        output_name : Optional[str]
            Base output filename.
        scale_bar_nm : Optional[int]
            Default scale bar length for GIFs.
        io : object or None, optional
            I/O abstraction with `input` and `print`-like methods. If None, standard
            input/output will be used. Mainly intended for testing or scripted runs.

        Raises
        ------
        FileNotFoundError
            If `input_file` does not exist.
        LoadError
            If AFM file cannot be loaded into an AFMImageStack.
        """
        self.input_file = input_file
        self.input_path = Path(input_file)
        self.channel = channel
        self.output_folder = output_folder
        self.output_name = output_name
        self.scale_bar_nm = scale_bar_nm
        self.io = io or IO()

        if not self.input_path.exists():
            raise FileNotFoundError(f"File not found: {self.input_path}")

        try:
            self.afm_stack = AFMImageStack.load_data(self.input_path, channel=channel)
        except Exception as e:
            raise LoadError(f"Failed to load {input_file}") from e

        # wizard state
        self.process_steps: List[Tuple[str, Dict[str, Any]]] = []
        self.analysis_steps: List[Tuple[str, Dict[str, Any]]] = []
        self.processed_stack = None
        self.processed_steps_snapshot: List[Tuple[str, Dict[str, Any]]] = []

    # -------------------------
    # Utilities / UX
    # -------------------------
    def print_help(self) -> None:
        """Print available wizard commands."""
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
        """Pretty-print current processing steps."""
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
        """Pretty-print current analysis steps."""
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
        Prompt user for a processing step's parameters (simple fallback).

        This method provides a small set of built-in prompts for commonly-used
        processing filters. It is a fallback — the module-level
        `ask_for_processing_params` (which introspects/loads processing callables)
        may provide a richer interactive experience and is used in `handle_add`.

        Parameters
        ----------
        step_name : str
            Name of the processing step.

        Returns
        -------
        Dict[str, Any]
            Mapping of parameter names to values.
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
        """Add a processing step interactively or from the command line."""
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
        """Insert a clear-mask step into the processing pipeline."""
        self.process_steps.append(("clear", {}))
        self.io.say("Added clear-mask step.")

    def handle_remove(self, parts: List[str]) -> None:
        """Remove a processing step by index."""
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
        """Reorder processing steps by moving one from old_index to new_index."""
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

    def handle_save(self, parts: List[str]) -> None:
        """Save current processing steps to a YAML file."""
        if len(parts) != 2:
            self.io.say("Usage: save <path/to/output.yaml>")
            return
        save_path = Path(parts[1])
        processing_dict = {"filters": []}
        for name, kw in self.process_steps:
            entry = {"name": name}
            entry.update(kw)
            processing_dict["filters"].append(entry)
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w", encoding="utf8") as f:
                yaml.dump(processing_dict, f, sort_keys=False)
            self.io.say(f"Processing saved to {save_path}")
        except Exception as e:
            self.io.say(f"Error saving processing: {e}")

    def handle_load(self, parts: List[str]) -> None:
        """Load processing steps from a YAML file exported by `save`."""
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
        Run the current processing steps and cache the processed AFM stack.

        The method calls into `process_stack` with the accumulated `process_steps`
        and stores the returned AFMImageStack in `self.processed_stack` for
        subsequent in-memory analysis.

        Returns
        -------
        AFMImageStack
            The processed AFM stack object.

        Raises
        ------
        LoadError
            If loading/processing fails.
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
        Execute the currently queued processing steps and optionally export outputs.

        Behavior
        ----
        - Runs processing and caches the result.
        - Prompts the user whether to export (tif/npz/h5) and/or create a GIF.

        Parameters
        ----------
        parts : List[str]
            Tokenized user command (not currently used).
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
            export_bundles(
                afm_stack_local, self.output_folder, self.output_name, formats
            )
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
            zmin_choice = self._parse_scale_value(zmin_raw)
            zmax_choice = self._parse_scale_value(zmax_raw)
            export_gif(
                afm_stack_local,
                True,
                self.output_folder,
                self.output_name,
                self.scale_bar_nm,
                zmin=zmin_choice,
                zmax=zmax_choice,
            )
        self.io.say("Processing complete; processed stack cached.")

    # -------------------------
    # Analysis handlers
    # -------------------------
    def handle_alist(self, parts: List[str]) -> None:
        """List current analysis steps."""
        self.io.say("Analysis steps:")
        self.list_analysis()

    def handle_aadd(self, parts: List[str]) -> None:
        """Add an analysis step (inline or interactive)."""
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
        """Remove an analysis step by index."""
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
        """Reorder analysis steps."""
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
        """Load analysis steps from a YAML/JSON file, restoring Python types."""
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

    def handle_asave(self, parts: List[str]) -> None:
        """Save analysis steps to a YAML/JSON file (sanitized for portability)."""
        if len(parts) != 2:
            self.io.say("Usage: asave <path.yaml or .json>")
            return
        path = Path(parts[1])

        # Build safe steps where 'name' is the first key for readability
        safe_steps = []
        for name, kw in self.analysis_steps:
            safe_kw = _sanitize_for_dump(kw) if kw else {}
            # Ensure name is first key in the emitted mapping
            # (yaml.safe_dump + sort_keys=False)
            entry = {"name": name}
            entry.update(safe_kw)
            safe_steps.append(entry)

        out = {"analysis": safe_steps}

        # Write YAML or JSON; YAML via safe_dump to avoid Python tags
        if path.suffix.lower() in (".yaml", ".yml"):
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf8") as f:
                yaml.safe_dump(
                    out, f, sort_keys=False, default_flow_style=False
                )  # noqa
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf8") as f:
                json.dump(out, f, indent=2)
        self.io.say(f"Analysis saved to {path}")

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
        to the current working directory if not set.

        The base filename is taken from `self.output_name`, or derived from the
        stem of `self.input_file` if `output_name` is not provided.

        Results are saved atomically to `<base_name>.json` and `<base_name>.h5`
        in the output directory.
        """
        pipeline = self._build_pipeline()
        raw_record = pipeline.run(stack, log_to=None)

        out_dir = Path(self.output_folder or ".")
        base_name = self.output_name or Path(self.input_file).stem

        safe_record = make_json_safe(raw_record)
        json_path = out_dir / f"{base_name}.json"
        h5_path = out_dir / f"{base_name}.h5"

        # atomic writes
        self._atomic_write_json(json_path, safe_record)
        self._atomic_write_hdf5(h5_path, raw_record)

        return raw_record

    def handle_arun(self, parts: List[str]) -> None:
        """
        Execute the analysis pipeline, running processing if needed and export results.

        The function supports two modes:
        - If the wizard has processing steps configured (self.process_steps), it will
          ensure the processed stack is available and up-to-date and then run analysis
          in-memory on that processed stack.
        - If there are no processing steps, analysis runs directly on the loaded AFM
        stack.

        Results are exported as JSON and HDF5 into `self.output_folder` (or current
        directory). The JSON is created with `make_json_safe` and both JSON/HDF5 are
        written atomically.

        Parameters
        ----------
        parts : List[str]
            Tokenized CLI command (unused).
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
        except Exception as e:
            # log full traceback for debugging while keeping CLI output friendly
            logger.exception("Analysis failed")
            self.io.say(f"Analysis failed: {e}")

    # -------------------------
    # REPL
    # -------------------------
    def run(self) -> None:
        """Start the interactive REPL loop."""
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
    Print the current playNano environment metadata.

    Returns
    -------
    None
    """
    import json

    from playNano.utils.system_info import gather_environment_info

    env = gather_environment_info()
    print(json.dumps(env, indent=2))


class IO:
    """Simple I/O adapter used by Wizard for testability and swapping UI later."""

    def ask(self, prompt: str) -> str:
        """Prompt for a line of input from the user."""
        return input(prompt)

    def say(self, msg: str) -> None:
        """Write a message to stdout."""
        print(msg)
