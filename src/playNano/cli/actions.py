"""Core logic for CLI actions in playNano."""

import json
import logging
import sys
from pathlib import Path

from playNano.afm_stack import AFMImageStack
from playNano.analysis.pipeline import AnalysisPipeline
from playNano.analysis.utils.common import export_to_hdf5, make_json_safe
from playNano.cli.utils import (
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
    except LoadError as e:
        logger.error(e)
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
    """Emit a warning if `stack.processed` is not a dict containing 'raw'."""
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
    else:
        steps = parse_analysis_string(analysis_str)

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
    line_rate = afm_stack.frame_metadata[0].get("line_rate", None)
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


def wizard_mode(
    input_file: str,
    channel: str,
    output_folder: str | None,
    output_name: str | None,
    scale_bar_nm: int | None,
) -> None:
    """
    Interactive REPL for step-by-step pipeline construction and execution.

    In wizard mode the user can:
      - `add <filter_name>`          Add a processing step
      - `remove <index>`             Remove a step
      - `move <old> <new>`           Reorder steps
      - `list`                       Show current steps
      - `save <path>`                Dump steps to YAML
      - `run`                        Execute pipeline and exit
      - `quit`                       Exit without running

    Parameters
    ----------
    input_file : str
        Path to the AFM input file.
    channel : str
        Data channel (e.g., "height_trace").
    output_folder : str or None
        Directory for any exports triggered during the session.
    output_name : str or None
        Base filename (no extension) for wizard exports.
    scale_bar_nm : int or None
        Scale bar length (in nm) for any GIF created.

    Returns
    -------
    None
    """
    logger = logging.getLogger(__name__)
    # Check if input file exists
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    logger.info("Launching processing wizard…")
    # Load the AFM stack
    try:
        afm_stack = AFMImageStack.load_data(input_path, channel=channel)
    except Exception as e:
        raise LoadError(f"Failed to load {input_file}") from e

    # We'll keep a list of (filter_name, kwargs) in wizard_steps
    wizard_steps: list[tuple[str, dict[str, object]]] = []

    def print_help():
        print("\nCommands:")
        print("  add <filter_name>     - Add a new step to the end")
        print("  remove <index>        - Remove step at 1-based index")
        print("  move <old> <new>      - Move step from old index to new index")
        print("  list                  - List current processing steps")
        print("  save <path>           - Save processing to YAML file")
        print("  run                   - Execute the processing now")
        print("  help                  - Show this help message")
        print("  quit                  - Exit without running\n")

    def list_steps():
        if not wizard_steps:
            print("  [no steps yet]\n")
            return
        for i, (name, kw) in enumerate(wizard_steps, start=1):
            if kw:
                params = ", ".join(f"{k}={v}" for k, v in kw.items())
                print(f"  {i}) {name} ({params})")
            else:
                print(f"  {i}) {name}")
        print()

    print(f"\nLoaded AFM stack: {input_path}")
    print(
        f"Channel: {channel}, frames={afm_stack.n_frames}, shape={afm_stack.image_shape}\n"  # noqa
    )
    print("Enter `help` for a list of commands.\n")

    while True:
        try:
            cmd = input("playNano wizard> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting wizard.")
            sys.exit(0)

        if not cmd:
            continue

        parts = cmd.split()
        verb = parts[0].lower()

        if verb in ("quit", "exit"):
            print("Exiting wizard without running.")
            sys.exit(0)

        elif verb == "help":
            print_help()

        elif verb == "list":
            print("\nCurrent processing steps:")
            list_steps()

        elif verb == "add":
            if len(parts) < 2:
                print("Usage: add <filter_name>\n")
                continue
            step_name = parts[1]
            if not is_valid_step(step_name):
                print(f"Unknown processing step: '{step_name}'")
                continue

            # Determine which parameters this filter can take, if any
            params_to_ask = []
            if step_name == "gaussian_filter":
                params_to_ask = [("sigma", float, 1.0)]
            elif step_name == "polynomial_flatten":
                params_to_ask = [("order", int, 2)]
            elif step_name == "mask_mean_offset":
                params_to_ask = [("factor", float, 1.0)]
            elif step_name == "mask_threshold":
                params_to_ask = [("threshold", float, 1.0)]
            elif step_name == "mask_below_threshold":
                params_to_ask = [("threshold", float, 1.0)]
            # (You can extend this list as new filters appear)

            kwargs: dict[str, object] = {}
            for param_name, param_type, default in params_to_ask:
                while True:
                    val_str = input(
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
                            val = val_str  # fallback
                        kwargs[param_name] = val
                        break
                    except ValueError:
                        print(
                            f"  Invalid {param_name}! Expecting a {param_type.__name__}. Try again."  # noqa
                        )  # noqa

            wizard_steps.append((step_name, kwargs))
            print(f"Added: {step_name} {kwargs}\n")

        elif verb == "remove":
            if len(parts) != 2 or not parts[1].isdigit():
                print("Usage: remove <index>\n")
                continue
            idx = int(parts[1])
            if idx < 1 or idx > len(wizard_steps):
                print(f"Index out of range (1-{len(wizard_steps)}).\n")
                continue
            removed = wizard_steps.pop(idx - 1)
            print(f"Removed step {idx}: {removed[0]}\n")

        elif verb == "move":
            if len(parts) != 3 or not parts[1].isdigit() or not parts[2].isdigit():
                print("Usage: move <old_index> <new_index>\n")
                continue
            old_i = int(parts[1]) - 1
            new_i = int(parts[2]) - 1
            if (
                old_i < 0
                or old_i >= len(wizard_steps)
                or new_i < 0
                or new_i > len(wizard_steps)
            ):
                print("Indices out of range.\n")
                continue
            item = wizard_steps.pop(old_i)
            wizard_steps.insert(new_i, item)
            print(f"Moved step from position {old_i+1} to {new_i+1}.\n")

        elif verb == "save":
            if len(parts) != 2:
                print("Usage: save <path/to/output.yaml>\n")
                continue
            save_path = Path(parts[1])
            processing_dict = {"filters": []}
            for name, kw in wizard_steps:
                entry = {"name": name}
                entry.update(kw)
                processing_dict["filters"].append(entry)
            try:
                import yaml

                with open(save_path, "w") as f:
                    yaml.dump(processing_dict, f, sort_keys=False)
                print(f"processing saved to {save_path}\n")
            except Exception as e:
                print(f"Error saving processing: {e}\n")

        elif verb == "run":
            if not wizard_steps:
                print("No steps to run. Use `add <filter_name>` first.\n")
                continue

            print("\nExecuting processing…\n")
            try:
                afm_stack = process_stack(Path(input_file), channel, wizard_steps)
            except LoadError as e:
                print(f"Error: {e}")
                continue

            print("processing execution complete.\n")

            # After run, ask if user wants to export
            export_choice = input("Export results? (y/n): ").strip().lower()
            if export_choice in ("y", "yes"):
                fmt_str = input(
                    "Enter formats (comma-separated, e.g. tif,npz,h5): "
                ).strip()
                formats = [
                    fmt.strip().lower() for fmt in fmt_str.split(",") if fmt.strip()
                ]
                export_bundles(afm_stack, output_folder, output_name, formats)

            # Ask if user wants to make a GIF
            gif_choice = input("Create a GIF? (y/n): ").strip().lower()
            if gif_choice in ("y", "yes"):
                zmin_choice = (
                    input("Enter a minimum value for the Z scale (or auto): ")
                    .strip()
                    .lower()
                )
                zmax_choice = (
                    input("Enter a maxiumum value for the Z scale (or auto): ")
                    .strip()
                    .lower()
                )
                export_gif(
                    afm_stack,
                    True,
                    output_folder,
                    output_name,
                    scale_bar_nm,
                    zmin=zmin_choice,
                    zmax=zmax_choice,
                )

            print("Wizard finished. Exiting.\n")
            sys.exit(0)
            return

        else:
            print(f"Unknown command: {verb}. Type `help` for a list of commands.\n")


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
