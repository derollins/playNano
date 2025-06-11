"""Core logic for CLI actions in playNano."""

import logging
import sys
from pathlib import Path

from playNano.afm_stack import AFMImageStack
from playNano.cli.utils import (
    is_valid_step,
    parse_processing_file,
    parse_processing_string,
)
from playNano.errors import LoadError
from playNano.io.export import (
    export_bundles,
    save_h5_bundle,
    save_npz_bundle,
    save_ome_tiff_stack,
)
from playNano.io.gif_export import export_gif
from playNano.playback.vis import play_stack_cv
from playNano.processing.core import process_stack

logger = logging.getLogger(__name__)


def run_pipeline_mode(
    input_file: str,
    channel: str,
    processing_str: str | None,
    processing_file: str | None,
    export: str | None,
    make_gif: bool,
    output_folder: str | None,
    output_name: str | None,
    scale_bar_nm: int | None,
) -> None:
    """
    Load an AFM file, apply processing, and optionally export results and GIF.

    Priority for parsing steps:
      1. If --processing-file is provided, read YAML/JSON from that file.
      2. Else if --processing is provided, parse the processing string.

    After building the ordered list of (step_name, kwargs),
    we build a ProcessingPipeline and run it. Finally, handle
    exports and GIF as before.

    Parameters
    ----------
    input_file : str
        Path to the AFM input file.
    channel : str
        Data channel to extract from the file (e.g., "height_trace").
    processing_str : str or None
        Optional inline string defining filters,
        e.g., "remove_plane; gaussian_filter:sigma=2".
    processing_file : str or None
        Path to a YAML or JSON file specifying processing steps.
    export : str or None
        Comma-separated list of export formats (e.g., "tif,npz,h5").
    make_gif : bool
        Whether to create a GIF of the filtered data.
    output_folder : str or None
        Output directory for exports. Defaults to "output/".
    output_name : str or None
        Optional override for output file base name.
    scale_bar_nm : int or None
        Length of scale bar for GIF. Defaults to 100 nm.
    """
    # 1) Build steps_with_kwargs
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
        formats = [f.strip() for f in export.split(",") if f.strip()]
        export_bundles(afm_stack, formats, output_folder, output_name)

    # 4) GIF
    export_gif(afm_stack, make_gif, output_folder, output_name, scale_bar_nm)


def play_pipeline_mode(
    input_file: str,
    channel: str,
    processing_str: str | None,
    processing_file: str | None,
    output_folder: str | None,
    output_name: str | None,
    scale_bar_nm: int | None,
) -> None:
    """
    Load an AFM file and display it interactively with optional filters.

    Parameters
    ----------
    input_file : str
        Path to the AFM input file.
    channel : str
        Data channel to extract from the file.
    processing_str : str or None
        Inline filter string (e.g., "gaussian_filter:sigma=2").
    processing_file : str or None
        YAML or JSON file describing processing steps.
    output_folder : str or None
        Directory for saving on-the-fly exports (if enabled in viewer).
    output_name : str or None
        Base name for exports generated from the viewer.
    scale_bar_nm : int or None
        Scale bar length in nanometers for visualization.
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

    if processing_file:
        steps_with_kwargs = parse_processing_file(processing_file)
    elif processing_str:
        steps_with_kwargs = parse_processing_string(processing_str)
    else:
        steps_with_kwargs = []

    play_stack_cv(
        afm_stack,
        fps=fps,
        output_dir=output_folder,
        output_name=output_name,
        steps_with_kwargs=steps_with_kwargs,
        scale_bar_nm=scale_bar_nm or 100,
    )


def wizard_mode(
    input_file: str,
    channel: str,
    output_folder: str | None,
    output_name: str | None,
    scale_bar_nm: int | None,
) -> None:
    """
    Launch an interactive REPL for building and applying a processing pipeline.

    Parameters
    ----------
    input_file : str
        Path to the AFM input file.
    channel : str
        Channel to extract (e.g., "height_trace").
    output_folder : str or None
        Directory to save exports or GIFs.
    output_name : str or None
        Optional override for base output name.
    scale_bar_nm : int or None
        Length of scale bar in nanometers for GIF export.
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
    )  # noqa
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
            elif step_name == "threshold_mask":
                params_to_ask = [("threshold", float, 1.0)]
            # (You can extend this list as new filters appear)

            kwargs: dict[str, object] = {}
            for param_name, param_type, default in params_to_ask:
                while True:
                    val_str = input(
                        f"  Enter {param_name} (default={default}): "
                    ).strip()  # noqa
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
            ):  # noqa
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
                ).strip()  # noqa
                formats = [
                    fmt.strip().lower() for fmt in fmt_str.split(",") if fmt.strip()
                ]  # noqa
                export_bundles(afm_stack, formats, output_folder, output_name)

            # Ask if user wants to make a GIF
            gif_choice = input("Create a GIF? (y/n): ").strip().lower()
            if gif_choice in ("y", "yes"):
                export_gif(afm_stack, True, output_folder, output_name, scale_bar_nm)

            print("Wizard finished. Exiting.\n")
            sys.exit(0)
            return

        else:
            print(f"Unknown command: {verb}. Type `help` for a list of commands.\n")


def write_exports(
    afm_stack: AFMImageStack,
    out_folder: Path,
    base_name: str,
    formats: list[str],
    raw: bool = False,
) -> None:
    """
    Write out requested bundles from an AFM stack (.data must be final version).

    Parameters
    ----------
    afm_stack : AFMImageStack
        The AFM stack containing final .data, .pixel_size_nm, .frame_metadata, .channel
    out_folder : Path
        Directory to write export files (will be created if needed)
    base_name : str
        Base file name (no extension) for each export, e.g. "sample_01"
    formats : list of str
        Which formats to write; valid set = {"tif", "npz", "h5"}.
    raw : bool, optional
        If True, use the raw data from `afm_stack.processed["raw"]`.
        If False, use the final processed data in `afm_stack.data`.
        Default is False (use processed data).

    Raises
    ------
    SystemExit
        If any element of `formats` is not in {"tif","npz","h5"}.
    """
    if raw is False:
        stack_data = afm_stack.data
    elif raw is True and "raw" in afm_stack.processed:
        stack_data = afm_stack.processed["raw"]
    px_nm = afm_stack.pixel_size_nm
    timestamps = [md.get("timestamp") for md in afm_stack.frame_metadata]
    channel = afm_stack.channel

    out_folder.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(__name__)

    valid = {"tif", "npz", "h5"}
    for fmt in formats:
        if fmt not in valid:
            logger.error(f"Unsupported export format '{fmt}'. Choose from {valid}.")
            sys.exit(1)

    if "tif" in formats:
        tif_path = out_folder / f"{base_name}.ome.tif"
        logger.info(f"Writing OME-TIFF → {tif_path}")
        save_ome_tiff_stack(
            path=tif_path,
            stack=stack_data,
            pixel_size_nm=px_nm,
            timestamps=timestamps,
            channel=channel,
        )

    if "npz" in formats:
        npz_path = out_folder / f"{base_name}"
        logger.info(f"Writing NPZ bundle → {npz_path}.npz")
        save_npz_bundle(
            path=npz_path,
            stack=stack_data,
            pixel_size_nm=px_nm,
            timestamps=timestamps,
            channel=channel,
        )

    if "h5" in formats:
        h5_path = out_folder / f"{base_name}"
        logger.info(f"Writing HDF5 bundle → {h5_path}.h5")
        save_h5_bundle(
            path=h5_path,
            stack=stack_data,
            pixel_size_nm=px_nm,
            timestamps=timestamps,
            frame_metadata=afm_stack.frame_metadata,
            channel=channel,
        )
