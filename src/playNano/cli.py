"""Main script implementing CLI for playNano."""

import argparse
import logging
import sys
from importlib import metadata
from pathlib import Path

from playNano.io.export import save_h5_bundle, save_npz_bundle, save_ome_tiff_stack
from playNano.processing.filters import register_filters
from playNano.processing.masked_filters import register_mask_filters
from playNano.processing.masking import register_masking
from playNano.stack.afm_stack import AFMImageStack

INVALID_CHARS = r'\/:*?"<>|'
INVALID_FOLDER_CHARS = r'*?"<>|'

# Built-in filters and mask dictionaries
FILTER_MAP = register_filters()
MASK_MAP = register_masking()
MASK_FILTERS_MAP = register_mask_filters()

# Names of all entry-point plugins (if any third-party filters are installed)
ALL_ENTRYPOINT_NAMES = {
    ep.name for ep in metadata.entry_points(group="playNano.filters")
}

preset_steps = [
    "remove_plane",
    "mask_mean_offset",
    "row_median_align",
    "zero_mean",
    "gaussian_filter",
]


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure the logging format and level.

    Parameters
    ----------
    level : int, optional
        Logging level (e.g., logging.INFO, logging.DEBUG). Default is logging.INFO.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def sanitize_output_name(name: str, default: str) -> str:
    """
    Sanitize output file names by removing extensions and stripping whitespace.

    Parameters
    ----------
    name : str
        The output file name provided by the user.
    default : str
        Default name to use if `name` is empty or None.

    Returns
    -------
    str
        Sanitized base file name without extension.
    """
    if not name:
        return default
    name = name.strip()
    # Remove extension if any
    try:
        name = Path(name).with_suffix("").name
    except ValueError:
        return default

    if any(c in name for c in INVALID_CHARS):
        raise ValueError(f"Invalid characters in output name: {INVALID_CHARS}")

    return name


def prepare_output_directory(folder: str | None, default: str = "output") -> Path:
    """
    Validate, resolve, and create the output directory if it doesn't exist.

    Parameters
    ----------
    folder : str or None
        User-provided output folder path. If None, use `default`.
    default : str, optional
        Default folder name to use if `folder` not specified.

    Returns
    -------
    Path
        A Path object pointing to the created output directory.

    Raises
    ------
    ValueError
        If any part of the folder path contains invalid characters.
    """
    folder_path = Path(folder.strip()) if folder else Path(default)
    for part in folder_path.parts:
        if any(c in part for c in INVALID_FOLDER_CHARS):
            raise ValueError(
                f"Invalid characters in output folder path: {INVALID_FOLDER_CHARS}"
            )
    folder_path.mkdir(parents=True, exist_ok=True)
    return folder_path


def parse_filter_list(filter_arg: str | None) -> list[str]:
    """
    Given comma-separated strings of filters and masks, produce a list of names.

    Parameters
    ----------
    filter_arg : str or None
        Comma-separated filters, e.g. "topostats_flatten,median_filter"

    Returns
    -------
    list of str
        List of stripped filter names. Empty if filter_arg is None or blank.
    """
    if not filter_arg:
        steps = []
    elif filter_arg.startswith("preset"):
        steps = preset_steps
    else:
        steps = [s.strip() for s in filter_arg.split(",") if s.strip()]
    return steps


def write_exports(
    afm_stack: AFMImageStack,
    out_folder: Path,
    base_name: str,
    formats: list[str],
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

    Raises
    ------
    SystemExit
        If any element of `formats` is not in {"tif","npz","h5"}.
    """
    stack_data = afm_stack.data
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


def handle_play(args: argparse.Namespace) -> None:
    """
    Handle the 'play' subcommand: launch OpenCV window and return.

    Parameters
    ----------
    args : argparse.Namespace
        Must contain:
        - input_file (str): path to AFM data
        - channel (str)
        - filters (str or None)
        - output_folder (str or None)
        - output_name (str or None)
    """
    logger = logging.getLogger(__name__)
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"File not found: {input_path}")
        sys.exit(1)

    logger.info("Starting AFM stack in PLAY mode…")
    try:
        afm_stack = AFMImageStack.load_data(input_path, channel=args.channel)
    except Exception as e:
        logger.error(f"Failed to load AFM stack: {e}")
        sys.exit(1)

    fps = afm_stack.frame_metadata[0]["line_rate"] / afm_stack.image_shape[0]
    filter_steps = parse_filter_list(args.filters)

    from playNano.io.vis import play_stack_cv

    if args.scale_bar_nm:
        _scale_bar_nm = args.scale_bar_nm
    else:
        _scale_bar_nm = 100

    play_stack_cv(
        afm_stack,
        fps=fps,
        output_dir=args.output_folder,
        output_name=args.output_name,
        filter_steps=filter_steps,
        scale_bar_nm=_scale_bar_nm,
    )
    # Once the user closes the window, we simply return.
    return


def handle_run(args: argparse.Namespace) -> None:
    """
    Handle the 'run' subcommand: apply filters, then write exports & GIF.

    Parameters
    ----------
    args : argparse.Namespace
        Must contain:
        - input_file (str)
        - channel (str)
        - filters (str or None)
        - export (str or None)
        - make_gif (bool)
        - output_folder (str or None)
        - output_name (str or None)
    """
    logger = logging.getLogger(__name__)
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"File not found: {input_path}")
        sys.exit(1)

    logger.info("Starting AFM stack in RUN mode…")
    try:
        afm_stack = AFMImageStack.load_data(input_path, channel=args.channel)
    except Exception as e:
        logger.error(f"Failed to load AFM stack: {e}")
        sys.exit(1)

    input_stem = input_path.stem

    # 1) Apply filters (if any)
    steps = parse_filter_list(args.filters)
    if steps:
        try:
            afm_stack.apply(steps)
        except ValueError as e:
            logger.error(f"Error in apply(…): {e}")
            sys.exit(1)

    # 2) Write any requested bundles
    if args.export:
        formats = [fmt.strip().lower() for fmt in args.export.split(",") if fmt.strip()]
        try:
            out_dir = prepare_output_directory(args.output_folder, default="output")
        except ValueError as e:
            logger.error(str(e))
            sys.exit(1)

        base_name = sanitize_output_name(args.output_name, input_stem)
        write_exports(afm_stack, out_dir, base_name, formats)

    # 3) Write GIF if requested
    if args.make_gif:
        try:
            gif_dir = prepare_output_directory(args.output_folder, default="output")
        except ValueError as e:
            logger.error(str(e))
            sys.exit(1)

        base_name = sanitize_output_name(args.output_name, input_stem)
        gif_name = f"{base_name}_filtered" if steps else base_name
        gif_path = gif_dir / f"{gif_name}.gif"

        from playNano.io.gif_export import create_gif_with_scale_and_timestamp

        timestamps = [md["timestamp"] for md in afm_stack.frame_metadata]

        if args.scale_bar_nm:
            _scale_bar_nm = args.scale_bar_nm
        else:
            _scale_bar_nm = 100

        logger.info(f"Writing GIF → {gif_path}")
        create_gif_with_scale_and_timestamp(
            afm_stack.data,
            afm_stack.pixel_size_nm,
            timestamps,
            output_path=gif_path,
            scale_bar_length_nm=_scale_bar_nm,
            cmap_name="afmhot",
        )

    logger.info("Run processing complete.")
    return


def main() -> None:
    """
    Parse command-line arguments and dispatch to the appropriate CLI subcommand.

    - Set up argument parsing for 'play' and 'run' subcommands,
    each with their own options.
    - Configure logging level based on user input.
    - Show help and exit if no subcommand is provided.
    - Call the handler function associated with the chosen subcommand.

    Usage:
      playnano play  <input_file> [--filters …] [--output-folder …] [--output-name …]
      playnano run   <input_file> [--filters …] [--export …] [--make-gif]
        [--output-folder …] [--output-name …]
    """
    parser = argparse.ArgumentParser(
        description="playNano: Load, filter, export, or play HS-AFM image stacks."
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging level (default=INFO).",
    )

    subparsers = parser.add_subparsers(
        title="subcommands",
        dest="command",
        help="Choose one subcommand: 'play' or 'run'.",
    )

    # 1) 'play' subcommand
    play_parser = subparsers.add_parser(
        "play", help="Interactive play mode (OpenCV window)."
    )
    play_parser.add_argument(
        "input_file", type=str, help="Path to AFM input file or folder."
    )
    play_parser.add_argument(
        "--channel",
        type=str,
        default="height_trace",
        help="Channel to read (default=height_trace).",
    )
    play_parser.add_argument(
        "--filters",
        type=str,
        help="Comma-separated filter names to apply when user presses 'f'.",
    )
    play_parser.add_argument(
        "--output-folder",
        type=str,
        help="Folder to save any exported GIF (if user hits 'e').",
    )
    play_parser.add_argument(
        "--output-name", type=str, help="Base name for exported GIF (no extension)."
    )
    play_parser.add_argument(
        "--scale-bar-nm",
        type=int,
        help="Interger length of scale bar in nm",
    )
    play_parser.set_defaults(func=handle_play)

    # 2) 'run' subcommand
    run_parser = subparsers.add_parser(
        "run", help="Run mode: apply filters & export bundles/GIF."
    )
    run_parser.add_argument(
        "input_file", type=str, help="Path to AFM input file or folder."
    )
    run_parser.add_argument(
        "--channel",
        type=str,
        default="height_trace",
        help="Channel to read (default=height_trace).",
    )
    run_parser.add_argument(
        "--filters", type=str, help="Comma-separated filter names to apply in order."
    )
    run_parser.add_argument(
        "--export",
        type=str,
        help="Comma-separated formats to export: 'tif', 'npz', 'h5'.",
    )
    run_parser.add_argument(
        "--make-gif",
        action="store_true",
        help="Also write an animated GIF after filtering.",
    )
    run_parser.add_argument(
        "--output-folder",
        type=str,
        help="Folder to write bundles and/or GIF (default='./output').",
    )
    run_parser.add_argument(
        "--output-name", type=str, help="Base name for output files (no extension)."
    )
    run_parser.add_argument(
        "--scale-bar-nm",
        type=int,
        help="Interger length of scale bar in nm",
    )
    run_parser.set_defaults(func=handle_run)

    args = parser.parse_args()
    setup_logging(getattr(logging, args.log_level.upper()))

    if args.command is None:
        # No subcommand: just show help and exit
        parser.print_help(file=sys.stderr)
        sys.exit(0)

    # Dispatch to the chosen subcommand
    args.func(args)


if __name__ == "__main__":
    main()
