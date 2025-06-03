"""Main script implimenting CLI for playNano."""

import argparse
import logging
import sys
from pathlib import Path

from playNano.stack.afm_stack import AFMImageStack

INVALID_CHARS = r'\/:*?"<>|'
INVALID_FOLDER_CHARS = r'*?"<>|'


def setup_logging(level=logging.INFO):
    """
    Configure the logging format and level.

    Parameters
    ----------
    level : int, optional
        Logging level (e.g., logging.INFO, logging.DEBUG). Default is logging.INFO.

    Returns
    -------
    None
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def parse_args():
    """
    Parse command-line arguments for AFM stack processing.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Flatten an AFM .h5-jpk video stack or folder of .jpk video frames."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to AFM input file (.h5-jpk, etc.) or folder of .jpk files",
    )
    parser.add_argument(
        "--channel", type=str, default="height_trace", help="Channel to read"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--filter",
        type=str,
        help="Name of filter to apply (e.g., topostats_filter). Applies non-interactively.",  # noqa
    )
    parser.add_argument("--output-folder", type=str, help="Folder to save outputs")
    parser.add_argument(
        "--output-name",
        type=str,
        help="Base name for output file (without extension, e.g., 'sample_01')",
    )
    parser.add_argument(
        "--make-gif", action="store_true", help="Export flattened stack as animated GIF"
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Pop up an OpenCV window to play the stack (raw first, space to flatten)",
    )

    return parser.parse_args()


def sanitize_output_name(name: str, default: str) -> str:
    """
    Sanitize output file names by removing extensions like .gif & stripping whitespace.

    Parameters
    ----------
    name : str
        The output file name provided by the user.

    Returns
    -------
    str
        Sanitized base file name without extension.
    """
    if not name:
        return default
    name = name.strip()
    name = Path(name).with_suffix("").name
    if any(c in name for c in INVALID_CHARS):
        raise ValueError(f"Invalid characters in output name: {INVALID_CHARS}")
    return name


def prepare_output_directory(folder: str | None, default: str = "output") -> Path:
    """
    Validate, resolve, and create the output directory if it doesn't exist.

    Parameters
    ----------
    folder : str or None
        User-provided output folder path. If None, a default folder is used.
    default : str, optional
        Default folder name to use if `folder` is not specified.

    Returns
    -------
    Path
        A resolved Path object pointing to the created output directory.

    Raises
    ------
    ValueError
        If any part of the folder path contains invalid characters.
    """
    folder_path = Path(folder.strip()) if folder else Path(default)

    # Validate each part of the folder path (excluding drive/root on Windows)
    for part in folder_path.parts:
        if any(c in part for c in INVALID_FOLDER_CHARS):
            raise ValueError(
                f"Invalid characters in output folder path: {INVALID_FOLDER_CHARS}"
            )

    folder_path.mkdir(parents=True, exist_ok=True)
    return folder_path


def main():
    """
    Process an AFM image stack.

    Workflow:
    - Loads the input file or folder.
    - Flattens the AFM image stack using TopoStats.
    - Exports a GIF with timestamps and scale bar.

    Returns
    -------
    None

    Notes
    -----
    - Input can be a `.h5-jpk` file or a folder of `.jpk` files.
    - Uses `playNano.io.loader.load_afm_stack` to load data.
    - Uses `playNano.io.gif_export.create_gif_with_scale_and_timestamp` to export GIFs.
    """
    args = parse_args()
    setup_logging(getattr(logging, args.log_level.upper()))
    logger = logging.getLogger(__name__)
    logger.info("Starting AFM stack processing...")

    # Load and verify input
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"File not found: {input_path}")
        sys.exit(1)  # Exit with error code 1

    # Load data
    logger.info(f"Loading AFM stack from {input_path}")
    try:
        afm_stack = AFMImageStack.load_data(input_path, channel=args.channel)
        input_stem = input_path.stem
    except Exception as e:
        logger.exception(f"Failed to load AFM stack: {e}")
        sys.exit(1)  # Exit with error code 1
        return
    fps = afm_stack.frame_metadata[0]["line_rate"] / afm_stack.image_shape[0]

    # If user wants to _play_ the stack interactively, pop up video now:
    if args.play:
        from playNano.io.vis import play_stack_cv

        _output_dir = args.output_folder
        _output_name = args.output_name

        # fps approximated from metadata of first frame
        play_stack_cv(
            afm_stack,
            fps=fps,
            output_dir=_output_dir,
            output_name=_output_name,
        )

        # after user closes window, we fall through into flatten/GIF logic--- want to stop this? # noqa

    # Optional: Export as GIF
    if args.make_gif:
        # Determine and create output directory
        try:
            output_dir = prepare_output_directory(args.output_folder)
        except ValueError as e:
            logger.error(str(e))
            sys.exit(1)  # Exit with error code 1
        logger.info(f"Saving outputs to: {output_dir}")

        # Apply filters if selected
        if args.filter:
            logger.info(f"Applying filter: {args.filter}")
            # Example: apply filter by name
            afm_stack.apply([args.filter])

        # Remove stem and whitepace from file name input or use input file stem if none provided. # noqa
        try:
            output_stem = sanitize_output_name(args.output_name, input_stem)
        except ValueError as e:
            logger.error(str(e))
            sys.exit(1)  # Exit with error code 1

        # Add _filtered to stem if filters applied.
        if args.filter:
            gif_path = output_dir / f"{output_stem}_filtered.gif"
        else:
            gif_path = output_dir / f"{output_stem}.gif"

        from playNano.io.gif_export import create_gif_with_scale_and_timestamp

        timestamps = [meta["timestamp"] for meta in afm_stack.frame_metadata]
        create_gif_with_scale_and_timestamp(
            afm_stack.data,
            afm_stack.pixel_size_nm,
            fps,
            timestamps,
            output_path=gif_path,
        )
        logger.info(f"Exported GIF to {gif_path}")

    logger.info("Processing complete.")
