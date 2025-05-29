"""Main script implimenting CLI for playNano."""

import argparse
import logging
import sys
from pathlib import Path

from playNano.io.loader import load_afm_stack

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
        "--save-raw", action="store_true", help="Keep a copy of the raw image stack"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
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
    - Optionally saves the raw stack and exports a GIF with timestamps and scale bar.

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

    # Determine and create output directory
    try:
        output_dir = prepare_output_directory(args.output_folder, "output")
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)  # Exit with error code 1
    logger.info(f"Saving outputs to: {output_dir}")

    # Remove stem and whitepace from file name input or use flattened if none provided.
    try:
        output_stem = sanitize_output_name(args.output_name, "flattened")
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)  # Exit with error code 1
    gif_path = output_dir / f"{output_stem}.gif"

    # Load data
    logger.info(f"Loading AFM stack from {input_path}")
    try:
        afm_stack = load_afm_stack(input_path, channel=args.channel)
    except Exception as e:
        logger.exception(f"Failed to load AFM stack: {e}")
        sys.exit(1)  # Exit with error code 1
        return

    # Flatten stack
    logger.info("Flattening AFM image stack...")
    afm_stack.flatten_images(keep_raw=args.save_raw)

    # Optional: Export as GIF
    if args.make_gif:
        from playNano.io.gif_export import create_gif_with_scale_and_timestamp

        timestamps = [meta["timestamp"] for meta in afm_stack.frame_metadata]
        create_gif_with_scale_and_timestamp(
            afm_stack.image_stack,
            afm_stack.pixel_size_nm,
            timestamps,
            output_path=gif_path,
        )
        logger.info(f"Exported GIF to {gif_path}")

    logger.info("Processing complete.")
