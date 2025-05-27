import argparse
import logging
from pathlib import Path

from playNano.io.loader import load_afm_stack


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
        "--make-gif", action="store_true", help="Export flattened stack as animated GIF"
    )
    return parser.parse_args()


def main():
    """
    Main function to process an AFM image stack:
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

    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"File not found: {input_path}")
        return

    # Load data
    logger.info(f"Loading AFM stack from {input_path}")
    try:
        afm_stack = load_afm_stack(input_path, channel=args.channel)
    except Exception as e:
        logger.exception(f"Failed to load AFM stack: {e}")
        return

    # Flatten stack
    logger.info("Flattening AFM image stack...")
    afm_stack.flatten_images(keep_raw=args.save_raw)

    # Optional: Export as GIF
    if args.make_gif:
        # Determine and create output directory
        output_dir = Path(args.output_folder) if args.output_folder else Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving outputs to: {output_dir}")

        from playNano.io.gif_export import create_gif_with_scale_and_timestamp

        gif_path = output_dir / "flattened.gif"
        timestamps = [meta["timestamp"] for meta in afm_stack.frame_metadata]
        create_gif_with_scale_and_timestamp(
            afm_stack.image_stack,
            afm_stack.pixel_size_nm,
            timestamps,
            output_path=gif_path,
        )
        logger.info(f"Exported GIF to {gif_path}")

    logger.info("Processing complete.")


if __name__ == "__main__":
    main()
