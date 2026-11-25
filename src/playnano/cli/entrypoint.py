"""The ``playnano`` command: argument parsing and dispatch to the subcommands."""

import argparse
import logging
import sys

from playnano import __version__
from playnano.cli.actions import print_env_info
from playnano.cli.handlers import (
    handle_analyze,
    handle_play,
    handle_process,
    handle_wizard,
)
from playnano.errors import LoadError
from playnano.utils.colormaps import DEFAULT_CMAP

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """
    Set the log format and level for the command line.

    Parameters
    ----------
    level : int, optional
        Logging level, for example ``logging.DEBUG``. Default is
        ``logging.INFO``.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )


def make_fps_type(fallback: float):
    """
    Make an argparse ``type`` that reads a frame rate, with a fallback.

    Parameters
    ----------
    fallback : float
        Frame rate used, with a logged warning, when the value is not a number.

    Returns
    -------
    callable
        Function that converts the option's text to a float.
    """

    def fps_type(value: str) -> float:
        """If a non-numerical value is provided for fps, return default value."""
        try:
            return float(value)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid --fps value '%s'; using default %.1f FPS", value, fallback
            )
            return fallback

    return fps_type


def main() -> None:
    """
    Run the ``playnano`` command.

    Parses the command line, sets the log level (``--log-level``) and runs the
    subcommand:

    - ``play``: view a stack in the GUI (:func:`~playnano.cli.handlers.handle_play`);
    - ``process``: process a stack and export it
      (:func:`~playnano.cli.handlers.handle_process`);
    - ``analyze``: analyse a stack and save the results
      (:func:`~playnano.cli.handlers.handle_analyze`);
    - ``wizard``: build and run processing and analysis interactively
      (:func:`~playnano.cli.handlers.handle_wizard`);
    - ``env-info``: print the software environment.

    Run ``playnano <subcommand> --help`` for each subcommand's options.

    Exit status is 0 on success; 1 if the command fails (the subcommands log
    the error and exit with 1); 2 for a usage error, including no
    subcommand, or an unexpected error.
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

    parser.add_argument(
        "--version",
        action="version",
        version=f"playnano {__version__}",
    )
    subparsers = parser.add_subparsers(
        title="subcommands",
        dest="command",
        help="Choose one subcommand: 'play', 'process', 'analyze' or 'wizard'.",
    )

    # 1) 'play' subcommand
    play_parser = subparsers.add_parser(
        "play", help="Interactive play mode (GUI window)."
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
        "--output-folder",
        type=str,
        help="Folder to save any exported GIF (if user hits 'e').",
    )
    play_parser.add_argument(
        "--output-name", type=str, help="Base name for exported GIF (no extension)."
    )
    play_parser.add_argument(
        "--scale-bar-nm",
        dest="scale_bar_nm",
        type=int,
        default=None,
        help="Integer length of scale bar in nm (default=100) set to 0 to disable scale bar.",  # noqa
    )
    play_parser.add_argument(
        "--fps",
        nargs="?",
        default=None,
        const=5.0,
        type=make_fps_type(5.0),
        help=(
            "Frames per second for animated exports. "
            "If not specified, the frame rate is taken from the data. "
            "If specified without a value, or with a non-numeric value, "
            "a default of 5 frames per second is used."
        ),
    )
    play_parser.add_argument(
        "--zmin",
        type=str,
        default="auto",
        help="The minimum value of the z scale, float or 'auto' (default=('auto').",  # noqa
    )
    play_parser.add_argument(
        "--zmax",
        type=str,
        default="auto",
        help="The maximum value of the z scale, float or 'auto' (default=('auto').",  # noqa
    )
    play_parser.add_argument(
        "--cmap",
        type=str,
        default=DEFAULT_CMAP,
        help="The initial colormap to use for visualisation (default=afm_brown).",
    )
    # Mutually exclusive: either processing string or processing file (or none)
    group = play_parser.add_mutually_exclusive_group()
    group.add_argument(
        "--processing",
        type=str,
        help=(
            "One-line processing string. Semicolon-delimited steps, where each step is "
            "'filter_name' or 'filter_name:param=value'. "
            "Example: "
            '"remove_plane; gaussian_filter:sigma=2.0; mask_threshold:threshold=2"'
        ),
    )
    group.add_argument(
        "--processing-file",
        type=str,
        help="Path to a YAML (or JSON) file describing the processing.",
    )
    play_parser.set_defaults(func=handle_play)

    # 2) 'wizard' subcommand (wizard)
    wizard_parser = subparsers.add_parser(
        "wizard", help="Launch interactive processing builder (wizard)."
    )
    wizard_parser.add_argument(
        "input_file", type=str, help="Path to AFM input file or folder."
    )
    wizard_parser.add_argument(
        "--channel",
        type=str,
        default="height_trace",
        help="Channel to read (default=height_trace).",
    )
    wizard_parser.add_argument(
        "--output-folder",
        type=str,
        help="Folder to write bundles/GIF from the wizard.",
    )
    wizard_parser.add_argument(
        "--output-name", type=str, help="Base name for output files (no extension)."
    )
    wizard_parser.add_argument(
        "--scale-bar-nm",
        type=int,
        help="Integer length of scale bar in nm",
    )
    wizard_parser.add_argument(
        "--cmap",
        type=str,
        default=DEFAULT_CMAP,
        help="The colormap to use for animated export (default=afm_brown).",
    )
    wizard_parser.set_defaults(func=handle_wizard)

    # 3) 'process' subcommand
    process_parser = subparsers.add_parser(
        "process", help="Process mode: apply filters & export bundles/GIF."
    )
    process_parser.add_argument(
        "input_file", type=str, help="Path to AFM input file or folder."
    )
    process_parser.add_argument(
        "--channel",
        type=str,
        default="height_trace",
        help="Channel to read (default=height_trace).",
    )
    process_parser.add_argument(
        "--export",
        type=str,
        help="Comma-separated formats to export: 'tif', 'npz', 'h5'.",
    )
    process_parser.add_argument(
        "--make-gif",
        action="store_true",
        help="Also write an animated GIF after filtering.",
    )
    process_parser.add_argument(
        "--make-sequence",
        action="store_true",
        help="Also create a folder with a sequence of images (png) after filtering.",
    )
    process_parser.add_argument(
        "--make-video",
        nargs="?",
        type=str,
        const="mp4",
        default=None,
        help="Export video(s); optionally specify comma-separated formats (default: mp4). Supported: mp4, avi.",  # noqa
    )
    process_parser.add_argument(
        "--output-folder",
        type=str,
        help="Folder to write bundles and/or GIF (default='./output').",
    )
    process_parser.add_argument(
        "--output-name", type=str, help="Base name for output files (no extension)."
    )
    process_parser.add_argument(
        "--scale-bar-nm",
        type=int,
        help="Integer length of scale bar in nm",
    )
    process_parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="Frames per second for animated exports (default=5.0).",
    )
    process_parser.add_argument(
        "--zmin",
        type=str,
        default="auto",
        help="The minimum value of the z scale, float or 'auto' (default=('auto').",  # noqa
    )
    process_parser.add_argument(
        "--zmax",
        type=str,
        default="auto",
        help="The maximum value of the z scale, float or 'auto' (default=('auto').",  # noqa
    )
    process_parser.add_argument(
        "--draw-ts",
        action="store_true",
        help=(
            "Draw frame timestamps on the GIF, video or image sequence "
            "(off by default)."
        ),
    )
    process_parser.add_argument(
        "--cmap",
        type=str,
        default=DEFAULT_CMAP,
        help="The colormap to use for animated export (default=afm_brown).",
    )
    # Mutually exclusive: either processing string or processing file (or none)
    filter_group = process_parser.add_mutually_exclusive_group()
    filter_group.add_argument(
        "--processing",
        type=str,
        help=(
            "One-line processing string. Semicolon-delimited steps, where each step is "
            "'filter_name' or 'filter_name:param=value'. "
            "Example: "
            '"remove_plane; gaussian_filter:sigma=2.0; mask_threshold:threshold=2"'
        ),
    )
    filter_group.add_argument(
        "--processing-file",
        type=str,
        help="Path to a YAML (or JSON) file describing the processing.",
    )

    process_parser.set_defaults(func=handle_process)

    # 4) 'analyze' subcommand
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze a processed stack: process analysis modules & export JSON/HDF5.",
    )
    analyze_parser.add_argument(
        "input_file", type=str, help="Path to AFM input file or folder."
    )
    analyze_parser.add_argument(
        "--channel",
        type=str,
        default="height_trace",
        help="Channel to read (default=height_trace).",
    )
    # inline string _or_ yaml file, mutually exclusive:
    analysis_group = analyze_parser.add_mutually_exclusive_group(required=True)
    analysis_group.add_argument(
        "--analysis-steps",
        type=str,
        help="Semicolon-delimited analysis steps, e.g. "
        "'feature_detection:mask_fn=mask_threshold,threshold=5;particle_tracking:max_distance=3.0'",  # noqa
    )
    analysis_group.add_argument(
        "--analysis-file",
        type=str,
        help="YAML/JSON file defining your analysis pipeline.",
    )
    analyze_parser.add_argument(
        "--output-folder",
        type=str,
        help="Folder to write the analysis JSON and HDF5 files (default='./output').",
    )
    analyze_parser.add_argument(
        "--output-name", type=str, help="Base name for output files (no extension)."
    )
    analyze_parser.set_defaults(func=handle_analyze)

    parser_env = subparsers.add_parser("env-info", help="Print environment info")
    parser_env.set_defaults(func=lambda args: print_env_info())

    args = parser.parse_args()
    setup_logging(getattr(logging, args.log_level.upper()))

    if args.command is None:
        # No subcommand is a usage error: show help and exit with code 2, as
        # argparse does for other usage errors.
        parser.print_help(file=sys.stderr)
        sys.exit(2)

    # Dispatch to the chosen subcommand
    try:
        args.func(args)
    except LoadError as e:
        logger.error(e)
        sys.exit(1)
    except Exception:
        logger.error("Unexpected error", exc_info=True)
        sys.exit(2)
    return


if __name__ == "__main__":
    main()
