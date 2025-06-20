"""Entry point for the playNano CLI."""

import argparse
import logging
import sys

from playNano.cli.actions import print_env_info
from playNano.cli.handlers import handle_play, handle_processing_wizard, handle_run
from playNano.errors import LoadError

logger = logging.getLogger(__name__)


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


def main() -> None:
    """
    Parse command-line arguments and dispatch to the appropriate CLI subcommand.

    This function sets up the top-level argparse parser, configures logging,
    and then calls the appropriate handler function (`handle_play`, `handle_run`,
    or `handle_processing_wizard`) based on the chosen subcommand.

    - Set up argument parsing for 'play', 'run' and 'wizard' subcommands,
    each with their own options.
    - Configure logging level based on user input.
    - Show help and exit if no subcommand is provided.
    - Call the handler function associated with the chosen subcommand.

    Usage:
      playnano play  <input_file> [--processing …] [--output-folder …] [--output-name …]
        [--scale-bar-nm …] [--channel …]
      playnano run   <input_file> [--processing …] [--export …] [--make-gif]
        [--output-folder …] [--output-name …] [--scale-bar-nm …] [--channel …]
      playnano wizard <input_file> [--channel …] [--scale-bar-nm …]
        [--output-folder …] [--output-name …]

    Returns
    -------
    None
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
        help="Choose one subcommand: 'play', 'run' or 'processing'.",
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
    # Mutually exclusive: either processing string or processing file (or none)
    group = play_parser.add_mutually_exclusive_group()
    group.add_argument(
        "--processing",
        type=str,
        help=(
            "One-line processing string. Semicolon-delimited steps, where each step is"
            "'filter_name' or 'filter_name:param=value'. "
            "Example: "
            '"remove_plane; gaussian_filter:sigma=2.0; threshold_mask:threshold=2"'
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
    wizard_parser.set_defaults(func=handle_processing_wizard)

    # 3) 'run' subcommand
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
    # Mutually exclusive: either processing string or processing file (or none)
    group = run_parser.add_mutually_exclusive_group()
    group.add_argument(
        "--processing",
        type=str,
        help=(
            "One-line processing string. Semicolon-delimited steps, where each step is"
            "'filter_name' or 'filter_name:param=value'. "
            "Example: "
            '"remove_plane; gaussian_filter:sigma=2.0; threshold_mask:threshold=2"'
        ),
    )
    group.add_argument(
        "--processing-file",
        type=str,
        help="Path to a YAML (or JSON) file describing the processing.",
    )

    run_parser.set_defaults(func=handle_run)

    parser_env = subparsers.add_parser("env-info", help="Print environment info")
    parser_env.set_defaults(func=lambda args: print_env_info())

    args = parser.parse_args()
    setup_logging(getattr(logging, args.log_level.upper()))

    if args.command is None:
        # No subcommand: just show help and exit
        parser.print_help(file=sys.stderr)
        sys.exit(0)

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
