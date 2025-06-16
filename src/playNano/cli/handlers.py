"""Handlers for the playNano CLI commands."""

import argparse
import logging
import sys

from playNano.cli.actions import play_pipeline_mode, run_pipeline_mode, wizard_mode


def setup_logging(level: int = logging.INFO) -> None:
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


def handle_play(args: argparse.Namespace) -> None:
    """
    Handle the 'play' subcommand: feed arguments into the run_pipeline_mode function.

    Parameters
    ----------
    args : argparse.Namespace
        Must contain:
        - input_file (str): path to AFM data
        - channel (str)
        - processing_str (str or None)
        - processing_file (str or None)
        - output_folder (str or None)
        - output_name (str or None)
        - scale_bar_nm (int or None), 0 turns off scale bar

    Returns
    -------
    None
    """
    try:
        play_pipeline_mode(
            input_file=args.input_file,
            channel=args.channel,
            processing_str=args.processing,
            processing_file=args.processing_file,
            output_folder=args.output_folder,
            output_name=args.output_name,
            scale_bar_nm=args.scale_bar_nm,
        )
    except Exception as e:
        logging.getLogger(__name__).error(e)
        sys.exit(1)


def handle_run(args: argparse.Namespace) -> None:
    """
    Handle the 'run' subcommand: feed arguments into the run_pipeline_mode function.

    Parameters
    ----------
    args : argparse.Namespace
        Must contain:
        - input_file (str): path to AFM data
        - channel (str)
        - processing_str (str or None)
        - processing_file (str or None)
        - export (str or None), comma-separated formats like "tif,npz,h5"
        - make_gif (bool), whether to create a GIF
        - output_folder (str or None)
        - output_name (str or None)
        - scale_bar_nm (int or None), 0 turns off scale bar

    Returns
    -------
    None
    """
    try:
        run_pipeline_mode(
            input_file=args.input_file,
            channel=args.channel,
            processing_str=args.processing,
            processing_file=args.processing_file,
            export=args.export,
            make_gif=args.make_gif,
            output_folder=args.output_folder,
            output_name=args.output_name,
            scale_bar_nm=args.scale_bar_nm,
        )
    except Exception as e:
        logging.getLogger(__name__).error(e)
        sys.exit(1)


def handle_processing_wizard(args: argparse.Namespace) -> None:
    """
    Interactive “wizard” for building a processing processing step by step.

    Usage:
      playnano processing <input_file>  [--channel ...] \
                                        [--output-folder ...] \
                                        [--output-name ...]

    The user then interacts with a simple REPL:
      add remove_plane
      add gaussian_filter
        sigma: 2.0
      list
      run
      quit

    Parameters
    ----------
    args : argparse.Namespace
        Must contain:
        - input_file (str): path to AFM data
        - channel (str)
        - output_folder (str or None)
        - output_name (str or None)
        - scale_bar_nm (int or None), 0 turns off scale bar

    Returns
    -------
    None
    """
    try:
        wizard_mode(
            input_file=args.input_file,
            channel=args.channel,
            output_folder=args.output_folder,
            output_name=args.output_name,
            scale_bar_nm=args.scale_bar_nm,
        )
    except Exception as e:
        logging.getLogger(__name__).error(e)
        sys.exit(1)
