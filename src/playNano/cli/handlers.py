"""Handlers for the playNano CLI commands."""

import argparse
import logging
import sys
from pathlib import Path

from playNano.cli.actions import (
    Wizard,
    analyze_pipeline_mode,
    play_pipeline_mode,
    process_pipeline_mode,
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
        - zmin (float or str), minimum Z-value for normalization
        ('auto` for 1st percentile)
        - zmax (float or str), maximum Z-value for normalization
        ('auto` for 99th percentile)

    Returns
    -------
    None
    """
    if isinstance(args.input_file, str):
        raw_input = args.input_file.strip()

        # Check if known argument flags are embedded in the path
        # (user forgot closing quote)
        if "--channel" in raw_input or "--" in raw_input:
            raise ValueError(
                f"\nInvalid input path: '{raw_input}'\n"
                "⚠️  It looks like your input path includes CLI flags (e.g., '--channel').\n"  # noqa: E501
                "This usually happens when a quoted Windows path ends with a backslash, "  # noqa: E501
                "which escapes the closing quote.\n\n"
                "💡 FIX: Either:\n"
                '  - Add another backslash: "C:\\path\\to\\folder\\\\"\n'
                '  - Remove the trailing backslash: "C:\\path\\to\\folder"\n'
                '  - Use forward slashes: "C:/path/to/folder"\n'
            )

        file_path = Path(raw_input).expanduser()
    else:
        file_path = Path(args.input_file)

    try:
        play_pipeline_mode(
            input_file=file_path,
            channel=args.channel,
            processing_str=args.processing,
            processing_file=args.processing_file,
            output_folder=args.output_folder,
            output_name=args.output_name,
            scale_bar_nm=args.scale_bar_nm,
            zmin=args.zmin,
            zmax=args.zmax,
        )
    except Exception as e:
        logging.getLogger(__name__).error(e)
        sys.exit(1)


def handle_process(args: argparse.Namespace) -> None:
    """
    Handle the 'process' subcommand: feed arguments into the run_pipeline_mode function.

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
        process_pipeline_mode(
            input_file=args.input_file,
            channel=args.channel,
            processing_str=args.processing,
            processing_file=args.processing_file,
            export=args.export,
            make_gif=args.make_gif,
            output_folder=args.output_folder,
            output_name=args.output_name,
            scale_bar_nm=args.scale_bar_nm,
            zmin=args.zmin,
            zmax=args.zmax,
        )
    except Exception as e:
        logging.getLogger(__name__).error(e)
        sys.exit(1)


def handle_wizard(args: argparse.Namespace) -> None:
    """
    Interactive wizard for building both processing and analysis pipelines.

    Usage:
      playnano processing <input_file> [--channel ...] \
                                 [--output-folder ...] \
                                 [--output-name ...]

    After launch, commands include:
      add/remove/move/list/save/load/run        for processing steps
      aadd/aremove/amove/alist/asave/aload/arun  for analysis steps
      help                                  show available commands
      quit                                  exit without running

    Parameters
    ----------
    args : argparse.Namespace
        Must contain:
        - input_file (str): path to AFM data
        - channel (str)
        - output_folder (str or None)
        - output_name (str or None)
        - scale_bar_nm (int or None), 0 disables scale bar

    Returns
    -------
    None
    """
    try:
        Wizard(
            input_file=args.input_file,
            channel=args.channel,
            output_folder=args.output_folder,
            output_name=args.output_name,
            scale_bar_nm=args.scale_bar_nm,
        ).run()
    except Exception as e:
        logging.getLogger(__name__).error(e)
        sys.exit(1)


def handle_analyze(args: argparse.Namespace) -> None:
    """Handle the 'analyze' subcommand: run only analysis & export results."""
    try:
        analyze_pipeline_mode(
            input_file=args.input_file,
            channel=args.channel,
            analysis_str=args.analysis_steps,
            analysis_file=args.analysis_file,
            output_folder=args.output_folder,
            output_name=args.output_name,
        )
    except Exception as e:
        logging.getLogger(__name__).error(e, exc_info=True)
        sys.exit(1)
