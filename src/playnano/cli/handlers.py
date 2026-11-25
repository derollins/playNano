"""
Functions that run each ``playnano`` subcommand from its parsed arguments.

Each handler passes the options to the matching function in
:mod:`playnano.cli.actions`. If that raises an error, the handler logs it
(with the traceback at ``--log-level DEBUG``) and exits with status 1.
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import NoReturn

from playnano.cli.actions import (
    Wizard,
    analyze_pipeline_mode,
    play_pipeline_mode,
    process_pipeline_mode,
)

logger = logging.getLogger(__name__)


def _exit_with_error(exc: Exception) -> NoReturn:
    """
    Log a command's error and exit with status 1.

    The traceback is included when logging is at DEBUG level
    (``--log-level DEBUG``), and left out otherwise.

    Parameters
    ----------
    exc : Exception
        The error that stopped the command.
    """
    logger.error(exc, exc_info=logger.isEnabledFor(logging.DEBUG))
    sys.exit(1)


def handle_play(args: argparse.Namespace) -> None:
    """
    Run ``playnano play``: open a stack in the GUI.

    Calls :func:`playnano.cli.actions.play_pipeline_mode`.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed options: ``input_file``, ``channel``, ``processing``,
        ``processing_file``, ``output_folder``, ``output_name``,
        ``scale_bar_nm``, ``zmin``, ``zmax``, ``cmap`` and ``fps``.

    Raises
    ------
    ValueError
        If ``input_file`` contains option flags such as ``--channel``, which
        happens when a quoted Windows path ends in a backslash. This is raised
        before the handler's error handling, so the command exits with status 2.
    """
    if isinstance(args.input_file, str):
        raw_input = args.input_file.strip()

        # Check if argument flags are embedded in the path (a quoted Windows path
        # ending in a backslash swallows the rest of the command line). Only a
        # flag after whitespace counts, so folder names like "run--2" are allowed.
        if re.search(r"(^|\s)--[A-Za-z]", raw_input):
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
            cmap=args.cmap,
            fps=args.fps,
        )
    except Exception as e:
        _exit_with_error(e)


def handle_process(args: argparse.Namespace) -> None:
    """
    Run ``playnano process``: process a stack and export the results.

    Calls :func:`playnano.cli.actions.process_pipeline_mode`.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed options: ``input_file``, ``channel``, ``processing``,
        ``processing_file``, ``export``, ``make_gif``, ``make_video``,
        ``make_sequence``, ``output_folder``, ``output_name``,
        ``scale_bar_nm``, ``zmin``, ``zmax``, ``draw_ts``, ``cmap`` and
        ``fps``.
    """
    try:
        process_pipeline_mode(
            input_file=args.input_file,
            channel=args.channel,
            processing_str=args.processing,
            processing_file=args.processing_file,
            export=args.export,
            make_gif=args.make_gif,
            make_video=args.make_video,
            make_sequence=args.make_sequence,
            output_folder=args.output_folder,
            output_name=args.output_name,
            scale_bar_nm=args.scale_bar_nm,
            zmin=args.zmin,
            zmax=args.zmax,
            draw_ts=args.draw_ts,
            cmap=args.cmap,
            fps=args.fps,
        )
    except Exception as e:
        _exit_with_error(e)


def handle_wizard(args: argparse.Namespace) -> None:
    """
    Run ``playnano wizard``: build and run pipelines interactively.

    Starts a :class:`playnano.cli.actions.Wizard`; type ``help`` at its prompt
    for the commands.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed options: ``input_file``, ``channel``, ``output_folder``,
        ``output_name``, ``scale_bar_nm`` and ``cmap``.
    """
    try:
        Wizard(
            input_file=args.input_file,
            channel=args.channel,
            output_folder=args.output_folder,
            output_name=args.output_name,
            scale_bar_nm=args.scale_bar_nm,
            cmap=args.cmap,
        ).run()
    except Exception as e:
        _exit_with_error(e)


def handle_analyze(args: argparse.Namespace) -> None:
    """
    Run ``playnano analyze``: analyse a stack and save the results.

    Calls :func:`playnano.cli.actions.analyze_pipeline_mode`.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed options: ``input_file``, ``channel``, ``analysis_steps``,
        ``analysis_file``, ``output_folder`` and ``output_name``.
    """
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
        _exit_with_error(e)
