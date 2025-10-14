"""Core functions for loading and processing AFMImageStacks."""

from pathlib import Path
from typing import Dict, List, Tuple

from playNano.afm_stack import AFMImageStack
from playNano.errors import LoadError
from playNano.processing.pipeline import ProcessingPipeline


def process_stack(
    input_path: Path,
    channel: str,
    steps: List[Tuple[str, Dict]],
) -> AFMImageStack:
    """
    Load an AFMImageStack from a file, apply a list of processing steps, and return it.

    Parameters
    ----------
    input_path : Path
        Path to the AFM stack file.

    channel : str
        Channel to load (e.g., 'h', 'z', etc.).

    steps : list of tuple
        List of processing steps in the form (step_name, kwargs). Special step_name
        values:
        - "clear" : clears the current mask
        - "mask"  : applies a mask function with kwargs
        - otherwise : treated as a filter name with kwargs

    Returns
    -------
    AFMImageStack
        The processed AFMImageStack.

    Raises
    ------
    LoadError
        If the AFM stack cannot be loaded from `input_path`.
    """
    try:
        stack = AFMImageStack.load_data(input_path, channel=channel)
    except Exception as e:
        raise LoadError(f"Failed to load {input_path}") from e

    pipeline = ProcessingPipeline(stack)
    for name, kwargs in steps:
        if name == "clear":
            pipeline.clear_mask()
        elif name == "mask":
            pipeline.add_mask(name, **kwargs)
        else:
            pipeline.add_filter(name, **kwargs)
    pipeline.run()
    return stack
