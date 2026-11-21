"""Load an AFM file and run a list of processing steps on it."""

from pathlib import Path
from typing import Dict, List, Tuple

from playnano.afm_stack import AFMImageStack
from playnano.errors import LoadError
from playnano.processing.pipeline import ProcessingPipeline


def process_stack(
    input_path: Path,
    channel: str,
    steps: List[Tuple[str, Dict]],
) -> AFMImageStack:
    """
    Load an AFM file, run a list of processing steps on it and return the stack.

    This is what ``playnano process`` uses.

    Parameters
    ----------
    input_path : pathlib.Path
        AFM file, or folder of frame files, to load.
    channel : str
        Channel to load, for example ``"height_trace"``.
    steps : list of tuple of (str, dict)
        Processing steps in order, as ``(name, kwargs)`` pairs, for example
        ``[("remove_plane", {}), ("mask_threshold", {"threshold": 1.0})]``. A
        name can be any filter, mask, video processing or stack edit step;
        ``"clear"`` clears the current mask.

    Returns
    -------
    AFMImageStack
        The processed stack. ``stack.data`` holds the result and
        ``stack.processed["raw"]`` the original data.

    Raises
    ------
    LoadError
        If the file cannot be loaded.
    ValueError
        If a step name is not recognised.

    See Also
    --------
    playnano.processing.pipeline.ProcessingPipeline : Runs the steps.
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
        elif name:
            pipeline.add_filter(name, **kwargs)
    pipeline.run()
    return stack
