from pathlib import Path
from playNano.loaders.read_h5jpk import load_h5jpk
from playNano.stack.image_stack import AFMImageStack
# from playNano.io.load_nanonis import load_nanonis  # future reader
# from playNano.io.load_asylum import load_asylum  # future reader

def load_afm_stack(file_path: Path | str, channel: str = "height_trace") -> AFMImageStack:
    """
    Unified interface to load AFM stacks from various file formats.

    Parameters
    ----------
    file_path : Path | str
        Path to the AFM data file.
    channel : str
        Scan channel name (only used for JPK).

    Returns
    -------
    AFMImageStack
        Loaded image stack with metadata.
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".h5-jpk":
        return load_h5jpk(file_path, channel=channel)

    raise ValueError(f"Unsupported file type: {suffix}")
