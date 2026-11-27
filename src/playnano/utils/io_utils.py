"""
Helpers for reading and writing data.

Frame metadata, height units, scaling images to 8-bit, output names and
folders, and converting results to types JSON can store.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

### --- CONSTANTS --- ###

INVALID_CHARS = r'\/:*?"<>|'
INVALID_FOLDER_CHARS = r'*?"<>|'


FRAME_METADATA_KEYS = (
    "timestamp",
    "frame_pixel_size_nm",
    "frame_duration_s",
    "start_epoch_ms",
    "scan_direction",
    "line_rate",
)

HEIGHT_UNITS = ["m", "cm", "mm", "um", "nm", "pm"]

#: File extensions removed from user-supplied output names. Anything else after a
#: dot is kept, so names such as ``exp_1.5nM`` or ``run.v2`` stay intact.
OUTPUT_EXTENSIONS = (
    ".ome.tiff",
    ".ome.tif",
    ".tiff",
    ".tif",
    ".h5",
    ".hdf5",
    ".npz",
    ".json",
    ".gif",
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".png",
    ".jpg",
    ".jpeg",
    ".csv",
    ".yaml",
    ".yml",
)

# --- FRAME METADATA ------------------------------------------------


def build_frame_metadata(
    *,
    timestamp: float,
    frame_pixel_size_nm: float,
    frame_duration_s: float | None = None,
    start_epoch_ms: int | None = None,
    scan_direction: str | None = None,
    line_rate: float | None = None,
) -> dict:
    """
    Make one frame's metadata dict, as every playNano reader does.

    Values are converted to plain Python types so the dict can be saved as
    JSON. Settings of the whole recording, such as bidirectional scanning, go in
    :attr:`playnano.afm_stack.AFMImageStack.acquisition` instead.

    Parameters
    ----------
    timestamp : float
        Time of the frame in seconds, from the start of frame 0.
    frame_pixel_size_nm : float
        Pixel size of this frame, in nm.
    frame_duration_s : float, optional
        Time taken to record the frame, in seconds. Default is None (unknown).
    start_epoch_ms : int, optional
        Start of the frame, in milliseconds since 1970 (UTC). Default is None
        (unknown).
    scan_direction : str, optional
        Slow-scan direction of this frame, for example ``"topDown"`` or
        ``"bottomUp"`` (NanoScope ``.spm`` sequences can alternate). Default is
        None (unknown).
    line_rate : float, optional
        Lines scanned per second. Default is None (unknown).

    Returns
    -------
    dict
        The keys of ``FRAME_METADATA_KEYS``, with None for unknown values.
    """
    return {
        "timestamp": float(timestamp),
        "frame_pixel_size_nm": float(frame_pixel_size_nm),
        "frame_duration_s": (
            None if frame_duration_s is None else float(frame_duration_s)
        ),
        "start_epoch_ms": None if start_epoch_ms is None else int(start_epoch_ms),
        "scan_direction": None if scan_direction is None else str(scan_direction),
        "line_rate": None if line_rate is None else float(line_rate),
    }


# --- HEIGHT UNIT HANDLING ------------------------------------------------


def guess_height_data_units(stack: np.ndarray) -> str:
    """
    Guess the unit of height data from the range of its values.

    The range (largest minus smallest finite value) gives:

    - more than 1e4: ``"pm"``;
    - more than 1e-2, up to 1e4: ``"nm"``;
    - more than 1e-4, up to 1e-2: ``"um"``;
    - more than 1e-5, up to 1e-4: ``"mm"``;
    - up to 1e-5: ``"m"``.

    A single extreme value can change the result.

    Parameters
    ----------
    stack : numpy.ndarray
        Height data, any shape. NaN and infinite values are ignored.

    Returns
    -------
    {"pm", "nm", "um", "mm", "m"}
        The guessed unit.

    Raises
    ------
    ValueError
        If there are no finite values.

    Examples
    --------
    >>> import numpy as np
    >>> from playnano.utils.io_utils import guess_height_data_units
    >>> guess_height_data_units(np.array([0.0, 5e-9, 2e-8]))
    'm'
    >>> guess_height_data_units(np.array([0.0, 3.0]))
    'nm'
    """
    finite = stack[np.isfinite(stack)]
    if finite.size == 0:
        raise ValueError("No finite values in data.")

    z_range = finite.max() - finite.min()
    if z_range > 1e4:
        return "pm"
    elif 1e-2 < z_range <= 1e4:
        return "nm"
    elif 1e-4 < z_range <= 1e-2:
        return "um"
    elif 1e-5 < z_range <= 1e-4:
        return "mm"
    else:
        return "m"


def convert_height_units_to_nm(data: np.ndarray, unit: str) -> np.ndarray:
    """
    Convert height data to nm.

    Parameters
    ----------
    data : numpy.ndarray
        Height data.
    unit : {"pm", "nm", "um", "mm", "m"}
        Unit of ``data``, for example from :func:`guess_height_data_units`.

    Returns
    -------
    numpy.ndarray
        A new array in nm.

    Raises
    ------
    ValueError
        If ``unit`` is not one of the units above.
    """
    unit_to_multiplier = {
        "pm": 1e-3,  # 1 pm = 1e-3 nm
        "nm": 1.0,  # already in nm
        "um": 1e3,  # 1 µm = 1000 nm
        "mm": 1e6,  # 1 mm = 1e6 nm
        "m": 1e9,  # 1 m = 1e9 nm
    }

    if unit not in unit_to_multiplier:
        raise ValueError(
            f"Unrecognized unit '{unit}'. Must be one of: {list(unit_to_multiplier)}"
        )

    return data * unit_to_multiplier[unit]


# --- IMAGE UTILITIES ------------------------------------------------


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Scale an image to 0-255 as uint8, from its minimum to its maximum.

    NaN and infinite values are set to 0 first, so they count as 0. Values are
    rounded down. An image with one value gives all zeros.

    Parameters
    ----------
    image : numpy.ndarray
        Image, any shape.

    Returns
    -------
    numpy.ndarray
        Array of the same shape, dtype uint8.
    """
    # Replace NaNs and Infs with 0
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute min and max
    min_val = np.min(image)
    max_val = np.max(image)

    # Avoid division by zero
    if max_val - min_val == 0:
        return np.zeros_like(image, dtype=np.uint8)

    # Normalize to [0, 255]
    norm = (image - min_val) / (max_val - min_val) * 255
    return norm.astype(np.uint8)


def compute_zscale_range(
    data: np.ndarray,
    zmin: float | str = "auto",
    zmax: float | str = "auto",
    lower_percentile: int = 1,
    upper_percentile: int = 99,
) -> tuple[float, float]:
    """
    Work out the lower and upper limits of a colour scale.

    Parameters
    ----------
    data : numpy.ndarray
        Heights, any shape. NaN and infinite values are ignored.
    zmin : float, "auto" or None, optional
        Lower limit: a number is used as given, ``"auto"`` uses the
        ``lower_percentile`` percentile of ``data`` and None its minimum.
        Default is ``"auto"``.
    zmax : float, "auto" or None, optional
        Upper limit, in the same way, using ``upper_percentile`` or the
        maximum. Default is ``"auto"``.
    lower_percentile : float, optional
        Percentile for ``zmin="auto"``. Default is 1.
    upper_percentile : float, optional
        Percentile for ``zmax="auto"``. Default is 99.

    Returns
    -------
    tuple of float
        ``(zmin, zmax)``.

    Raises
    ------
    ValueError
        If ``zmin`` or ``zmax`` is not a number, ``"auto"`` or None, or the
        lower limit is greater than the upper. Also if ``data`` has no finite
        values and a limit is None.
    IndexError
        If ``data`` has no finite values and a limit is ``"auto"``.

    Examples
    --------
    >>> import numpy as np
    >>> from playnano.utils.io_utils import compute_zscale_range
    >>> data = np.arange(101.0)
    >>> [float(z) for z in compute_zscale_range(data)]
    [1.0, 99.0]
    >>> [float(z) for z in compute_zscale_range(data, zmin=None, zmax=50)]
    [0.0, 50.0]
    """
    flat = data.ravel()
    flat = flat[np.isfinite(flat)]

    # Process zmin
    if zmin == "auto":
        zmin_val = np.percentile(flat, lower_percentile)
    elif zmin is None:
        zmin_val = np.min(flat)
    else:
        try:
            zmin_val = float(zmin)
        except (TypeError, ValueError):
            raise ValueError("zmin must be a float, 'auto', or None.") from None

    # Process zmax
    if zmax == "auto":
        zmax_val = np.percentile(flat, upper_percentile)
    elif zmax is None:
        zmax_val = np.max(flat)
    else:
        try:
            zmax_val = float(zmax)
        except (TypeError, ValueError):
            raise ValueError("zmax must be a float, 'auto', or None.") from None

    # Validation
    if zmin_val > zmax_val:
        raise ValueError("zmin must be less than or equal to zmax.") from None

    logger.debug(f"[Z-scaling] zmin={zmin_val:.3f} nm, zmax={zmax_val:.3f} nm")
    return zmin_val, zmax_val


# --- OUTPUT FILE AND FOLDER UTILS ------------------------------------------------


def sanitize_output_name(name: str, default: str) -> str:
    """
    Clean a user-supplied output name.

    Folders and surrounding spaces are removed, then one known extension
    (``OUTPUT_EXTENSIONS``, in any case), so other dots are kept:
    ``"exp_1.5nM.gif"`` becomes ``"exp_1.5nM"``.

    Parameters
    ----------
    name : str or None
        The name given by the user.
    default : str
        Returned if ``name`` is None or empty, or nothing is left after
        cleaning.

    Returns
    -------
    str
        The cleaned name, without an extension.

    Raises
    ------
    ValueError
        If the cleaned name contains a character in ``INVALID_CHARS``.

    Examples
    --------
    >>> from playnano.utils.io_utils import sanitize_output_name
    >>> sanitize_output_name(" results/exp_1.5nM.GIF ", "default")
    'exp_1.5nM'
    >>> sanitize_output_name(None, "default")
    'default'
    """
    if not name:
        return default
    name = Path(name.strip()).name
    lower = name.lower()
    for ext in OUTPUT_EXTENSIONS:
        if lower.endswith(ext):
            name = name[: -len(ext)]
            break
    name = name.strip()
    if not name:
        return default

    if any(c in name for c in INVALID_CHARS):
        raise ValueError(f"Invalid characters in output name: {INVALID_CHARS}")

    return name


def prepare_output_directory(folder: str | None, default: str = "output") -> Path:
    """
    Check an output folder and create it if missing.

    Parameters
    ----------
    folder : str, pathlib.Path or None
        The folder; relative paths are relative to the current folder. If None
        or empty, ``default``.
    default : str, optional
        Folder used when ``folder`` is not given. Default is ``"output"``.

    Returns
    -------
    pathlib.Path
        The absolute path of the folder, which now exists.

    Raises
    ------
    ValueError
        If any part of the path contains any of ``* ? " < > |``.
    """
    if folder is None:
        folder = default
    elif not isinstance(folder, str):
        try:
            folder = str(folder)
        except Exception as e:
            raise ValueError(f"Invalid folder path: {e}") from e
    folder = folder.strip() if folder else default
    folder_path = Path(folder).resolve()
    for part in folder_path.parts:
        if any(c in part for c in INVALID_FOLDER_CHARS):
            raise ValueError(
                f"Invalid characters in output folder path: {INVALID_FOLDER_CHARS}"
            )
    folder_path.mkdir(parents=True, exist_ok=True)
    return folder_path


def ensure_not_input_path(
    output_path: str | Path, input_path: str | Path | None
) -> None:
    """
    Raise an error if writing to ``output_path`` would overwrite the input file.

    Parameters
    ----------
    output_path : str or Path
        File that is about to be written.
    input_path : str, Path or None
        File or folder the data was loaded from. If None, nothing is checked.

    Raises
    ------
    ValueError
        If ``output_path`` and ``input_path`` refer to the same file.
    """
    if input_path is None:
        return
    out = Path(output_path)
    inp = Path(input_path)
    same = out.resolve() == inp.resolve()
    if not same:
        # Catches the same file reached by a different path, such as a different
        # letter case on Windows or macOS, or a link. Fails if either is missing.
        try:
            same = out.samefile(inp)
        except OSError:
            same = False
    if same:
        raise ValueError(
            f"Refusing to overwrite the input file {inp}. Choose a different "
            "--output-folder or --output-name."
        )


# --- SERIALIZATION AND TYPE CONVERSION ------------------------------------------------


def _json_safe_key(key):
    """Convert a dict key to a type JSON accepts (str, int, float, bool or None)."""
    if isinstance(key, np.generic):
        key = key.item()
    if key is None or isinstance(key, (str, int, float, bool)):
        return key
    return str(key)


def make_json_safe(obj):
    """
    Convert NumPy values and functions in nested data to types JSON can store.

    - NumPy arrays become lists, and NumPy integers and floats become ``int``
      and ``float``.
    - Dicts, lists and tuples are converted item by item; tuples stay tuples
      (JSON writes them as lists). Dict keys that JSON cannot store become
      strings.
    - Functions and classes are replaced by their name.
    - Anything else is returned unchanged, including NumPy booleans and sets,
      which JSON cannot store.

    Parameters
    ----------
    obj : object
        The data to convert. Not changed.

    Returns
    -------
    object
        The converted data.

    Examples
    --------
    >>> import numpy as np
    >>> from playnano.utils.io_utils import make_json_safe
    >>> make_json_safe({"n": np.int64(3), "xy": np.array([1.5, 2.0]), "f": len})
    {'n': 3, 'xy': [1.5, 2.0], 'f': 'len'}
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, dict):
        return {_json_safe_key(k): make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_safe(v) for v in obj)
    elif callable(obj):  # convert functions to their names
        return obj.__name__
    else:
        return obj


# --- ATTRIBUTE DECODING ------------------------------------------------


def decode_hdf5_attr(attr: bytes | str) -> str:
    """
    Return an HDF5 attribute as a string.

    Parameters
    ----------
    attr : bytes or str
        The attribute value. Bytes are decoded as UTF-8; anything else is
        passed to ``str``.

    Returns
    -------
    str
        The value as a string.
    """
    if isinstance(attr, bytes):
        return attr.decode("utf-8")
    return str(attr)
