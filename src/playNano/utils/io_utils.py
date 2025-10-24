"""Utility functions for IO operations in playNano."""

import logging
from pathlib import Path

import numpy as np

INVALID_CHARS = r'\/:*?"<>|'
INVALID_FOLDER_CHARS = r'*?"<>|'

logger = logging.getLogger(__name__)


height_units = ["m", "cm", "mm", "um", "nm", "pm"]


def pad_to_square(img: np.ndarray, border_color: int = 0) -> np.ndarray:
    """Pad a 2D grayscale image to a square canvas by centring it."""
    h, w = img.shape[:2]
    size = max(h, w)
    canvas = np.full((size, size), border_color, dtype=img.dtype)
    y = (size - h) // 2
    x = (size - w) // 2
    canvas[y : y + h, x : x + w] = img  # noqa
    return canvas


def guess_height_data_units(stack: np.ndarray) -> str:
    """
    Guess the most likely units of AFM height data from the data range.

    Parameters
    ----------
    stack : np.ndarray
        AFM height data array, typically 2D or 3D. Non-finite values
        (NaN or infinity) are ignored when determining the range.

    Returns
    -------
    str
        A string indicating the guessed unit of the height data, one of:
        'pm' (picometers), 'nm' (nanometers), 'um' (micrometers),
        'mm' (millimeters), or 'm' (meters).

    Raises
    ------
    ValueError
        If the input array contains no finite values.

    Notes
    -----
    The unit is estimated based on the numeric range of the data as follows:
        - Range > 1e4           : 'pm'
        - 1e-2 < Range <= 1e4   : 'nm'
        - 1e-4 < Range <= 1e-2  : 'um'
        - 1e-5 < Range <= 1e-4  : 'mm'
        - Range <= 1e-5         : 'm'

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([[0, 5e-9], [1e-8, 2e-8]])
    >>> guess_height_data_units(data)
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
    Convert AFM height data from the guessed unit to nanometers.

    Parameters
    ----------
    data : np.ndarray
        Input height data array, typically 2D or 3D.
    unit : str
        Unit string as returned by `guess_height_data_units`. Must be one of:
        'pm', 'nm', 'um', 'mm', or 'm'.

    Returns
    -------
    np.ndarray
        The input data converted to nanometers.

    Raises
    ------
    ValueError
        If the provided unit string is not recognized.
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


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Normalize a float image to the uint8 [0, 255] range, handling NaNs and Infs.

    Parameters
    ----------
    image : np.ndarray
        Input image as a NumPy array of floats. May contain NaNs or infinite values.

    Returns
    -------
    np.ndarray
        Normalized image as a uint8 NumPy array with values in the range [0, 255].
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


def sanitize_output_name(name: str, default: str) -> str:
    """
    Sanitize output file names by removing extensions and stripping whitespace.

    Parameters
    ----------
    name : str
        The output file name provided by the user.
    default : str
        Default name to use if `name` is empty or None.

    Returns
    -------
    str
        Sanitized base file name without extension.
    """
    if not name:
        return default
    name = name.strip()
    # Remove extension if any
    try:
        name = Path(name).with_suffix("").name
    except ValueError:
        return default

    if any(c in name for c in INVALID_CHARS):
        raise ValueError(f"Invalid characters in output name: {INVALID_CHARS}")

    return name


def prepare_output_directory(folder: str | None, default: str = "output") -> Path:
    """
    Validate, resolve, and create the output directory if it doesn't exist.

    Parameters
    ----------
    folder : str or None
        User-provided output folder path. If None, use `default`.
    default : str, optional
        Default folder name to use if `folder` not specified.

    Returns
    -------
    Path
        A Path object pointing to the created output directory.

    Raises
    ------
    ValueError
        If any part of the folder path contains invalid characters.
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


def compute_zscale_range(
    data: np.ndarray,
    zmin: float | str = "auto",
    zmax: float | str = "auto",
    lower_percentile: int = 1,
    upper_percentile: int = 99,
) -> tuple[float, float]:
    """
    Compute robust Z-scale bounds (height or intensity range) for normalization.

    Parameters
    ----------
    data : np.ndarray
        2D or 3D array of AFM image data.
    zmin : float, "auto", or None
        Lower bound: "auto" uses percentile, None uses data min, float uses value.
    zmax : float or "auto"
        Upper bound: "auto" uses percentile, None uses data max, float uses value.
    lower_percentile : int, optional
        Percentile to use for lower bound when zmin == "auto". Default is 1.
    upper_percentile : int, optional
        Percentile to use for upper bound when zmax == "auto". Default is 99.

    Returns
    -------
    (float, float)
        zmin and zmax values suitable for normalization.

    Raises
    ------
    ValueError
        If zmin > zmax after processing or invalid input types.
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


def make_json_safe(obj):
    """Recursively convert NumPy types and non-JSON objects into serializable ones."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_safe(v) for v in obj)
    elif callable(obj):  # convert functions to their names
        return obj.__name__
    else:
        return obj
