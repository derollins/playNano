"""
Read the results of AFMReader's loaders across its versions.

AFMReader 0.0.7 and earlier return a tuple from each loader:
``(image, pixel_to_nm)`` from ``load_jpk`` and ``load_spm``, and
``(image, pixel_to_nm, metadata)`` from ``load_asd``. From 0.0.8 they return an
``AFMScan`` object with ``image``, ``pixel_to_nanometre_scaling``, ``z_units``
and ``metadata`` attributes. The readers in this package pass every AFMReader
result through :func:`unpack_afmreader_result` so they work with both.
"""

from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np


def _afmreader_version() -> str:
    """Return the installed AFMReader version, or ``"unknown"``."""
    try:
        return version("afmreader")
    except PackageNotFoundError:
        return "unknown"


def unpack_afmreader_result(result: Any) -> tuple[np.ndarray, float, dict]:
    """
    Return the image, pixel size and metadata from an AFMReader loader.

    Parameters
    ----------
    result : tuple or AFMScan
        The value returned by ``AFMReader.asd.load_asd``,
        ``AFMReader.jpk.load_jpk`` or ``AFMReader.spm.load_spm``.

    Returns
    -------
    image : numpy.ndarray
        The image data, as returned by AFMReader.
    pixel_to_nm : float
        The size of one pixel in nanometres.
    metadata : dict
        The file metadata, or an empty dict if the loader returns none.

    Raises
    ------
    TypeError
        If ``result`` is neither a 2- or 3-item tuple nor an object with
        ``image`` and ``pixel_to_nanometre_scaling`` attributes.

    Notes
    -----
    ``AFMScan.z_units`` (0.0.8 and later) is not used. The readers keep their
    existing behaviour with both versions: ``.jpk`` and ``.spm`` data are taken
    as nanometres, and ``.asd`` topography is rescaled from a unit guessed
    from the data range.
    """
    # Tuple from AFMReader 0.0.7 and earlier
    if isinstance(result, tuple) and len(result) in (2, 3):
        image, pixel_to_nm = result[0], result[1]
        metadata = result[2] if len(result) == 3 else None
    # AFMScan object from AFMReader 0.0.8 onwards
    elif hasattr(result, "image") and hasattr(result, "pixel_to_nanometre_scaling"):
        image = result.image
        pixel_to_nm = result.pixel_to_nanometre_scaling
        metadata = getattr(result, "metadata", None)
    else:
        raise TypeError(
            f"Unexpected result from AFMReader {_afmreader_version()}: "
            f"{type(result).__name__}. playNano supports AFMReader 0.0.7 and 0.0.8."
        )
    return np.asarray(image), float(pixel_to_nm), dict(metadata or {})
