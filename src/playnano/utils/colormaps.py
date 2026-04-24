"""
Utilities for loading, registering, and validating custom colormaps.

This module provides helper functions to register custom colormaps shipped
with the playNano package, validate colormap names, and safely resolve
user-provided colormap inputs with sensible fallbacks.
"""

import logging
from importlib import resources
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

logger = logging.getLogger(__name__)

DEFAULT_CMAP = "afm_brown"


def _load_and_register_cmap(name: str, path: Path) -> None:
    """
    Load a colormap from a CSV file and register it with Matplotlib.

    The CSV file is expected to contain a 256x3 array of RGB values in the
    range [0, 1] or [0, 255], compatible with `matplotlib.colors.ListedColormap`.
    If the colormap name is already registered, no action is taken.

    Parameters
    ----------
    name : str
        Name under which the colormap will be registered.
    path : pathlib.Path
        Path to the CSV file containing RGB data.

    Returns
    -------
    None

    Raises
    ------
    Exception
        Any exception raised by NumPy, Matplotlib, or file I/O is propagated
        to the caller. Callers are expected to handle failures gracefully.
    """
    rgb_data = np.loadtxt(path, delimiter=",")

    if name not in plt.colormaps():
        cmap = ListedColormap(rgb_data, name=name)
        plt.colormaps.register(cmap)
        # Register reversed version (conventional in AFM analysis)
        plt.colormaps.register(cmap.reversed(), name=f"{name}_r")


def register_custom_colormaps() -> None:
    """
    Load and register all custom playNano colormaps.

    Colormap definitions are stored as CSV files within the
    ``playnano.resources.colormaps`` package. Each colormap is registered
    globally with Matplotlib under its canonical name, along with a reversed
    variant (``<name>_r``).

    This function is safe to call multiple times; colormaps that are already
    registered will not be re-registered.

    Any failures encountered while loading individual colormaps are logged
    as errors, but do not prevent remaining colormaps from being processed.

    Returns
    -------
    None
    """
    # Resolve the resource directory even when installed as a wheel or zip
    resource_path = resources.files("playnano.resources.colormaps")

    cmaps_to_load = {
        "playnano_gold": "playnano_gold.csv",
        "afm_brown": "afm_brown.csv",
        "classic_afm": "classic_afm.csv",
    }

    for name, filename in cmaps_to_load.items():
        try:
            _load_and_register_cmap(name, resource_path / filename)
        except Exception as e:
            logger.error("Failed to load colormap %s: %s", name, e)


def is_valid_cmap(name: str) -> bool:
    """
    Check whether a colormap name is available in Matplotlib.

    This includes both:
    - Built-in Matplotlib colormaps
    - Custom playNano colormaps (if registered)

    Parameters
    ----------
    name : str
        Name of the colormap to check.

    Returns
    -------
    bool
        True if the colormap exists and can be used by Matplotlib,
        False otherwise.
    """
    if not isinstance(name, str):
        return False

    return name in plt.colormaps()


def resolve_cmap(cmap: str) -> str:
    """
    Resolve a user-provided colormap name to a valid colormap.

    If the requested colormap is not recognized, a warning is logged and
    a default colormap is returned instead. This function never raises
    an exception and always returns a usable colormap name.

    Parameters
    ----------
    cmap : str
        Name of the requested colormap.

    Returns
    -------
    str
        A valid colormap name. If `cmap` is invalid or unknown, this will be
        `DEFAULT_CMAP`.

    Notes
    -----
    This function is intended for user-facing inputs (e.g. CLI arguments,
    configuration files, or API parameters) where robustness is preferred
    over strict validation.

    See Also
    --------
    is_valid_cmap : Check whether a colormap name is supported.
    DEFAULT_CMAP : Default colormap used as a fallback.
    """
    if not is_valid_cmap(cmap):
        logger.warning(
            "Unknown colormap '%s', falling back to '%s'.",
            cmap,
            DEFAULT_CMAP,
        )
        return DEFAULT_CMAP
    return cmap


def get_available_cmaps() -> list[str]:
    """
    Return a sorted list of available colormap names.

    This includes all built-in Matplotlib colormaps as well as any custom
    colormaps that have been registered.

    Returns
    -------
    list of str
        Sorted list of colormap names.
    """
    return sorted(plt.colormaps())
