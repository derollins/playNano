"""
playNano's colormaps, and helpers for checking colormap names.

Importing :mod:`playnano` registers the colormaps ``"afm_brown"`` (the
default), ``"playnano_gold"`` and ``"classic_afm"``, each with a reversed
``"_r"`` version, so they can be used by name anywhere in Matplotlib.
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
    Register playNano's colormaps with Matplotlib.

    Reads ``afm_brown``, ``playnano_gold`` and ``classic_afm`` from the
    package's ``resources/colormaps`` folder and registers each, with its
    reversed ``_r`` version. Colormaps already registered are skipped, so it is
    safe to call again. A colormap that fails to load is logged as an error and
    the others are still registered.
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
    Check whether a name is a registered Matplotlib colormap.

    Parameters
    ----------
    name : str
        Colormap name. Matplotlib's own and playNano's colormaps are both
        registered.

    Returns
    -------
    bool
        True if ``name`` is a string naming a registered colormap.
    """
    if not isinstance(name, str):
        return False

    return name in plt.colormaps()


def resolve_cmap(cmap: str) -> str:
    """
    Return a colormap name that can be used, falling back to the default.

    For names given by users, for example on the command line: an unknown
    name gives a logged warning and ``"afm_brown"`` (``DEFAULT_CMAP``) instead
    of an error.

    Parameters
    ----------
    cmap : str
        Requested colormap name.

    Returns
    -------
    str
        ``cmap`` if it is a registered colormap, otherwise ``"afm_brown"``.

    See Also
    --------
    is_valid_cmap : Check a name without a fallback.

    Examples
    --------
    >>> from playnano.utils.colormaps import resolve_cmap
    >>> resolve_cmap("viridis")
    'viridis'
    >>> resolve_cmap("not_a_colormap")
    'afm_brown'
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
    Return the names of all registered colormaps, sorted.

    Returns
    -------
    list of str
        Matplotlib's and playNano's colormaps, including reversed ``_r``
        versions.
    """
    return sorted(plt.colormaps())
