"""Module for loading and registering custom colormaps for AFM data visualisation."""

import logging
from importlib import resources

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

logger = logging.getLogger(__name__)

DEFAULT_CMAP = "afm_brown"


def register_custom_colormaps():
    """
    Load colormap data from CSV and registers them with Matplotlib for global access.

    The colormaps are defined in CSV files within the package resources and are
    registered under specific names. Both the original and reversed versions are
    registered.
    """
    # Locates the files inside src/playnano/resources/colormaps/
    # This works even if the package is installed as a zip or site-package
    resource_path = resources.files("playnano.resources.colormaps")

    cmaps_to_load = {
        "playnano_gold": "playnano_gold.csv",
        "afm_brown": "afm_brown.csv",
        "classic_afm": "classic_afm.csv",
    }

    for name, filename in cmaps_to_load.items():
        try:
            path = resource_path / filename
            # Load the 256x3 RGB array
            rgb_data = np.loadtxt(path, delimiter=",")

            if name not in plt.colormaps():
                new_cmap = ListedColormap(rgb_data, name=name)
                plt.colormaps.register(new_cmap)
                # Register reversed version (standard in AFM analysis)
                plt.colormaps.register(new_cmap.reversed(), name=f"{name}_r")
                logger.debug(
                    f"Registered colormap '{name}' and '{name}_r' from {filename}"
                )
        except Exception as e:
            import logging

            logging.getLogger(__name__).error(f"Failed to load colormap {name}: {e}")


def is_valid_cmap(name: str) -> bool:
    """
    Check whether a colormap name is valid.

    This includes:
    - Built-in Matplotlib colormaps
    - Custom playNano colormaps (after registration)

    Parameters
    ----------
    name : str
        Name of the colormap.

    Returns
    -------
    bool
        True if the colormap exists, False otherwise.
    """
    if not isinstance(name, str):
        return False

    return name in plt.colormaps()


def get_available_cmaps() -> list[str]:
    """Return a sorted list of available colormap names."""
    return sorted(plt.colormaps())
