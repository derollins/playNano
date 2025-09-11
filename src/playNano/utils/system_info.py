"""Module contianing helper funcitons for collecting system info."""

import importlib.metadata
import platform
import sys

from playNano.utils.time_utils import utc_now_iso

KEY_DEPENDENCIES = [
    "numpy",
    "h5py",
    "scipy",
    "scikit-learn",
    "scikit-image",
    "pandas",
    "Pillow",
    "matplotlib",
    "afmreader",
]


def gather_environment_info() -> dict:
    """Gather the system info for provenance records."""
    info = {
        "timestamp": utc_now_iso(),
        "python_version": sys.version.replace("\n", " "),
        "platform": f"{platform.system()}-{platform.release()}-{platform.machine()}",
    }
    try:
        info["playNano_version"] = importlib.metadata.version("playNano")
    except importlib.metadata.PackageNotFoundError:
        info["playNano_version"] = None

    for pkg in KEY_DEPENDENCIES:
        try:
            info[f"{pkg}_version"] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            pass
    return info
