"""Module contianing helper funcitons for collecting system info"""

import importlib.metadata
import platform
import sys
from datetime import UTC, datetime

KEY_DEPENDENCIES = ["numpy", "h5py", "scipy", "opencv-python", "scikit-image", "pandas"]


def gather_environment_info() -> dict:
    info = {
        "timestamp": datetime.now(UTC).isoformat() + "Z",
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
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
