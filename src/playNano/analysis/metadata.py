"""Module contianing helper funcitons for building ananlysis metadata"""

import importlib.metadata
import platform
import sys
from datetime import UTC, datetime


def gather_environment_info() -> dict:
    """
    Collect environment information for provenance:
      - timestamp
      - python version
      - platform info
      - playNano version
      - key dependency versions (optional)
    """
    info = {}
    info["timestamp"] = datetime.now(UTC).isoformat() + "Z"
    info["python_version"] = sys.version.replace("\n", " ")
    info["platform"] = platform.platform()
    # playNano version:
    try:
        version = importlib.metadata.version("playNano")
    except importlib.metadata.PackageNotFoundError:
        version = None
    info["playNano_version"] = version
    # Optionally, record versions of key libs:
    for pkg in ("numpy", "h5py", "scipy", "opencv-python", "scikit-image", "pandas"):
        try:
            info[pkg + "_version"] = importlib.metadata.version(pkg)
        except Exception:
            pass
    return info
