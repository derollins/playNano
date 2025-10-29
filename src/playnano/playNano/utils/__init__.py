"""Backward compatibility shim and warning"""

import warnings

warnings.warn(
    "Importing 'playNano' (mixed case) is deprecated and will be removed in a future release. "
    "Please import 'playnano' (lowercase) instead.",
    DeprecationWarning,
    stacklevel=2,
)
from playnano.utils import *  # noqa: F401,F403,E402
