"""Public package initialization."""

try:
    from ._version import (  # type: ignore[import]; written by hatch-vcs at build/install time noqa
        __version__,
    )
except Exception:
    __version__ = "0+unknown"


__all__ = ["__version__"]
