"""Public package initialization."""

import importlib
import importlib.abc
import importlib.util
import sys
import warnings

# --- Back-compat import shim: map 'playNano' -> 'playnano' for all subpackages ---
# This avoids needing a parallel 'playNano' folder on case-insensitive filesystems.
# Keep for one deprecation cycle, then remove.


class _PlayNanoAliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    _old = "playNano"
    _new = "playnano"
    _warned = False  # process-wide, one-shot

    def find_spec(self, fullname, path, target=None):
        # Map 'playNano' and 'playNano.*' to 'playnano' equivalents
        if fullname == self._old or fullname.startswith(self._old + "."):
            mapped = self._new + fullname[len(self._old) :]
            real_spec = importlib.util.find_spec(mapped)
            if real_spec is None:
                return None
            # Create a spec for the *old* name that we will populate from the mapped
            # module
            spec = importlib.util.spec_from_loader(
                fullname, self, origin=real_spec.origin
            )
            # Preserve package-ness and package search locations (for subpackages)
            spec.submodule_search_locations = real_spec.submodule_search_locations
            # Store the mapping so exec_module can import the real one
            spec._mapped = mapped  # type: ignore[attr-defined]
            return spec
        return None

    def create_module(self, spec):
        # Use default module creation semantics
        return None

    def exec_module(self, module):
        mapped = module.__spec__._mapped  # type: ignore[attr-defined]
        real = importlib.import_module(mapped)

        # Emit the deprecation once either when top-level is imported,
        # or if a submodule is imported and we haven't warned yet.
        if module.__name__ == self._old or (
            module.__name__.startswith(self._old + ".") and not self._warned
        ):
            warnings.warn(
                "Importing 'playNano' (mixed case) is deprecated and will be removed in a future release. "  # noqa
                "Please import 'playnano' (lowercase) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            _PlayNanoAliasFinder._warned = True

        # Copy attributes and register the alias
        module.__dict__.update(real.__dict__)
        sys.modules[module.__name__] = module


# Put our alias finder at the front so it wins before others
if not any(isinstance(h, _PlayNanoAliasFinder) for h in sys.meta_path):
    sys.meta_path.insert(0, _PlayNanoAliasFinder())
# -------------------------------------------------------------------------------

try:
    from ._version import (  # type: ignore[import]; written by hatch-vcs at build/install time noqa
        __version__,
    )
except Exception:
    __version__ = "0+unknown"


__all__ = ["__version__"]
