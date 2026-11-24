"""
The built-in analysis modules.

All submodules are imported with the package. Each defines one subclass of
:class:`~playnano.analysis.base.AnalysisModule`.
"""

import importlib
import pkgutil

# Discover all submodules in this package
__all__ = [str(m.name) for m in pkgutil.iter_modules(__path__)]

# Import all submodules so they are available under the package namespace
for module_name in __all__:
    try:
        importlib.import_module(f".{module_name}", package=__name__)
    except ImportError as e:
        raise ImportError(
            f"Failed to import analysis module '{module_name}': {e}"
        ) from e
