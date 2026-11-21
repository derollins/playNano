"""
Find an analysis module by name and create it.

Built-in modules are looked up first, then plugins registered under the
``playnano.analysis`` entry point group.
"""

from __future__ import annotations

import importlib.metadata

from playnano.analysis import BUILTIN_ANALYSIS_MODULES
from playnano.analysis.base import AnalysisModule


def load_analysis_module(name: str) -> AnalysisModule:
    """
    Create the analysis module with the given name.

    Parameters
    ----------
    name : str
        The module's name, for example ``"feature_detection"``.

    Returns
    -------
    AnalysisModule
        A new instance of the module.

    Raises
    ------
    ValueError
        If no built-in module or plugin has this name.
    TypeError
        If the plugin is not an :class:`~playnano.analysis.base.AnalysisModule`.
    """
    cls = BUILTIN_ANALYSIS_MODULES.get(name)
    if cls is None:
        eps = importlib.metadata.entry_points().select(
            group="playnano.analysis", name=name
        )
        if not eps:
            raise ValueError(
                f"Analysis module '{name}' not found in registry or entry points"
            )
        cls = eps[0].load()

    instance = cls()
    if not isinstance(instance, AnalysisModule):
        raise TypeError(f"Loaded '{name}' is not a subclass of AnalysisModule")

    return instance
