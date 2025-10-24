"""
Module for resolving and instantiating analysis modules.

Handles lookup in the built-in registry and via entry points
registered under the group 'playNano.analysis'.
"""

from __future__ import annotations

import importlib.metadata

from playNano.analysis import BUILTIN_ANALYSIS_MODULES
from playNano.analysis.base import AnalysisModule


def load_analysis_module(name: str) -> AnalysisModule:
    """
    Load and instantiate an AnalysisModule by name.

    Parameters
    ----------
    name : str
        Name of the analysis module to load.

    Returns
    -------
    AnalysisModule
        Instantiated module.

    Raises
    ------
    ValueError
        If the module cannot be found.
    TypeError
        If the loaded object is not a subclass of AnalysisModule.
    """
    cls = BUILTIN_ANALYSIS_MODULES.get(name)
    if cls is None:
        eps = importlib.metadata.entry_points().select(
            group="playNano.analysis", name=name
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
