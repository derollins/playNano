"""Utility to attach parameter-condition functions to processing callables."""

from typing import Callable


def param_conditions(**conds: Callable[[dict], bool]):
    """
    Decorate a processing or analysis callable with parameter-conditions.

    Each keyword is the parameter name and the value is a callable that will be
    called with the currently collected kwargs (a dict) and must return True if
    that parameter should be asked/present, False otherwise.

    Uses example:
    @param_conditions(hole_area=lambda p: p.get("fill_holes", False)
    def run()
    """

    def decorator(fn: Callable) -> Callable:
        fn._param_conditions = conds
        return fn

    return decorator


def prune_kwargs(fn, kwargs):
    """Remove inactive or None-valued parameters from kwargs."""
    cleaned = {}
    conds = getattr(fn, "_param_conditions", {})

    for name, val in kwargs.items():
        if val is None:
            continue
        cond = conds.get(name)
        if cond is not None:
            try:
                if not cond(kwargs):
                    continue
            except Exception:
                continue
        cleaned[name] = val

    return cleaned
