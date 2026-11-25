"""
Mark which parameters of a processing or analysis function apply when.

The command line uses these conditions to skip questions and drop settings
that do not apply, for example a hole size when holes are not filled.
"""

from typing import Callable


def param_conditions(**conds: Callable[[dict], bool]):
    """
    Attach conditions saying when a function's parameters apply.

    Parameters
    ----------
    **conds : callable
        One per parameter name: a function taking the dict of parameters given
        so far and returning True if this parameter applies.

    Returns
    -------
    callable
        A decorator that stores ``conds`` on the function as
        ``_param_conditions`` and returns the function unchanged.

    See Also
    --------
    prune_kwargs : Remove parameters that do not apply.

    Examples
    --------
    >>> from playnano.utils.param_utils import param_conditions
    >>> @param_conditions(hole_area=lambda p: p.get("fill_holes", False))
    ... def mask_fn(frame, fill_holes=False, hole_area=None):
    ...     return frame
    >>> sorted(mask_fn._param_conditions)
    ['hole_area']
    """

    def decorator(fn: Callable) -> Callable:
        fn._param_conditions = conds
        return fn

    return decorator


def prune_kwargs(fn, kwargs):
    """
    Remove parameters that are None or whose condition is not met.

    Parameters
    ----------
    fn : callable
        Function the parameters are for, possibly decorated with
        :func:`param_conditions`.
    kwargs : dict
        Parameters. Not changed.

    Returns
    -------
    dict
        The parameters that are not None and whose condition, if any, returns
        True for ``kwargs``. A condition that raises an error counts as False.

    Examples
    --------
    >>> from playnano.utils.param_utils import param_conditions, prune_kwargs
    >>> @param_conditions(hole_area=lambda p: p.get("fill_holes", False))
    ... def mask_fn(frame, fill_holes=False, hole_area=None, sigma=None):
    ...     return frame
    >>> prune_kwargs(mask_fn, {"fill_holes": False, "hole_area": 10, "sigma": None})
    {'fill_holes': False}
    """
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
