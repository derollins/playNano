"""Module for applying verision numbes to filters and masks."""


def versioned_filter(version: str):
    """
    Add decorator to assign a version number to a filter or mask function.

    This adds a ``__version__`` attribute to the decorated function, which can
    be used for provenance tracking or documentation purposes.

    Parameters
    ----------
    version : str
        The version string to assign to the function, e.g., "1.0.0".

    Returns
    -------
    callable
        The decorated function with a ``__version__`` attribute.

    Examples
    --------
    >>> @versioned_filter("1.0.0")
    ... def gaussian_filter(frame, sigma=1.0):
    ...     ...
    >>> gaussian_filter.__version__
    '1.0.0'
    """

    def decorator(fn):
        fn.__version__ = version
        return fn

    return decorator
