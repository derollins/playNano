"""Module for applying verision numbes to filters and masks."""


def versioned_filter(version: str):
    """
    Add decorator tag to a filter or mask function with a __version__ attribute.

    Usage:
        @versioned_filter("1.0.0")
        def gaussian_filter(frame, sigma=1.0):
            ...
    """

    def decorator(fn):
        fn.__version__ = version
        return fn

    return decorator
