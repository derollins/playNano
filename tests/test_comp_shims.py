"""Tests for the compatability shim."""

import warnings
import sys
import importlib


def test_uppercase_package_still_imports():
    """Test that the playNano still imports with shim."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", DeprecationWarning)
        import playNano  # noqa: F401  # type: ignore[reportMissingImports]

        assert any(isinstance(x.message, DeprecationWarning) for x in w)


def test_uppercase_subpackages_still_import_if_used():
    """Test that subpackages import with playNano with shim."""
    # Reset shim state for this process so we can observe the warning in this test
    playnano_pkg = importlib.import_module("playnano")
    # Access the module object that actually defines _PlayNanoAliasFinder
    # playnano.__spec__.loader is not our alias finder, so we reach into the package
    alias_finder_cls = getattr(playnano_pkg, "_PlayNanoAliasFinder", None)
    if alias_finder_cls is not None:
        alias_finder_cls._warned = False  # reset one-shot warning gate

    # Also remove any cached 'playNano' alias modules from sys.modules so the
    # import path will run through the alias loader again
    for name in [
        m for m in list(sys.modules) if m == "playNano" or m.startswith("playNano.")
    ]:
        del sys.modules[name]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", DeprecationWarning)

        from playNano import (  # noqa: F401  # type: ignore[reportMissingImports]
            processing,
        )

        assert any(isinstance(x.message, DeprecationWarning) for x in w)
