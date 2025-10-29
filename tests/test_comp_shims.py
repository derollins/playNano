import warnings


def test_uppercase_package_still_imports():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", DeprecationWarning)
        import playNano  # noqa: F401

        assert any(isinstance(x.message, DeprecationWarning) for x in w)


def test_uppercase_subpackages_still_import_if_used():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", DeprecationWarning)
        from playNano import processing  # noqa: F401

        assert any(isinstance(x.message, DeprecationWarning) for x in w)
