import os
import sys
import pkgutil
import importlib

# ------------------------------------------------------------------------------
# Make repo importable in all scenarios (SMV temp checkouts, local, CI)
# ------------------------------------------------------------------------------
conf_dir = os.path.abspath(os.path.dirname(__file__))
repo_root = os.path.abspath(os.path.join(conf_dir, ".."))
src_path = os.path.join(repo_root, "src")

# Add both repo root and src/ to sys.path to support old/new layouts
for p in (src_path, repo_root):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

print("DEBUG: sys.path head:", sys.path[:5])

# ------------------------------------------------------------------------------
# Mock imports EARLY to prevent import-time failures in CI/SMV
# ------------------------------------------------------------------------------
autodoc_mock_imports = [
    "PySide2",
    "PySide6",
    "PyQt5",
    "PyQt6",
    "playNano.gui.main",
    "playNano.gui.window",
    "playNano.cli.actions",
    "playNano.cli.entrypoint",
    "playNano.cli.handlers",
    "shiboken6",
]

# In CI, be extra defensive and mock the package if needed
if os.environ.get("CI", "false").lower() == "true":
    autodoc_mock_imports += ["playNano", "playNano.analysis.modules"]

# ------------------------------------------------------------------------------
# Try to discover analysis submodules (optional)
# This must never crash. If import fails, we just skip listing.
# ------------------------------------------------------------------------------
module_names = []
try:
    import playNano  # noqa: F401

    try:
        import playNano.analysis.modules as modules  # noqa: F401

        module_names = [name for _, name, _ in pkgutil.iter_modules(modules.__path__)]
        print("DEBUG: discovered analysis modules:", module_names)
    except Exception as e:
        print(f"WARNING: Could not import playNano.analysis.modules: {e}")
except Exception as e:
    print(f"WARNING: playNano not importable: {e}")

# ------------------------------------------------------------------------------
# Project info
# ------------------------------------------------------------------------------
project = "playNano"
author = "Daniel E. Rollins"
copyright = "2025, Daniel E. Rollins"

# ------------------------------------------------------------------------------
# Versioning
# ------------------------------------------------------------------------------
version_env = os.environ.get("VERSION", "latest")
version = version_env
release = version_env

# ------------------------------------------------------------------------------
# Extensions
# ------------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinxcontrib.programoutput",
    "nbsphinx",
    "myst_parser",
    "sphinx_multiversion",
]

autosummary_generate = True
exclude_patterns = []

# Sphinx-Multiversion selection (adjust as needed)
smv_tag_whitelist = r"^v\d+\.\d+.*$"
smv_branch_whitelist = r"^(main|dev)$"
smv_remote_whitelist = r"^origin$"

# ------------------------------------------------------------------------------
# HTML theme and static files
# ------------------------------------------------------------------------------
html_theme = "furo"
templates_path = ["_templates"]
html_static_path = ["_static"]
html_js_files = ["version-switcher.js"]
html_css_files = ["version-switcher.css"]

html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/navigation.html",
        "sidebar/scroll-start.html",
        "sidebar/versions.html",
        "sidebar/scroll-end.html",
    ]
}

# ------------------------------------------------------------------------------
# Generated module list (only if we actually discovered modules)
# ------------------------------------------------------------------------------
if module_names:
    generated_list_path = "_generated/generated_module_list.rst"
    os.makedirs(os.path.dirname(generated_list_path), exist_ok=True)

    api_folder = os.path.abspath("html/api")
    rel_api_folder = os.path.relpath(api_folder, os.path.dirname(generated_list_path))

    with open(generated_list_path, "w", encoding="utf-8") as f:
        for name in module_names:
            full_name = f"playNano.analysis.modules.{name}"
            module_html = "playNano.analysis.modules.html"
            anchor = f"#module-playNano.analysis.modules.{name}"
            link = os.path.join(rel_api_folder, module_html) + anchor
            link = link.replace(os.sep, "/")
            try:
                mod = importlib.import_module(full_name)
                summary = (mod.__doc__ or "").strip().splitlines()[0]
            except Exception:
                summary = "No description available."
            f.write(f"- `{name} <{link}>`_\n")
            if summary:
                f.write(f"  - {summary}\n")

# ------------------------------------------------------------------------------
# Nitpick and intersphinx
# ------------------------------------------------------------------------------
nitpick_ignore = [
    ("py:class", "np.ndarray"),
    ("py:class", "numpy.ndarray"),
    ("py:class", "json.encoder.JSONEncoder"),
    ("py:class", "pd.DataFrame"),
    ("py:class", "lists"),
    ("py:class", "Axes"),
    ("py:class", "matplotlib Axes"),
    ("py:class", "matplotlib.axes._axes.Axes"),
    ("py:class", "QWidget"),
    ("py:class", "PySide6.QtWidgets.QWidget"),
    ("py:class", "QResizeEvent"),
    ("py:class", "PySide6.QtGui.QResizeEvent"),
    ("py:class", "QFont"),
    ("py:class", "PySide6.QtGui.QFont"),
    ("py:class", "QPaintEvent"),
    ("py:class", "h5py._hl.group.Group"),
    ("py:class", "Path"),
    ("py:class", "pathlib.Path"),
    ("py:class", "optional"),
    ("py:class", "callable"),
    ("py:class", "AnalysisOutputs"),
    ("py:class", "analysis_record"),
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "qt": ("https://doc.qt.io/qtforpython-6/", None),
}
