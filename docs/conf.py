import importlib
import os
import pkgutil
import subprocess
import sys

import playNano.analysis.modules as modules

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# -- Project info --------------------------------------------------
project = "playNano"
author = "Daniel E. Rollins"
copyright = "2025, Daniel E. Rollins"

# -- Version and release -----------------------------------------------------
# Pull version from environment variable set by GitHub Actions
# Default to 'latest' if building locally
version_env = os.environ.get("VERSION", "latest")

if version_env != "latest":
    # Tagged release
    version = version_env
    release = version_env
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
        release = f"{version}+{commit}"
    except Exception:
        pass
else:
    # Main branch or local
    version = "latest"
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
        release = f"{version}+{commit}"
    except Exception:
        release = version

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinxcontrib.programoutput",
    "nbsphinx",
    "myst_parser",
]

exclude_patterns = []

autosummary_generate = True

extensions.append("sphinx_multiversion")

# Optional: Configure which branches/tags to include
smv_tag_whitelist = r"^v\d+\.\d+.*$"
smv_branch_whitelist = r"^(main|dev)$"
smv_remote_whitelist = r"^origin$"

# Mock imports for modules that may not be installed
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

# -- HTML output options -----------------------------------------------------
html_theme = "furo"


# Make sure Sphinx knows where your templates & static files live
templates_path = ["_templates"]  # ← don't forget this line
html_static_path = ["_static"]

# Load your switcher files (filenames are relative to _static/)
html_js_files = ["version-switcher.js"]
# optional
html_css_files = ["version-switcher.css"]

# Ensure the sidebar template path matches your file location
html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/navigation.html",
        "sidebar/scroll-start.html",
        "sidebar/versions.html",  # ← must exist at docs/_templates/sidebar/versions.html
        "sidebar/scroll-end.html",
    ]
}

# ---------------------------------------------------------------------------
# Automatically generate the module list and autosummary stubs
# ---------------------------------------------------------------------------
module_names = [name for _, name, _ in pkgutil.iter_modules(modules.__path__)]
autosummary_list = "\n   ".join(
    f"playNano.analysis.modules.{name}" for name in module_names
)

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
        f.write(f"- `{name} <{link}>`_  \n")
        if summary:
            f.write(f"  - {summary}\n")

# -- Nitpick ignore ------------------------------------------------
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

# Intersphinx mapping lets Sphinx resolve external references in our docstrings
# (e.g. numpy arrays, pandas DataFrames, matplotlib Axes, Qt types) and turn
# them into links to the official documentation of those projects.
# This avoids a flood of "reference not found" warnings and gives users
# clickable cross-references in the generated HTML.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "qt": ("https://doc.qt.io/qtforpython-6/", None),
}
