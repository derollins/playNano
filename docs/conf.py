# docs/conf.py
import importlib
import os
import sys
from pathlib import Path

# ------------------------------------------------------------------------------
# Make repo importable in all scenarios (SMV temp checkouts, local, CI)
# ------------------------------------------------------------------------------
conf_dir = os.path.abspath(os.path.dirname(__file__))
repo_root = os.path.abspath(os.path.join(conf_dir, ".."))
src_path = os.path.join(repo_root, "src")

for p in (src_path, repo_root):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

print("DEBUG: sys.path head:", sys.path[:5])

# ------------------------------------------------------------------------------
# Project info
# ------------------------------------------------------------------------------
project = "playnano"
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


# ------------------------------------------------------------------------------
# Defer package-dependent work to build time (no top-level imports!)
# Also handle case-insensitive import of your package for summaries.
# ------------------------------------------------------------------------------
def _discover_analysis_module_names():
    """
    Discover playnano/playnano.analysis.modules.* by scanning the source tree.
    Works even if the package cannot be imported.
    """
    candidates = []

    # Prefer src/ layout
    for pkg_dirname in ("playnano", "playNano"):  # support both cases
        base = Path(src_path) / pkg_dirname / "analysis" / "modules"
        if base.is_dir():
            for p in base.glob("*.py"):
                if p.name != "__init__.py":
                    candidates.append(p.stem)

    # Fallback to non-src layout (older tags)
    for pkg_dirname in ("playnano", "playnano"):
        base = Path(repo_root) / pkg_dirname / "analysis" / "modules"
        if base.is_dir():
            for p in base.glob("*.py"):
                if p.name != "__init__.py":
                    candidates.append(p.stem)

    if not candidates:
        print("DEBUG: modules directory not found in expected locations.")
    return sorted(set(candidates))


def _try_import_pkg():
    """
    Try importing either 'playNano' or 'playnano', return (module, import_name) or (None, None).
    """
    for name in ("playNano", "playnano"):
        try:
            mod = importlib.import_module(name)
            return mod, name
        except Exception:
            continue
    return None, None


def _write_generated_module_list(module_names, import_name):
    """
    Write _generated/generated_module_list.rst linking to the API page anchors.
    """
    if not module_names:
        print("DEBUG: No analysis modules discovered; skipping generated list.")
        return

    generated_list_path = Path("_generated") / "generated_module_list.rst"
    generated_list_path.parent.mkdir(parents=True, exist_ok=True)

    api_folder = Path("html") / "api"
    rel_api_folder = os.path.relpath(api_folder, generated_list_path.parent)

    # Use the import name for anchors (so playNano vs playnano anchors are consistent)
    package_prefix = (
        import_name or "playNano"
    )  # default to historical anchor if unknown

    with generated_list_path.open("w", encoding="utf-8") as f:
        for name in module_names:
            module_html = f"{package_prefix}.analysis.modules.html"
            anchor = f"#module-{package_prefix}.analysis.modules.{name}"
            link = os.path.join(rel_api_folder, module_html) + anchor
            link = link.replace(os.sep, "/")
            summary = "No description available."

            # Try to import to get a 1-line summary; if it fails, keep placeholder.
            if import_name:
                try:
                    mod = importlib.import_module(
                        f"{import_name}.analysis.modules.{name}"
                    )
                    doc = (mod.__doc__ or "").strip().splitlines()
                    if doc:
                        summary = doc[0]
                except Exception as e:
                    print(f"DEBUG: Could not import {name} for summary: {e}")

            f.write(f"- `{name} <{link}>`_\n")
            if summary:
                f.write(f"  - {summary}\n")


def setup(app):
    # Generate the list late, when the builder is set up. This avoids conf.py
    # import-time failures and lets SMV import the config safely.
    def _on_builder_inited(_app):
        names = _discover_analysis_module_names()
        mod, import_name = _try_import_pkg()
        if mod:
            print(
                f"DEBUG: Imported package '{import_name}' from:",
                getattr(mod, "__file__", "<namespace>"),
            )
        else:
            print("DEBUG: Could not import package; proceeding without summaries.")
        _write_generated_module_list(names, import_name)

    app.connect("builder-inited", _on_builder_inited)
