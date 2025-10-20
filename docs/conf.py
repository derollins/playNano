import importlib
import os
import pkgutil
import sys
import subprocess

import playNano.analysis.modules as modules

sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------
project = "playNano"
copyright = "2025, Daniel E. Rollins"
author = "Daniel E. Rollins"

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
    "sphinxcontrib.programoutput",
    "nbsphinx",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = []

autosummary_generate = True

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

# ---------------------------------------------------------------------------
# Version dropdown context for templates
# ---------------------------------------------------------------------------
html_build_dir = os.path.abspath("_build/html")
os.makedirs(html_build_dir, exist_ok=True)

# Scan all built versions
versions = [
    d
    for d in os.listdir(html_build_dir)
    if os.path.isdir(os.path.join(html_build_dir, d))
]
versions.sort(reverse=True)

html_context = {
    "versions": versions,
    "current_version": version,
}

# Sidebar: include the template for the dropdown
html_sidebars = {
    "**": [
        "sidebar/scroll-start.html",
        "sidebar/search.html",
        "sidebar/scroll-end.html",
        "version_selector.html",
    ]
}
