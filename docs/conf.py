import importlib
import os
import pkgutil
import sys

import playNano.analysis.modules as modules

sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------
project = "playNano"
copyright = "2025, Daniel E. Rollins"
author = "Daniel E. Rollins"
release = "0.1.0"

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

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
# html_static_path = ["_static"]

# ---------------------------------------------------------------------------
# Automatically generate the module list and autosummary stubs
# ---------------------------------------------------------------------------

# List all modules in playNano.analysis.modules
module_names = [name for _, name, _ in pkgutil.iter_modules(modules.__path__)]
autosummary_list = "\n   ".join(
    f"playNano.analysis.modules.{name}" for name in module_names
)

# Paths for the generated RST files
generated_list_path = "_generated/generated_module_list.rst"
os.makedirs(os.path.dirname(generated_list_path), exist_ok=True)


# Path to the html api folder
api_folder = os.path.abspath("html/api")

# Compute the relative path from the RST file to the API folder
rel_api_folder = os.path.relpath(api_folder, os.path.dirname(generated_list_path))

# Create bulleted list instead of autosummary table
module_names = [name for _, name, _ in pkgutil.iter_modules(modules.__path__)]

with open(generated_list_path, "w", encoding="utf-8") as f:
    for name in module_names:
        full_name = f"playNano.analysis.modules.{name}"
        module_html = "playNano.analysis.modules.html"
        anchor = f"#module-playNano.analysis.modules.{name}"
        link = os.path.join(rel_api_folder, module_html) + anchor
        # Normalize to forward slashes for Sphinx links
        link = link.replace(os.sep, "/")
        try:
            mod = importlib.import_module(full_name)
            # Get first line of module docstring
            summary = (mod.__doc__ or "").strip().splitlines()[0]
        except Exception:
            summary = "No description available."

        # Write as bullet with link and optional description
        f.write(f"- `{name} <{link}>`_")

        if summary:
            f.write(f"  - {summary}\n")
