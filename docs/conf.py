import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "playNano"
copyright = "2025, Daniel E. Rollins"
author = "Daniel E. Rollins"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # If you use NumPy or Google-style docstrings
    "sphinx.ext.viewcode",  # Optional: links to source code
    # "sphinx.ext.autosummary",  # Optional: for autosummary tables
]
