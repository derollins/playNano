import importlib
import os
import sys
from pathlib import Path

# ------------------------------------------------------------------------------
# Path Setup (Make repo importable in all scenarios)
# ------------------------------------------------------------------------------
conf_dir = os.path.abspath(os.path.dirname(__file__))
repo_root = os.path.abspath(os.path.join(conf_dir, ".."))
src_path = os.path.join(repo_root, "src")

for p in (src_path, repo_root):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

# ------------------------------------------------------------------------------
# Project Info
# ------------------------------------------------------------------------------
project = "playnano"
author = "Daniel E. Rollins"
copyright = "2026, Daniel E. Rollins"
version = ""
release = ""

# ------------------------------------------------------------------------------
# Extensions & Theme
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

html_theme = "furo"
templates_path = ["_templates"]
html_static_path = ["_static"]
html_js_files = ["version-switcher.js"]
html_css_files = ["version-switcher.css"]

# Multiversion Selection
smv_tag_whitelist = r"^v\d+\.\d+.*$"
smv_branch_whitelist = r"^(main|dev)$"
smv_remote_whitelist = r"^$"

# ------------------------------------------------------------------------------
# Helper Functions (Defined before they are used in setup)
# ------------------------------------------------------------------------------


def run_apidoc(_):
    """Automatically generates the API RST files from source code."""
    from sphinx.ext.apidoc import main

    output_path = os.path.join(conf_dir, "api")
    module_path = os.path.join(repo_root, "src", "playnano")

    # -e: put each module on its own page
    # --force: overwrite existing files
    # --no-toc: don't overwrite your custom modules.rst
    main(["-e", "-o", output_path, module_path, "--force", "--no-toc"])


def _set_title_and_version(app, config):
    """Detects version from environment or SMV context and sets html_title."""
    ctx = config.html_context or {}
    v = None
    for key in ("current_version", "smv_current_version"):
        cur = ctx.get(key)
        if cur:
            v = getattr(cur, "name", None) or (
                cur.get("name") if isinstance(cur, dict) else None
            )
            if v:
                break
    if not v:
        try:
            out_name = Path(app.outdir).name
            if out_name and out_name not in ("html",):
                v = out_name
        except:
            pass
    if not v:
        v = os.environ.get("VERSION", "")

    v_norm = "latest" if v in ("", None, "main", "latest") else v
    config.version = v_norm
    config.release = v_norm
    config.html_title = f"{config.project} {v_norm} documentation"
    ctx["version_label"] = v_norm
    config.html_context = ctx


def _discover_analysis_module_names():
    """
    Discover playnano.analysis.modules.* by scanning the source tree.

    Works even if the package cannot be imported.
    """
    candidates = []
    for pkg in ("playnano", "playNano"):
        base = Path(src_path) / pkg / "analysis" / "modules"
        if base.is_dir():
            candidates.extend(
                [p.stem for p in base.glob("*.py") if p.name != "__init__.py"]
            )
    return sorted(set(candidates))


def _try_import_pkg():
    """
    Try importing either 'playnano' or 'playNano', return (module, import_name) or (None, None).
    """
    for name in ("playnano", "playNano"):
        try:
            return importlib.import_module(name), name
        except:
            continue
    return None, None


def _write_generated_module_list(module_names, import_name):
    if not module_names:
        return
    path = Path("_generated") / "generated_module_list.rst"
    path.parent.mkdir(parents=True, exist_ok=True)
    rel_api = os.path.relpath(Path("html") / "api", path.parent).replace(os.sep, "/")

    with path.open("w", encoding="utf-8") as f:
        for name in module_names:
            link = f"{rel_api}/playnano.analysis.modules.html#module-playnano.analysis.modules.{name}"
            summary = "No description available."
            if import_name:
                try:
                    mod = importlib.import_module(
                        f"{import_name}.analysis.modules.{name}"
                    )
                    summary = (mod.__doc__ or "").strip().splitlines()[0]
                except:
                    pass
            f.write(f"- `{name} <{link}>`_\n  - {summary}\n")


# ------------------------------------------------------------------------------
# The Single setup(app) Hook
# ------------------------------------------------------------------------------


def setup(app):
    # 1. Generate API documentation automatically
    app.connect("builder-inited", run_apidoc)

    # 2. Generate the Dynamic Analysis Module list
    def _on_builder_inited(_app):
        names = _discover_analysis_module_names()
        mod, import_name = _try_import_pkg()
        _write_generated_module_list(names, import_name)

    app.connect("builder-inited", _on_builder_inited)

    # 3. Set dynamic titles and versions
    app.connect("config-inited", _set_title_and_version, priority=900)


# ------------------------------------------------------------------------------
# Nitpick & Intersphinx (Keep your existing full lists here)
# ------------------------------------------------------------------------------
nitpick_ignore = [("py:class", "np.ndarray"), ("py:class", "Path")]  # etc...
intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}
