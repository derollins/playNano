# Contributing to playNano

Thank you for your interest in contributing to **playNano**! Contributions of all kinds
are welcome — bug reports, documentation improvements, new features, and plugins.

Please read this guide before submitting a pull request.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Running the Tests](#running-the-tests)
- [Code Style and Pre-commit](#code-style-and-pre-commit)
- [Type Hints](#type-hints)
- [Docstrings](#docstrings)
- [Branching and Pull Requests](#branching-and-pull-requests)
- [Changelog](#changelog)
- [Versioning](#versioning)
- [Writing Analysis Modules](#writing-analysis-modules)
- [Writing Processing Plugins](#writing-processing-plugins)
- [Reporting Bugs and Requesting Features](#reporting-bugs-and-requesting-features)
- [AI Transparency](#ai-transparency)

---

## Getting Started

### Prerequisites

- [Anaconda or Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Python 3.10–3.12

### Setting up a development environment

```bash
git clone https://github.com/derollins/playNano.git
cd playNano
conda create -n playnano-dev python=3.11
conda activate playnano-dev
pip install -e ".[dev]"
```

This installs playNano in editable mode along with all development dependencies including testing and pre-commit tools.

To also install the optional notebook dependencies:

```bash
pip install -e ".[dev,notebooks]"
```

---

## Project Structure

```txt
playNano/
├── src/playnano/
│   ├── afm_stack.py       # Core data class (AFMImageStack) — all pipelines operate on this.
│   ├── processing/        # Processing pipeline, 2D filters, 3D stack filters, masks, and stack edits.
│   ├── analysis/          # Analysis pipeline, built-in modules, and utilities.
│   ├── io/
│   │   └── formats/       # File format readers (asd, jpk, spm, h5)
│   ├── cli/               # Command-line interface and interactive wizard
│   └── gui/               # Qt-based viewer
├── tests/                 # Test suite
├── docs/                  # Sphinx documentation
├── notebooks/             # Demo notebooks
├── CHANGELOG.md
└── pyproject.toml
```

### Key design concepts

**`AFMImageStack`** is the central data class. It holds raw and processed frame data,
metadata, provenance, masks, and analysis results. Both pipelines operate on it.
Raw AFM data is loaded into this class and the raw image data is held alongside
timestamp information and any processing or analysis outputs.

**Processing** transforms the data in `AFMImageStack.data`. Steps are resolved by
`_resolve_step()` which checks — in priority order — masks, bound methods, 2D plugins,
3D plugins, built-in filters, video filters, and stack edits. `ProcessingPipeline`
handles provenance recording and step sequencing.

**Analysis** reads from `AFMImageStack` and writes results into
`AFMImageStack.analysis_results`. Each analysis module is a self-contained class
inheriting from `AnalysisModule`. `AnalysisPipeline` handles sequencing and passes
results between dependent modules via `previous_results`.

**Plugins** extend either pipeline without modifying playNano source, via Python entry
points (`playnano.filters`, `playnano.video_processing`, `playnano.analysis`).

---

## Running the Tests

Tests are written with [pytest](https://pytest.org). To run the full test suite from the project root:

```bash
pytest .
```

To run a specific test file:

```bash
pytest tests/test_analysis_modules.py
```

All tests must pass before submitting a pull request. If you are adding new functionality,
please include tests covering the new behaviour. Tests are located in the `tests/` folder
in the repository root and are generally organised by subpackage, although some larger
modules have their own dedicated test file.

---

## Code Style and Pre-commit

playNano uses [pre-commit](https://pre-commit.com) to enforce consistent code style and catch
common issues. The hooks run automatically on commit once installed:

```bash
pre-commit install
```

To run the hooks manually against all files:

```bash
pre-commit run --all-files
```

The pre-commit configuration pins Python to 3.11. If you are on a different version, the hooks
will still run but this is the version used in CI.

You do not need to manually configure formatting — the hooks handle this. If a hook modifies a
file, stage the changes and commit again.

Beyond what pre-commit enforces, please follow these general conventions:

- Prefer explicit over implicit; avoid ambiguous variable names and hidden state changes
- Keep functions focused; extract helpers rather than nesting deeply
- Log significant internal decisions using the module-level `logger`
  (`logging.getLogger(__name__)`) rather than printing to stdout
- Use `# noqa` comments sparingly and always with a specific rule code (e.g. `# noqa: E501`)

---

## Type Hints

All new code should include type hints. playNano targets Python 3.10+, so built-in generics
(`list[int]`, `dict[str, Any]`) can be used directly. Import `Optional`, `Sequence`,
`Callable`, and similar from `typing` where needed.

```python
# Good
def process_frame(frame: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    ...

# Good — use Optional for arguments that can be None
def run(
    self,
    stack: AFMImageStack,
    previous_results: dict[str, Any] | None = None,
    mask_key: Optional[str] = None,
) -> dict[str, Any]:
    ...
```

Return types should always be annotated. For functions that return nothing, annotate
`-> None` explicitly. Type hints on private helpers (`_` prefix) are encouraged but
slightly less strictly required than on public API.

---

## Docstrings

playNano uses **NumPy-style docstrings** for all publicly exposed functions, methods, and
classes. **Every function, method, and class must have at least one line of documentation**
— undocumented code will not be accepted.

### One-line docstrings

Used for private helpers and simple functions where the signature alone is not self-explanatory:

```python
def _normalize_pad(pad) -> tuple:
    """Normalize pad argument to (top, bottom, left, right)."""
    ...
```

### NumPy-style docstrings

Used for all public API — any function, method, or class that is part of the user-facing
or developer-facing interface:

```python
def flatten_particle_features(
    features_per_frame: list,
    grouping_output: dict,
    object_key: Optional[str] = None,
) -> list[dict]:
    """
    Combine per-frame feature statistics with particle grouping results.

    Merges the output of a detection module with the output of a grouping
    module (tracks or clusters) into a flat list of row dicts suitable for
    conversion to a DataFrame.

    Parameters
    ----------
    features_per_frame : list[list[dict]]
        Per-frame feature dicts as returned by a detection module.
    grouping_output : dict
        Output of a tracking or clustering module containing object groups.
    object_key : str, optional
        Key in grouping_output listing the grouped objects. If None,
        auto-detected from ``"tracks"`` or ``"clusters"``.

    Returns
    -------
    list[dict]
        Flat list of row dicts, one per detection, with keys including
        frame, timestamp, label, centroid_x, centroid_y, area, mean, min, max.

    Raises
    ------
    KeyError
        If object_key is not found in grouping_output.

    Examples
    --------
    >>> rows = flatten_particle_features(features, tracking_result)
    >>> df = pd.DataFrame(rows)
    """
```

### Sections to include

Include the following sections as applicable:

| Section | When to include |
| --- | --- |
| Summary (first line) | Always — one line, imperative mood |
| Extended description | When the summary alone is insufficient |
| `Parameters` | Any function that takes arguments |
| `Returns` | Any function that returns a value |
| `Raises` | When specific exceptions are raised intentionally |
| `See Also` | When related functions or classes exist |
| `Notes` | For algorithmic detail, caveats, or references |
| `Examples` | For public API; optional for simple cases |

For **module-level docstrings**, follow the pattern established in `feature_detection.py`
and `particle_tracking.py`: a one-line summary, an extended description, a `See Also`
section linking related modules, a `.. versionadded::` directive, and an `Author` section.

---

## Branching and Pull Requests

- **Base all pull requests on `dev`**, not `main`. The `main` branch is reserved for releases.
- **Urgent** bug fixes and documentation changes can be merged directly into main under cirtain circumstaces.
- Use a descriptive branch name, e.g. `feature/morph-opening` or `fix/tracking-none-index`.
- Keep pull requests focused — one feature or fix per PR where possible.
- CI (tests and pre-commit) runs automatically on pull requests to both `dev` and `main`.

### Pull request checklist

- [ ] Tests pass locally (`pytest .`)
- [ ] Pre-commit hooks pass (`pre-commit run --all-files`)
- [ ] New functionality is covered by tests
- [ ] Docstrings are present and follow NumPy style for all public API
- [ ] At least a one-line docstring is present on every function and method
- [ ] Type hints are included on all new functions and methods
- [ ] A changelog entry has been added (see below)
- [ ] AI tool usage is disclosed if applicable (see [AI Transparency](#ai-transparency))

---

## Changelog

playNano maintains a [CHANGELOG.md](CHANGELOG.md) in the repo root following the
[Keep a Changelog](https://keepachangelog.com) format, and a `docs/whats_new/` directory
for longer release notes.

When submitting a pull request, please add a brief entry under the relevant section
(`Added`, `Changed`, `Fixed`, `Documentation`) in the `Unreleased` section of `CHANGELOG.md`.
If your change is a **breaking change**, make sure it is clearly noted under `Changed`
with a migration guide.

---

## Versioning

playNano uses **VCS-based versioning** — the package version is derived automatically from
Git tags. **Do not manually edit version numbers** in any file. Releases are made by tagging
`main` with a version string (e.g. `v0.3.0`) after merging from `dev`.

Individual processing filters and analysis modules carry their own version numbers via the
`@versioned_filter` decorator or the `version` class attribute. These should be updated when
the behaviour of that specific component changes.

---

## Writing Analysis Modules

playNano's analysis pipeline is modular and extensible. New analysis modules can be
contributed in two ways:

**Directly to playNano** — add the module to `src/playnano/analysis/modules/` following
the structure of existing modules, with corresponding tests in `tests/test_analysis_modules.py`.
This is appropriate for general-purpose modules with no heavy additional dependencies.

**Via the plugins repository** — a collection of playNano plugins (both processing and
analysis modules) is maintained at
[playNano-plugins](https://github.com/derollins/playNano-plugins). If your proposed module
has heavy or specialised dependencies, or is domain-specific, this is the better starting
point. Plugins can be installed into any playNano environment without modifying the core
package.

Full documentation is available in the
[custom analysis modules guide](https://derollins.github.io/playNano/custom_analysis_modules.html).

The key requirements for any analysis module are:

- Subclass `playnano.analysis.base.AnalysisModule`
- Implement the `name` property and the `run()` method
- Declare upstream dependencies using the `requires` class attribute if your module depends
  on the output of another module
- Set a `version` class attribute and update it when the module behaviour changes
- Include a NumPy-style class docstring covering parameters, returns, raises, and an example

---

## Writing Processing Plugins

Custom processing functions can be contributed in two ways:

**Directly to playNano** — add the function to the appropriate module in
`src/playnano/processing/` (`filters.py`, `video_processing.py`, or `mask_generators.py`)
and register it in the corresponding map (`FILTER_MAP`, `VIDEO_FILTER_MAP`, `MASK_MAP`).
Decorate it with `@versioned_filter` and include a NumPy-style docstring. This is
appropriate for general-purpose operations with no additional dependencies.

**Via the plugins repository** — processing functions can also live in
[playNano-plugins](https://github.com/derollins/playNano-plugins) and be installed
independently via Python entry points. Two entry point groups are supported:

- `playnano.filters` — for **2D frame operations** (frame → frame)
- `playnano.video_processing` — for **3D stack operations** (stack → stack)

Full documentation including `pyproject.toml` examples is available in the
[processing documentation](https://derollins.github.io/playNano/processing.html).

---

## Reporting Bugs and Requesting Features

Please use [GitHub Issues](https://github.com/derollins/playNano/issues) to report bugs or
request features.

When reporting a bug, please include:

- Your operating system and Python version
- The playNano version (`pip show playnano`)
- A minimal reproducible example if possible
- The full traceback if an exception was raised

For feature requests, a brief description of the use case is helpful alongside the proposed
behaviour.

---

## AI Transparency

AI-based tools may be used during development for tasks such as typing assistance, formatting,
debugging and refactoring suggestions, and documentation drafting.

Any use of AI tools in contributions to playNano should be disclosed in one of the following
ways:

- A note in the module-level docstring of any file where AI tools contributed substantially
  to the logic or structure (see existing examples in `feature_detection.py` and
  `particle_tracking.py`)
- A note in the pull request description

AI-generated code must be reviewed, tested, and validated by the contributor before
submission. The contributor is responsible for the correctness of all submitted code
regardless of how it was produced.
