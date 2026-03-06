<!-- markdownlint-disable MD033 MD024-->
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Processing Plugins**
  - Support for 3D stack plugins via the `playnano.video_processing` entry point,
    alongside the existing `playnano.filters` entry point for 2D frame plugins.
  - CLI and wizard now recognise and expose video plugins as a step type.

- **Feature Detection**
  - `morph_opening` and `sep_radius` parameters added to `FeatureDetectionModule`
    to optionally separate touching or overlapping particles using morphological opening.

- **Particle Tracking**
  - `max_missing` parameter: tracks can now persist across frames with missing
    detections (default: 0, preserving previous behaviour).
  - `distance_scale` parameter: distance threshold can now be scaled with time since
    last detection using `"constant"` (default), `"linear"`, or `"sqrt"` modes.

- **Video Processing**
  - `pad` parameter added to `intersection_crop` and `crop_square`, allowing outward
    expansion of the crop region beyond the finite-pixel intersection, filled with NaN.

- **CI**
  - Pre-commit and test workflows now also trigger on `dev` branch pull requests.
  - Test workflow supports `workflow_dispatch` for manual triggering.

- `CONTRIBUTING.md` covering development setup, project structure, code style,
  type hint and docstring requirements, branching workflow, and guidance on contributing
  analysis modules and processing plugins.

### Changed

- **Breaking: Particle Tracking output**
  - Track key renamed from `centroids` to `coords`.
  - Tracks now use a dense representation spanning first to last detection; frames
    with missing detections have `None` entries in `coords` and `point_indices`.
  - Migration: replace `track["centroids"]` with `track["coords"]` and handle
    `None` entries where `max_missing > 0`.

- **Breaking: `flatten_particle_features` output columns**
  - `mean_intensity`, `min_intensity`, `max_intensity` renamed to `mean`, `min`, `max`.
  - Migration: update any downstream code or saved pipelines referencing the old names.

- **Breaking: `intersection_crop` and `crop_square` metadata**
  - `bounds` key replaced by `intersection_bounds` (exclusive-end coordinates).
  - New keys added: `intersection_shape`, `requested_bounds`, `applied_pad`, `pad_param`.
  - Migration: replace `meta["bounds"]` with `meta["intersection_bounds"]`.

- **Internal API**
  - `_load_plugin` and `_get_plugin_version` removed from `AFMImageStack`; plugin
    loading is now handled entirely through `_resolve_step`.
  - `apply()` now snapshots results for all step types (video filters, stack edits,
    methods), not only 2D filters.

- **Tooling**
  - Pre-commit Python version pinned to 3.11; demo notebook kernels updated to 3.11.
  - `markdownlint-cli2` downgraded to v0.10.0 as a workaround for a CI dependency
    conflict (to be revisited in a future release).
  - Author name updated to "Daniel E. Rollins" in `pyproject.toml`.

### Fixed

- Numerous typos corrected across docstrings, inline comments, docs, and tests.
- CLI `SKIP_PARAM_NAMES` expanded to avoid exposing internal array parameter names
  (`imarray`, `array`, `frame`, `video`) in the interactive wizard.

### Documentation

- API reference restructured with captioned sections: Core, Analysis, Analysis Modules,
  Processing, IO, CLI, GUI.
- Module-level docstrings added/improved for `feature_detection` and
  `particle_tracking`, including `See Also` cross-references, version info, and
  AI transparency notes.
- `processing.rst` updated to document the `playnano.video_processing` plugin entry
  point and link to the plugins repository.
- Various improvements to quickstart, processing, and notebook documentation.

## [0.2.2]

### Fixed

- Resolved a bug where the `clear` processing step unintentionally replaced `stack.data`
  with an invalid or uninitialised array, causing corrupted or excessively large
  Z-values in processed frames. The `clear` step now correctly clears only the current
  mask and returns the current working array (`arr`) without modifying image data.
- Incorrect mapping of centroid_x and centroid_y in feature flattening logic. Centroids
  returned by regionprops were interpreted as (row, col) but incorrectly assigned as (x, y).
- RuntimeWarning caused by invalid value casting during Z-scaling normalization in GIF export.
  Improved robustness of GIF frame generation when AFM data contains missing or extreme Z-values.

### Changed

- Normalization step now uses np.nan_to_num and np.clip to ensure all values are within the valid
  range before casting.
- Updated centroid assignment:
  - `centroid_x = centroid[1] (column → x)`
  - `centroid_y = centroid[0] (row → y)`

### Added

- Safe normalization logic for GIF export in create_gif_with_scale_and_timestamp to handle NaN, inf,
  and out-of-range values before casting to np.uint8.
- Pytest coverage for:
  - Centroid coordinate mapping
  - Autodetection of object key (tracks vs clusters)
  - Handling of out-of-range indices
  - Column presence

## [0.2.1]

### Changed

- Package renames to all lowercase `playnano` to follow PEP8 and prevent cross-platform
 issues with case-sensitivity.
- The import path is now playnano (lowercase).
  Update your code: `import playnano` instead of `import playNano`.
- **Backward‑compatibility**: import playNano still works in v0.2.1 via a shim and emits
  a DeprecationWarning.
  The playNano path will be removed in a future release.
- CLI unchanged: Keep using playnano on the command line.
- Docs: Versioned documentation with main, stable, and tags; API reference updated
  to playnano.*.

## [0.2.0]

This release introduces video processing, stack editing, multi-version documentation, and major CLI enhancements.

### Added

- **Documentation**
  - New **Exporting Data** page (`docs/exporting.rst`) covering OME-TIFF, NPZ, HDF5, and GIF export formats
    with CLI and Python examples.
  - New **Processing Operations Reference** page (`docs/processing-operations-reference.rst`) listing all
    built-in filters, masks, and stack/video operations with parameters.
  - Added version-switcher support for Sphinx with:
    - `docs/_static/version-switcher.js` and `version-switcher.css`.
    - Sidebar template `docs/_templates/sidebar/versions.html`.
  - New `versions.json` generation and “stable” alias logic in docs build workflow.
  - Support for multi-version documentation builds via `sphinx-multiversion`.
  - Added project version/commit metadata injection to Sphinx `conf.py`.

- **Codebase**
  - `AFMImageStack` now registers and resolves new processing groups:
    - `video_processing` and `stack_edit` modules added.
  - Added internal state backup mechanism (`state_backups` attribute) for preserving
    origonal metadata.
  - Moved the versioning module for functions from the processing subpakage to the utils subpackage.

- **Docs Generation**
  - Automatic inclusion of `playNano.analysis.utils.loader`, `playNano.processing.video_processing`, and
    `playNano.processing.stack_edit` in the API reference.

- **GitHub**
  - Issue templates added for bug reports and feature requests.

### Changed

- **Documentation**
  - Documents and docstrings updated to correct typos address sphinx build warnings.
  - Instructions for installation from PyPi to the user docs.
  - Clearer instruction for the installation procedure to use notebooks added.
  - Major re-write of `processing.rst`:
    - Expanded explanation of pipeline structure, operation types, and provenance tracking.
    - Improved CLI/GUI examples, programmatic usage, and plugin registration guidance.
    - `index.rst`, `introduction.rst`, `gui.rst`, and `quickstart.rst` updated with links to new
      **Exporting** and **Processing Operations Reference** pages.
    - `analysis.rst` fixed Sphinx link formatting.
    - Improved generated module list formatting for analysis modules.
    - Updated Sphinx `conf.py` to:
      - Support multi-version builds, version detection, and sidebar switcher.
      - Move `src` path resolution to a relative form.
      - Reorganize theme and HTML sidebar configuration.
    - Added `sphinx-multiversion` to `pyproject.toml` under `[project.optional-dependencies.docs]`.

- **GitHub Actions**
  - Overhauled `docs.yaml` workflow:
    - Builds and deploys versioned docs on `main` and release tags.
    - Adds PR preview artifact upload.
    - Generates `versions.json` and root redirect index.
    - Creates “stable” alias for latest release.
  - Renamed job to “Build and Deploy Docs”.

  - **Notebooks**
    - Added a root search function so hard coded paths to demo data from the tests folder
      can be accessed wherever jupyter is launched from.

### Fixed

- **Documentation**
  - Corrected malformed Sphinx links and build warnings across multiple pages.
  - Improved auto-generated module list formatting and spacing.

## [0.1.0.post1] - 2025-10-14

### Changed

- **GitHub**
  - Add workflows for pypi publishing

- **Documentation**
  - Add badges for tests, PyPi python version and PyPi releases.
  - Added links to the documentation and user guide on github pages.
  - Added a PyPi installation guide to the README and user guide.
  - Some general rewriting and improvements

## [0.1.0] - 2025-09-17

### Added

- First public release 🎉

- **AFM data loading & playback**
  - Load HS‑AFM videos from .h5-jpk and .asd files and folders of .spm and .jpk files.
  - Time‑aware frame navigation and consistent pixel/scale metadata.

- **Processing pipeline with masks & full provenance**
  - Sequential filters and masks (e.g., plane removal, row/median alignment, polynomial flatten, Gaussian filtering).
  - Each step is recorded with index, name, parameters, timestamps, and environment details under `stack.provenance`.
  - Processed snapshots and masks are stored with ordered keys like `step_<n>_<name>` for reliable inspection and re‑use.

- **Reproducible export & re‑import (analysis‑ready)**
  - Save the current stack state (with stages, masks, and provenance) to **HDF5 (`.h5`)** or **NumPy bundles (`.npz`)**.
  - Re‑load bundles later to continue processing and run analyses with the full history intact.
  - Export to **OME‑TIFF** for interoperability and to **GIF** (with optional scale bars)
    for quick sharing and presentation.

- **Interactive GUI (PySide6) for exploration**
  - Real‑time playback, frame seeking, and snapshot previews.
  - **Z‑range control** (auto or manual) to maintain consistent height scaling across frames.
  - **Annotations/overlays** (i.e. timestamps, raw data label, scale bar) rendered on top of frames.
  - Built‑in dark theme stylesheet for high‑contrast analysis.

- **Analysis framework**
  - Build analysis pipelines from built-in and pluggable analysis modules.
  - Built-in analysis modules (e.g., LoG blob detection, DBSCAN/K‑Means/X‑Means clustering, particle tracking).
  - Produces labeled masks, per‑feature properties (area, min/max/mean, bbox, centroid), and summary statistics.
  - Analysis outputs are keyed and traced in provenance for reproducibility.

- **Command Line Interface (CLI)**
  - `playnano` entrypoint to run processing pipelines, export bundles (TIFF/NPZ/HDF5), and create GIFs from the shell.

- **Notebooks**
  - Jupyter notebooks included to demonstrate programmatic workflow.
  - Overview notebook covers the whole loading, processing, analysis and export workflow.
  - Processing notebook focuses on processing and export of loaded data.

- **Documentation**:
  - Created a Sphinx documentation site on GitHub Pages.
  - **User Guide** covering installation, quick start, GUI and CLI usage, processing, analysis and exports.
  - **API Reference** generated with `sphinx-autoapi` for all packages.
  - **CLI reference** with examples and typical workflows.
  - Furo theme and MyST Markdown configuration for a clean, consistent look.

### Changed

- N/A (initial public release).

### Fixed

- N/A (initial public release).

[Unreleased]: https://github.com/derollins/playNano/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/derollins/playNano/releases/tag/v0.1.0
[0.1.0.post1]: https://github.com/derollins/playNano/releases/tag/v0.1.0.post1
[0.2.0]: https://github.com/derollins/playNano/releases/tag/v0.2.0
[0.2.1]: https://github.com/derollins/playNano/releases/tag/v0.2.1
[0.2.2]: https://github.com/derollins/playNano/releases/tag/v0.2.2
