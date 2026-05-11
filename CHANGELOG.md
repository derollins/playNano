<!-- markdownlint-disable MD033 MD024-->
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Official support for Python 3.13
- Compatibility with NumPy 2.x (including NumPy 2.0+)

#### Processing filters

- **Flip filter**
  - Added the built-in processing filter ``vertical_flip``, which uses NumPy
    ``flipud()`` to reverse the row order of a 2D image.
  - No masked equivalent is provided.

#### File readers

- **ARIS reader**
  - Support for loading Asylum Research **``.aris``** high-speed AFM data files.
  - Comprehensive ARIS loader including:
    - Channel-name extraction from HDF5 metadata.
    - Global and per-frame pixel-size scaling with well-defined fallback behaviour.
    - Frame sorting and stacked-image construction.
    - Timestamp parsing and validation.
  - Multi-suffix file loading support, enabling correct detection of formats such as
    ``.ome.tif`` in addition to single-suffix formats.

#### Visual exports

- **Video and image sequence export**
  - New export formats for animated data visualisation:
    - **Video** (MP4, AVI, MOV, MKV)
    - **Image sequences** (PNG, JPG)
  - All visual exports support metadata overlays including scale bars, timestamps,
    and data labels.
  - Video and sequence exports are available via the API, CLI, and GUI.

#### Colormaps and perceptual scaling

- Introduced a unified colormap selection system for the GUI and all visual exports.
- Added custom colormaps designed for AFM data:
  - **afm_brown** (default): perceptually linear (R² ≈ 0.996), minimising contrast
    compression and flicker in HS-AFM videos.
  - **playnano_gold**: high-contrast, full-dynamic-range (L* 0–100) map for complex topography.
  - **classic_afm**: legacy-style map for continuity with older AFM software.

#### Command Line Interface (CLI)

- Added global ``--version`` flag.
- Added ``--cmap`` option to select colormaps in ``play``, ``process`` and ``wizard``.
- Added ``--fps`` option:
  - In ``play``:
    - Optional argument.
    - If not provided, the frame rate is derived from the data.
    - If provided without a value or with a non-numeric value, a default FPS (5) is used
      and a warning is logged.
  - In ``process``:
    - Requires a numeric value.
    - Invalid values raise an error.
- Added ``--make-sequence`` and ``--make-video`` options for exporting image sequences
  and videos in ``process``.
- Added ``--draw-ts`` to toggle timestamp rendering in exported media (default: off).
- Extended the wizard REPL:
  - After running a processing pipeline, GIF export prompts now include options for
    timestamps and scale bars (including scale-bar length).
  - After GIF export, the wizard can also export videos with configurable ``zmin``,
    ``zmax``, timestamps, and scale bars.
- Improved CLI UX and robustness:
  - ``--fps`` in ``play`` now logs a warning when a non-numeric value is supplied
    and falls back gracefully, instead of failing silently.
  - Help text now explicitly documents default FPS behaviour.

#### Resources and assets

- Introduced ``src/playnano/resources/`` for non-code assets.
- Moved bundled fonts into ``resources/fonts``.
- Added ``importlib.resources`` integration for robust, cross-platform asset loading.

#### Documentation and tests

- Integrated ``sphinx-apidoc`` into ``conf.py`` via a ``builder-inited`` hook for
  automatic API documentation generation.
- Added test coverage for:
  - ARIS HDF5 loading and edge cases.
  - Multi-suffix file handling (e.g. ``.ome.tif``).
  - Per-frame pixel-size overrides and fallback logic.
  - Timestamp mismatch and missing-channel errors.
  - Video and image-sequence export.
  - CLI FPS parsing and help text.
- Refined CLI test infrastructure:
  - CLI tests now patch the ``entrypoint`` dispatch directly, ensuring full coverage
    of argparse wiring and command dispatch.
  - Added regression tests for CLI help output and invalid argument handling.

---

### Changed

#### Pixel-size handling

- Updated ``pixel_size_nm`` semantics:
  - The ``AFMImageStack.pixel_size_nm`` attribute now represents the global or first-frame
    pixel size.
  - Per-frame overrides are stored in ``frame_metadata["frame_pixel_size_nm"]``.
  - Most datasets will have identical values; differences occur when scan size changes
    mid-scan (e.g. ARIS files).
  - GUI playback and animated exports now respect per-frame pixel size.
- Updated NPZ, OME-TIFF, and HDF5 export documentation to reflect per-frame support.

#### Visualisation rendering

- Introduced ``playnano.io.render_utils`` as a single source of truth for visual exports.
  - Centralises visual constants for consistent appearance across formats.
  - Enforces minimum output resolution to prevent pixelated annotations.
  - Ensures scale bars are physically correct based on ``pixel_size_nm`` and that text
    size and positioning are consistent across resolutions.
- Rendering logic previously in ``export_gif`` has been migrated to this module.

#### GUI

- Added a colormap selection dropdown to the export panel.
- The active colormap is now displayed on the z-scale histogram and updates dynamically
  when ``zmin`` or ``zmax`` change.

#### Logging and user feedback

- Improved warning messages for invalid CLI arguments (e.g. non-numeric ``--fps``),
  providing clearer runtime feedback without interrupting interactive workflows.

#### Developer tooling

- Updated pre-commit configuration, including Ruff, Black, isort and markdownlint
  version alignment and formatting fixes.

#### Tests

- Improved numerical robustness in tests (floating-point comparisons now use tolerances for cross-platform consistency)
- Added Python 3.13 to the test matrix

---

### Fixed

- Fixed loader errors relating to missing file extensions and no-suffix inputs.
- Connected FPS calculation based on ``line_rate`` to GUI playback logic.
- Fixed `frame_metadata` construction to ensure one entry per frame (previously overwritten or
  incomplete in some loaders, notably `.spm` and `.jpk`)
- Standardised metadata structure across all formats:
  - `timestamp`
  - `frame_pixel_size_nm`
  - `line_rate`
- Clarified pixel size behaviour per format:
  - Per-frame (`.jpk`, `.aris`, `.spm`)
  - Constant (`.h5-jpk`, `.asd`)
- Updated docstrings to document pixel size semantics and usage

## [0.3.1] - 2026-03-12

### Fixed

- Resolved incorrect mask output in video processing pipeline.
  Video filter functions that return (array, metadata) were previously interpreted as (array, mask),
  causing metadata to be mis‑assigned as a mask and leading to failures in pipelines using mask‑aware steps.

### Changed

- Updated `ProcessingPipeline._handle_video_filter_step()` to accept mask as an explicit optional parameter,
  ensuring its correct propagation through `run()` / `_run_single_step()` without overloading metadata as mask data.
- Metadata from video‑filter functions is now stored only in `step_record` as a side effect and is no longer returned
  as a mask-like object.
- Added debug‑level logging for clearer diagnosis of mask‑related failures.
- Updated the test suite to:
  - reflect the new behaviour of mask passthrough in _handle_video_filter_step(),
  - confirm that metadata is recorded correctly,
  - ensure mask value is preserved through video filter steps.
- Removed mask handling from filters.zero_mean() and improved robustness of masked_filter.zero_mean_masked() to make
  each function role‑specific.
- Incremented version numbers to reflect these changes.
- Updated docstrings for clarity and corrected supported Python versions in the README.

## [0.3.0] - 2026-03-06

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
[0.3.0]: https://github.com/derollins/playNano/releases/tag/v0.3.0
[0.3.1]: https://github.com/derollins/playNano/releases/tag/v0.3.1
