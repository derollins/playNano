<!-- markdownlint-disable MD033 MD024-->
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Next change in the project

## [0.1.0] - YYYY-MM-DD

### Added

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
  - **Annotations/overlays** (e.g., masks, regions of interest) rendered on top of frames.
  - Built‑in dark theme stylesheet for high‑contrast analysis.

- **Analysis framework**
  - Pluggable analysis modules (e.g., LoG blob detection, DBSCAN/K‑Means/X‑Means clustering, particle tracking).
  - Produces labeled masks, per‑feature properties (area, min/max/mean, bbox, centroid), and summary statistics.
  - Analysis outputs are keyed and traced in provenance for reproducibility.

- **Command Line Interface (CLI)**
  - `playnano` entrypoint to run processing pipelines, export bundles (TIFF/NPZ/HDF5), and create GIFs from the shell.

### Changed

- N/A (initial public release).

### Fixed

- N/A (initial public release).

---

[Unreleased]: https://github.com/derollins/playNano/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/derollins/playNano/releases/tag/v0.1.0
