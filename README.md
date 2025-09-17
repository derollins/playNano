<!-- markdownlint-disable MD033 -->
# 📽️ playNano

**AFM Video Reader for `.h5-jpk` files and other high-speed AFM video formats**

<div align="center">

[![GitHub release](https://img.shields.io/github/v/release/derollins/playNano?color=blue)](https://github.com/derollins/playNano/releases)
[![GitHub release downloads](https://img.shields.io/github/downloads/derollins/playNano/latest/total?color=orange)](https://github.com/derollins/playNano/releases)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue?logo=github)](https://derollins.github.io/playNano/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-GPLv3-blue)](LICENSE)
![CI](https://github.com/derollins/playNano/actions/workflows/pre-commit.yaml/badge.svg)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Code style: flake8](https://img.shields.io/badge/code%20style-flake8-456789.svg)](https://github.com/psf/flake8)
[![codecov](https://codecov.io/github/derollins/playNano/graph/badge.svg?token=NEV1OC12AV)](https://codecov.io/github/derollins/playNano)

</div>

**playNano** is a Python tool for loading, filtering, visualising, and exporting time-series AFM data,
such as high-speed AFM (HS-AFM) videos. It supports interactive playback, flexible processing pipelines,
and provenance-aware analysis tracking, and export in multiple formats, including OME-TIFF, NPZ (NumPy zipped archive),
HDF5 bundles, and animated GIFs.

**Files read:**
<div align="center">

**`.h5-jpk`, `.jpk`, `.asd`, `.spm`**

</div>

This project requires Python 3.10 or newer and is in development. If you find any issues, please open an issue at:
<https://github.com/derollins/playNano/issues>

Questions? Email: <d.e.rollins@leeds.ac.uk>

Full documentation (Sphinx): <https://derollins.github.io/playNano/>

📜 [Changelog](https://derollins.github.io/playNano/changelog.html)

---

## ✨ Features

- 📂 **Extracts AFM time-series (video) data** from `.h5-jpk` and `.asd` files and folders of `.jpk` and `.spm` files.
- ▶️ **Animated video viewer**, an interactive PySide6 viewer with playback, z-scale configuration, and export tools.
- 🪟 **Processing pipeline** (filters + masks) that records per-step provenance.
- 📏 **Analysis pipeline** for detection/tracking; stores outputs and provenance in the stack.
- 📩 **Exports** to OME-TIFF stacks, NPZ bundles, HDF5 bundles, and annotated GIFs.
- 🔌 **Plugin system** for custom filters.

---

## 📦 Installation and Dependencies

Requires Python 3.10–3.12.

Clone the repository into a new folder:

```bash
git clone https://github.com/derollins/playNano.git
cd playNano
```

It is recommended to use a virtual environment. Then install in editable mode:

```bash
pip install -e .
```

Key dependencies (install via pip install -e .): numpy, h5py, Pillow, matplotlib,
scipy, scikit-learn, python-dateutil, tifffile, AFMReader (optional).

More information is available in the documentation: <https://derollins.github.io/playNano/installation.html>

## 🚀 Quickstart

**Play a file (GUI):**

<p align="center">
  <img src="docs/images/GUI_window.png" alt="playNano GUI main window" width="400" />
</p>

```bash
playnano play ./tests/resources/sample_0.h5-jpk # This command opens example data if run in the project root
```

Replace the path with the location of your data (file for asd/h5-jpk or folder for spm/jpk)
This opens an interactive window that can be used to view the videos and configure
formatting for the display and GIF exports.
Press the **f** key to flatten with default steps.

**Batch process + make GIF:**

```bash
playnano process ./tests/resources/sample_0.h5-jpk \
  --processing "remove_plane;gaussian_filter:sigma=1.0" \
  --export tif,npz --make-gif --output-folder ./results
```

See the full docs for the complete [CLI reference](https://derollins.github.io/playNano/cli.html),
[GUI guide](https://derollins.github.io/playNano/gui.html), filters, YAML schemas, and examples.

## ⚠️ Notes

- Make sure the input file includes valid metadata like line_rate, or GIF generation may fail.

- If --channel is incorrect or missing from the file, you’ll receive an error.

- For .h5-jpk, .asd and other multi-frame formats, a single file is loaded. For formats like .jpk or .spm, provide a
    folder containing the frame files.

## Example Notebooks

- `notebooks/playnano_demo_notebook.ipynb`: step‑by‑step demo of loading, processing, analysing, and exporting time-series
 AFM data with playNano.

Install playNano using `pip install .[notebooks]` to include the `jupyter` dependency.

## 🧩 Dependencies

Requires Python 3.10, 3.11 or 3.12.

This project requires the following Python packages:

- `numpy`
- `h5py`
- `Pillow`
- `matplotlib`
- `opencv-python`
- `scipy`
- `scikit-learn`
- `python-dateutil`
- `tifffile`
- [`AFMReader`](https://github.com/AFM-SPM/AFMReader) — for reading `.jpk`, `.spm` and `.asd` files

## 🤝 Related Software

These are some software packages that have helped and inspired this project:

### [Topostats](https://github.com/AFM-SPM/TopoStats)

A general AFM image processing programme written in Python that batch processes AFM images.
Topostats is able to flatten raw AFM images, mask objects and provides advanced analysis tools
including U-net based masking.

### [AFMReader](https://github.com/AFM-SPM/AFMReader)

Spun out of Topostats, AFMReader is Python library for loading a variety of AFM file formats. It opens
each as a tuple containing a NumPy array and a float referring to the planar pixel to nanometer conversion
factor. Within playNano this library is used to open the folder-based AFM video formats.

### [NanoLocz](https://github.com/George-R-Heath/NanoLocz)

A free MATLAB app with an interactive GUI that is able to load, process and analyse AFM images and
high-speed AFM videos. Featuring mask analysis, particle detection and tracking, it also
integrates Localization  AFM [(L-AFM)](https://www.nature.com/articles/s41586-021-03551-x).

## 📜 License

This project is licensed under the [GNU General Public License v3.0 (GPLv3)](https://www.gnu.org/licenses/gpl-3.0.html)

## 📖 Citing playNano

If you use **playNano** in academic work, please cite it appropriately.

- For now, please cite the GitHub repository:
  > Rollins, D. (2025). playNano: AFM Video Reader and Analysis Toolkit.
  > GitHub repository: <https://github.com/derollins/playNano>

## Included Fonts

This project bundles the following fonts:

- **Steps Mono** by [Velvetyne Type Foundry](https://velvetyne.fr/fonts/steps-mono/),
  licensed under the SIL Open Font License 1.1.

- **Basic** by [Eben Sorkin](https://github.com/EbenSorkin),
  licensed under the SIL Open Font License 1.1.

Full license texts and attribution are provided in:

- `src/playNano/fonts/Steps-Mono/LICENCE.txt`
- `src/playNano/fonts/Basic/LICENCE.txt`
