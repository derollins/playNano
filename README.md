# 📽️ playNano

**AFM Video Reader for `.h5-jpk` files and other high speed AFM video formats**

<div align="center">

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-GPLv3-blue)](LICENSE)
![CI](https://github.com/derollins/playNano/actions/workflows/pre-commit.yaml/badge.svg)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Code style: flake8](https://img.shields.io/badge/code%20style-flake8-456789.svg)](https://github.com/psf/flake8)
[![codecov](https://codecov.io/github/derollins/playNano/graph/badge.svg?token=NEV1OC12AV)](https://codecov.io/github/derollins/playNano)

</div>

`playNano` is a lightweight Python library for reading and processing atomic force microscopy (AFM)
as **video/image stacks**. `playNano` focuses on extracting and reshaping time-series AFM data,
enabling dynamic visualisation, analysis, and export.

This project is in development and not fully stable. If you find any issues please open an issue at:
<https://github.com/derollins/playNano/issues>

If you have any questions please get in touch: <d.e.rollins@leeds.ac.uk>

---

## ✨ Features

- 📂 **Extracts AFM time-series (video) data** from `.h5-jpk` files and folders of `.jpk` files.
- 🔍 **Auto-detects** likely image channels (e.g., `/Height`) if not specified.
- ▶️ **Animated image viewer** to display high-speed AFM data.
- 🪟 **Applies basic filters** for levelling AFM images.
- 🖼️ **Exports animated GIFs** of AFM image stacks for quick visualisation.
- 🧠 Built for integration with analysis/visualisation pipelines and tools like `TopoStats`.

---

## 📦 Installation

Clone the repository into a new folder:

```bash
git clone https://github.com/derollins/playNano.git
cd playNano
```

It is recommended to install the package in a virtual environment.

Once in the virtual environment:

```bash
pip install -e .
```

## 🚀 Quickstart

Generate a flattened AFM image stack and export a GIF in one command:

```bash
playNano "example_data/sample.h5-jpk" --make-gif
```

Or open an interactive viewer window to inspect and flatten the data manually:

```bash
playNano "example_data/sample.h5-jpk" --play
```

## 🖥️ Interactive Viewer (--play)

Launches a window for visually exploring and flattening your AFM stack.

- **f** — Apply flattening and display the processed stack.
- **Space** — Toggle between raw and flattened view (after flattening).
- **e** — Export flattned video to `--output-folder` as `--output-name`.gif.
- **q** or **ESC** — Quit the viewer.

## 🛠️ CLI Usage

```bash
playNano path/to/file.h5-jpk --channel height_trace --output-folder ./output
--save-raw --make-gif --log-level DEBUG
```

You can also load a folder of .jpk files (not .h5-jpk) for batch processin

### Positional Arguments

`input_file` (positional): Path to your `.h5-jpk` file or folder of `.jpk` files.

### Common Options

`--channel`: Channel name, e.g. `height_trace` (default).

`--filter`: Select filter, options are `topostats_flatten`, `flatten_poly`
or `median_filter`.

`--make-gif`: Export a GIF of the flattened stack.

`--output-folder`: Where to save outputs.

`--output-name`: Name of file output. (Default: "flattened.gif")

`--log-level`: Logging verbosity (`DEBUG`, `INFO`, etc.)

N.B. If both `--make-gif` and `--play` are used the data is flattened again and the gif
generated after the interactive window is quit.

### Output

✅ Flattened image stack (in memory; save via --make-gif or e)

🎞️ Animated GIF with scale bar and timestamps

🧪 Planned formats: .npy, .tiff

## ⚠️ Notes

- Make sure the input file includes valid metadata like line_rate, or GIF generation may fail.

- If --channel is incorrect or missing from the file, you’ll receive an error.

## 📁 Project Structure

```text
playNano/
├── io/              # Input/output utilities (e.g. the common file loader and GIF export)
├── loaders/         # File format-specific loaders
├── processing/      # Image flattening, filtering, analysis etc.
├── stack/           # AFMImageStack class and metadata
└── main.py          # CLI entry point
```

## 🧩 Dependencies

This project requires the following Python packages:

- `numpy`
- `h5py`
- `Pillow`
- `matplotlib`
- `opencv-python`
- `scipy`
- `python-dateutil`
- [`AFMReader`](https://github.com/AFM-SPM/AFMReader) — for reading `.jpk` files
    (also planned for use in future `.asd` and `.spm` loading).
- [`TopoStats`](https://github.com/AFM-SPM/TopoStats) — for AFM image flattening and processing

## 🤝 Related Software

These are some software packages that have helped and inspired this project:

### [Topostats](https://github.com/AFM-SPM/TopoStats)

A general AFM image processing programme written in Python that batch processes AFM images.
Topostats is able to flatten raw AFM images, mask objects and provides advanced ananlysis tools
including U-net based masking. playNano leverages the `filters` module to flatten loaded AFM frames.

### [AFMReader](https://github.com/AFM-SPM/AFMReader)

Spun out of Topostats, AFMReader is Python library for loading a variety of AFM file formats. It opens
each as a tuple containing a Numpy arrayand a float refering to the planar pixel to nanometer convertion
factor. Within playNano this library is used to open the folder-based AFM video formats.

### [NanoLocz](https://github.com/George-R-Heath/NanoLocz)

A free MATLAB app with an interactive GUI that is able to load, process and analyse AFM images and
high- speed AFM videos. Faeturing mask analysis, particle detection and tracking, it also
intergrates Localization  AFM [(L-AFM)](https://www.nature.com/articles/s41586-021-03551-x).

## 📜 License

This project is licensed under the [GNU General Public License v3.0 (GPLv3)](https://www.gnu.org/licenses/gpl-3.0.html)
