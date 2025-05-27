# 📽️ playNano

**AFM Video Reader for `.h5-jpk` files and other high speed AFM video formats**

<div align="center">

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-GPLv3-blue)](LICENSE)

</div>

`playNano` is a lightweight Python library for reading and processing atomic force microscopy (AFM) as **video/image stacks**.
`playNano` focuses on extracting and reshaping time-series AFM data, enabling dynamic visualisation, analysis, and export.

---

## ✨ Features

- 📂 **Extracts AFM time-series (video) data** from `.h5-jpk` files and folders of `.jpk` files.
- 🔍 **Auto-detects** likely image channels (e.g., `/Height`) if not specified.
- 🖼️ **Exports animated GIFs** of AFM image stacks for quick visualisation
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

## 🛠️ CLI Usage

```bash
playNano path/to/file.h5-jpk --channel height_trace --output-folder ./output --save-raw --make-gif --log-level DEBUG
```

### Options

    `input_file` (positional): Path to your `.h5-jpk` file or folder of `.jpk` files.

    `--channel`: Channel name, e.g. `height_trace` (default).

    `--save-raw`: Keep a copy of the unflattened image stack.

    `--make-gif`: Export a GIF of the flattened stack.

    `--output-folder`: Where to save outputs.

    `--log-level`: Logging verbosity (`DEBUG`, `INFO`, etc.)

### Output

    📂 Flattened image stack (planned: `.npy`, `.tiff`)

    🎞️ Optional animated GIF: `flattened.gif` with scale and timestamp

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
- [`AFMReader`](https://github.com/AFM-SPM/AFMReader) — for reading `.jpk` files (also planned for use in future `.asd` and `.spm` loading).
- [`TopoStats`](https://github.com/AFM-SPM/TopoStats) — for AFM image flattening and processing

## 🤝 Related Software

These are some software packages that have helped and inspired this project:

### [Topostats](https://github.com/AFM-SPM/TopoStats)

A general AFM image processing programme written in Python that batch processes AFM images. Topostats is able to flatten raw AFM images,
mask objects and provides advanced ananlysis tools including U-net based masking. playNano leverages the `filters` module to flatten loaded AFM frames.

### [AFMReader](https://github.com/AFM-SPM/AFMReader)

Spun out of Topostats, AFMReader is Python library for loading a variety of AFM file formats. It opens each as a tuple containing a Numpy array
and a float refering to the planar pixel to nanometer convertion factor. Within playNano this library is used to open the folder-based AFM video formats.  

### [NanoLocz](https://github.com/George-R-Heath/NanoLocz)

A free MATLAB app with an interactive GUI that is able to load, process and analyse AFM images and high- speed AFM videos. Faeturing mask
analysis, particle detection and tracking, it also intergrates Localization  AFM [(L-AFM)](https://www.nature.com/articles/s41586-021-03551-x).

## 📜 License

This project is licensed under the [GNU General Public License v3.0 (GPLv3)](https://www.gnu.org/licenses/gpl-3.0.html).
