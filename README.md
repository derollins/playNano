# 📽️ playNano

**AFM Video Reader for `.h5-jpk` Files**

<div align="center">

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-GPLv3-blue)](LICENSE)

</div>

`playNano` is a lightweight Python library for reading and processing atomic force microscopy (AFM) as **video/image stacks**. `playNano` focuses on extracting and reshaping time-series AFM data, enabling dynamic visualisation, analysis, and export.

---

## ✨ Features

- 📂 **Extracts AFM time-series (video) data** from `.h5-jpk` files.
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

    `input_file` (positional): Path to your `.h5-jpk` file.

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
├── io/              # Input/output utilities (e.g., GIF export)
├── loaders/         # File format-specific loaders
├── processing/      # Image flattening, filtering, etc.
├── stack/           # AFMImageStack class and metadata
└── main.py          # CLI entry point
```

## 🧩 Dependencies

This project requires the following Python packages:

- `numpy` 
- `h5py` 
- `Pillow`
- `matplotlib`
- [`TopoStats`](https://github.com/AFM-SPM/TopoStats/) — for AFM image flattening and processing

## 📜 License

This project is licensed under the [GNU General Public License v3.0 (GPLv3)](https://www.gnu.org/licenses/gpl-3.0.html).
