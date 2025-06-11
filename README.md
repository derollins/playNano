# 📽️ playNano

**AFM Video Reader for `.h5-jpk` files and other high-speed AFM video formats**

<div align="center">

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-GPLv3-blue)](LICENSE)
![CI](https://github.com/derollins/playNano/actions/workflows/pre-commit.yaml/badge.svg)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Code style: flake8](https://img.shields.io/badge/code%20style-flake8-456789.svg)](https://github.com/psf/flake8)
[![codecov](https://codecov.io/github/derollins/playNano/graph/badge.svg?token=NEV1OC12AV)](https://codecov.io/github/derollins/playNano)

</div>

**playNano** is a Python tool for loading, filtering, visualising, and exporting time-series AFM data,
such as high-speed AFM (HS-AFM) videos. It supports interactive playback of AFM video data, application
of processing filters, and export in multiple formats, including OME-TIFF, NPZ (NumPy zipped archive),
HDF5 bundles, and animated GIFs.

**Files read:**
<div align="center">

**`.h5-jpk`, `.jpk`, `.asd`**

</div>

This project requires Python 3.10 or newer and is in development. If you find any issues, please open an issue at:
<https://github.com/derollins/playNano/issues>

Questions? Email: <d.e.rollins@leeds.ac.uk>

---

## ✨ Features

- 📂 **Extracts AFM time-series (video) data** from `.h5-jpk` and `.asd` files and folders of `.jpk` files.
- ▶️ **Animated image viewer** for high-speed AFM playback.
- 🪟 **Applies basic filters** and ordered filter chains to image data.
- 📩 **Exports** to OME-TIFF stacks, NPZ bundles, and HDF5 bundles.
- 🎞️ **Generates animated GIFs** of AFM stacks with annotations.
- 🧠 Built for integration with analysis pipelines and tools like `TopoStats`.

---

## 📦 Installation

Clone the repository into a new folder:

```bash
git clone https://github.com/derollins/playNano.git
cd playNano
```

It is recommended to use a virtual environment. Then install in editable mode:

```bash
pip install -e .
```

## 🚀 Quickstart

Generate a flattened AFM image stack and export a GIF:

```bash
playnano run "example_data/sample.h5-jpk" --processing "remove_plane;mask_mean_offset:factor=1;row_median_align" --make-gif
```

Or launch an interactive viewer to inspect and flatten the data:

```bash
playnano play "example_data/sample.h5-jpk"
```

Press the **f** key to flatten with default steps.

## ⌨️ CLI Usage

### General  Structure

```bash
playnano <command> <input_file> [options]
```

Commands:

- `play`: Launches the interactive viewer.
- `run`: Batch processing mode for applying filters and exporting.
- `wizard`: Open interactive processing wizard for applying filters and exporting.

### 👟 Command Line mode (`run`)

Apply filters and export without interaction.

```bash
playnano run /path/to/afm_file.h5-jpk \
  [--channel CHANNEL] \
  [--processing [PROCESSING_STEPS_STR] or [--processing-file [PATH_TO_PROCESSING_YAML] \
  [--export tif,npz,h5] \
  [--make-gif] \
  [--output-folder OUTPUT_DIR] \
  [--output-name BASE_NAME]
  [--scale-bar-nm SCALE_BAR_INT]

```

- `--channel`: (default: `height_trace`): Channel to load.

- `--output-folder`: Directory to write exports and/or GIF (default: ./output).

- `--output-name`: Base filename for output files (no extension).

- `--export`: Comma-separated list of formats to export (tif, npz, h5).

- `--make-gif`: Write an animated GIF after filtering.

- `--scale-bar-nm`: Length of scale bar annotation on GIF animation in nm. Set to 0 to remove sacle bar.

- `--processing`: Semi-colon-separated list of filters and masks to apply in order (see Flattening section
                  below), with parameters seperated with a colon.
                  I.e. "remove_plane;mask_mean_offset:factor=1;row_median_align;clear;gaussian_filter"

- `--processing-file`:  An alternative processing input feild which takes a yaml file listing filtering
                        steps and parameters.

> Expected YAML schema:

 ```yaml
    filters:
    - name: remove_plane
    - name: gaussian_filter
        sigma: 2.0
    - name: threshold_mask
        threshold: 2
    - name: polynomial_flatten
        order: 2
```

### 🧙‍♂️ Interactive Wizard mode (`wizard`)

Launches an interactive REPL for building and executing processing pipelines.

Takes ``--channel``, `--output-folder`, `--output-name` and `--scale-bar_nm` flags as above.

```bash
playnano wizard /path/to/afm_file.h5-jpk \
  [--channel CHANNEL] \
  [--output-folder OUTPUT_DIR] \
  [--output-name BASE_NAME]
  [--scale-bar-nm SCALE_BAR_INT]
```

Use `help` to see a list of calls.

Start my using the `add` call followed by the name of a filter or mask to add a processing step, the REPL will ask
for paramters if required. Once a few steps have been added use `list` to see the pipeline, `save` followed by a path
to a `.yaml` file to save the pipeline for use with the `--processing-file` flag and `run` to execute the processes.
      add remove_plane
      add gaussian_filter
        sigma: 2.0
      list
      run
      quit

Once run, the REPL will ask you about exporting the processed data and generating GIF animiations.
These will be save to the `--output-dir` set at initlization and as the `--output-name`.

Use `quit` to exit the wizard early.

### 🖥️ Interactive Playback mode (`play`)

Opens an OpenCV window for visualising and processing the AFM stack.

The window is initilized with similar flags to the `run` mode without the `--export` or `--make-gif` flags.

```bash
playnano play /path/to/afm_file.h5-jpk \
  [--channel CHANNEL] \
  [--processing [PROCESSING_STEPS_STR] or [--processing-file [PATH_TO_PROCESSING_YAML] \
  [--output-folder OUTPUT_DIR] \
  [--output-name BASE_NAME] \
  [--scale-bar-nm SCALE_BAR_INT]

```

**Viewer key bindings:**

Press keys to inteact with the video viewing window:

Apply filter:

- **f** — Apply filtering and update view.
- **Space** — Toggle between raw and filtered data.

Save and export:

- **t** — Export the current data as an OME-TIF (.ome.tif), loadable in many image analysis programmes.
- **n** — Export the current data as a NumPy zipped archive (.npz).
- **h** — Export the current data as a HDF5 bundle (.h5).
- **g** — Export the data as an animated GIF with the annotations in the viewed (scale bar and timestamps).

> 📝 Note: The exported data reflects the current view — if raw data is shown, raw is exported;
> if filters are applied, the filtered view is saved.

Other commands:

- **q** or **ESC** — Quit the viewer.

## 🪟 Flattening

### Filters

- **Remove Plane** (remove_plane): Fit a 2D plane to the image with inear regression and subtract it.

- **Polynomial Flatten** (polynomial_flatten): Fit and subtract a 2D polynomial of given order to remove slow surface trends.

  > Polynominal calculated from unmasked data if present. Parameter: Order(Int) default set to 2.

- **Row Median Align** (row_median_align): Subtract the median of each row from that row to remove horizontal banding.

  > Polynominal calculated from unmasked data if present.

- **Zero Mean** (zero mean): Subtract the overall mean height to center the background around zero.

  > Mean calculated from unmasked data if mask is present.

- **Gaussian Filter** (gaussian_filter): Apply a Gaussian low-pass filter to smooth high-frequency noise.

  > Parameter: Sigma(float) default set to 1 pixel.

### Masks

- **Mask with threshold** (mask_threshold): Mask data above a threshold.

> Parameter: Threshold(float) default set to 0.0.

- **Mask with mean offset** (mask_mean_offset): Mask data above the mean +/- (s.d. * factor).

> Paramter: Factor(float) deafult set to 1.0.

- **clear** resets mask.

## 📟 Outputs

Once loaded you can export AFM stacks in the following formats:

| Format   | Description                                | Extension  |
| -------- | ------------------------------------------ | ---------- |
| OME-TIFF | Multi-frame TIFF for image analysis        | `.ome.tif` |
| NumPy    | Zipped archive of array + metadata         | `.npz`     |
| HDF5     | Self-contained AFM stack bundle            | `.h5`      |
| GIF      | Animated GIF with scale bar and timestamps | `.gif`     |

- Use `--output-folder` and `--output-name` to customize where and how files are saved.
- Defaults:

  - Folder: `./output/`
  - Name: derived from input filename (with `_filtered` suffix if filters were used)

## Logging Level

Control verbosity with:

```bash
--log-level {DEBUG,INFO,WARNING,ERROR}
```

Default is `INFO`.

## 🧪 Examples

Interactive playback (`play`) with filters and export folder:

```bash
playnano play sample.h5 --processing "remove_plane;mask_mean_offset:factor=1;row_median_align" \
                        --output-folder ./gifs \
                        --output-name sample_view
```

Batch run (`run`) with filtering steps set by processiing yaml file, exporting OME-TIFF and NPZ bundles, plus GIF:

```bash
playnano run sample.h5 \
--processing-file my_filter_steps.yaml \
--export tif,npz \
--make-gif \
--output-folder ./results \
--output-name sample_processed
```

## 🧩 Filter Plugins

You can extend playNano by installing third-party filter plugins via entry points under playNano.filters. Edit the
`[project.entry-points."playNano.filters"]` section of the pyproject.toml file like this:

```toml
[project.entry-points."playNano.filters"]
other_filter   = "playNano.processing.filters:other_filter"
my_new_plugin  = "mylibrary.moldule:my_new_plugin"
```

These become available in the CLI filter lists automatically.

Plugins must have the format:

```python
def filter_plugin(2Ddata: np.ndarray, **kwargs) -> np.ndarray:
```

## ⚠️ Notes

- Make sure the input file includes valid metadata like line_rate, or GIF generation may fail.

- If --channel is incorrect or missing from the file, you’ll receive an error.

- For .h5-jpk and other multi-frame formats, a single file is loaded. For formats like .jpk or .spm, provide a folder
    containing the frame files.

## 📁 Project Structure

```text
playNano/
├── io/              # I/O utilities (e.g. file loaders and exports)
├── playback/        # Interactive window
├── processing/      # Image flattening, filters, and processing logic
├── utils/           # Utility functions
├── cli/             # CLI entry point and functions
└── afm_stack.py     # AFMImageStack class and metadata handling
```

## 🧩 Dependencies

Requires Python 3.10 or newer.

This project requires the following Python packages:

- `numpy`
- `h5py`
- `Pillow`
- `matplotlib`
- `opencv-python`
- `scipy`
- `python-dateutil`
- `tifffile`
- [`AFMReader`](https://github.com/AFM-SPM/AFMReader) — for reading `.jpk` and `.asd` files
    (also planned for use in future for `.spm` loading).

## 🤝 Related Software

These are some software packages that have helped and inspired this project:

### [Topostats](https://github.com/AFM-SPM/TopoStats)

A general AFM image processing programme written in Python that batch processes AFM images.
Topostats is able to flatten raw AFM images, mask objects and provides advanced analysis tools
including U-net based masking.

### [AFMReader](https://github.com/AFM-SPM/AFMReader)

Spun out of Topostats, AFMReader is Python library for loading a variety of AFM file formats. It opens
each as a tuple containing a NumPy array and a float referring to the planar pixel to nanometer convertion
factor. Within playNano this library is used to open the folder-based AFM video formats.

### [NanoLocz](https://github.com/George-R-Heath/NanoLocz)

A free MATLAB app with an interactive GUI that is able to load, process and analyse AFM images and
high-speed AFM videos. Featuring mask analysis, particle detection and tracking, it also
integrates Localization  AFM [(L-AFM)](https://www.nature.com/articles/s41586-021-03551-x).

## 📜 License

This project is licensed under the [GNU General Public License v3.0 (GPLv3)](https://www.gnu.org/licenses/gpl-3.0.html)
