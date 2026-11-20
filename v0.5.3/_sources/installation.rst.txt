Installation
============

This page describes recommended installation paths for **playNano**.
Conda is the recommended default because it gives the most reliable binary
support for scientific packages and Qt/PySide.

The latest stable release is available on PyPI as the ``playnano`` package
as well as through GitHub.

System requirements
-------------------

- Python **3.10 - 3.13** (3.11 recommended)
- Linux, macOS, or Windows
- Internet connection for downloading packages

Installation Guide
------------------

It is recommended that **playNano** is installed in a virtual environment to
ensure dependency isolation, prevent version conflicts with other Python packages,
and maintain a clean, reproducible setup across different systems.

Quick Install (recommended: conda)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a reproducible conda environment and install **playNano**.

.. note::

   To use the recommended conda-based installation, first install either
   `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ or
   `Anaconda <https://www.anaconda.com/products/distribution>`_, which provide
   the ``conda`` package manager. Miniconda is lightweight and ideal for custom
   setups, while Anaconda includes a full suite of scientific packages out of the box.

.. code-block:: bash

   # 1) Create and activate a conda env
   conda create -n playnano_env python=3.11
   conda activate playnano_env

   # 2) Install the package from PyPI
   pip install playnano

Notes & troubleshooting
-----------------------

- **PySide6 / Qt**
  The GUI uses ``PySide6-Essentials`` (the core Qt modules only), installed from PyPI by
  ``pip install playnano``. Do **not** also install ``pyside6`` from conda-forge into the same
  environment: pip does not recognise it as ``PySide6-Essentials`` and installs a second copy of
  Qt into the same folder. If an environment already has conda's ``pyside6``, remove it and
  reinstall from pip:

  .. code-block:: bash

     conda remove pyside6
     pip install --force-reinstall PySide6-Essentials

- **AFMReader**
  Required for reading some vendor formats (``.jpk``, ``.spm``, ``.asd``). It is installed
  automatically from PyPI. playNano supports AFMReader 0.0.7 and 0.0.8; to install a
  supported version by hand, use:

  .. code-block:: bash

     pip install "afmreader>=0.0.7,<0.0.9"

- **Exports and metadata**
  GIF and video exports use the frame timestamps for their annotations when the file
  provides them, and frame numbers otherwise. If an export fails, rerun with
  ``--log-level DEBUG`` and check the console log.

Verification
^^^^^^^^^^^^

After installation, verify CLI and import:

.. code-block:: bash

   playnano --help

Check version from Python:

.. code-block:: bash

   python -c "import playnano; print(playnano.__version__)"


Developer install
-----------------

If you wish to change the code or contribute to the project, clone the repository
from the GitHub repository and install in "editable" mode. This will give you access to
the latest development version and make changes to the codebase.

.. code-block:: bash

   # 1) Clone the repository (if you haven't already)
   git clone https://github.com/derollins/playNano.git
   # And navigate to the project root
   cd playNano

   # 2) Create and activate a conda env
   conda create -n playnano_env python=3.11
   conda activate playnano_env

   # 3) Install the package (from the project root)
   pip install -e .

Optional dependencies for the documentation, the example notebooks and development are
available as pip "extras". If you are contributing, install at least the ``dev`` extra,
which includes the linting and testing tools:

.. code-block:: bash

   pip install -e ".[docs]"       # docs build dependencies (Sphinx, theme, nbsphinx)
   pip install -e ".[notebooks]"  # notebook/demo dependencies (Jupyter)
   pip install -e ".[dev]"        # developer dependencies (linting, testing, pre-commit)

Developer usage
^^^^^^^^^^^^^^^

Install all of the dependencies for linting, tests, docs and notebooks:

.. code-block:: bash

   # After cloning the repo and activating your conda env
   pip install -e ".[dev,docs,notebooks]"

Run tests:

.. code-block:: bash

   pytest

Build the docs locally:

.. code-block:: bash

   sphinx-build docs docs/_build/html

Alternative manual install: pip + venv
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you prefer the standard library virtualenv workflow, use ``venv``:

.. code-block:: bash

   python -m venv .venv
   # Linux / macOS
   source .venv/bin/activate
   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1

   pip install -e .[dev,docs,notebooks]

Install via environment.yaml
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you prefer a reproducible setup using a pre-defined environment file, you can
use the provided ``environment.yaml`` to create a conda environment with all required
dependencies.

.. code-block:: bash

   conda env create -f environment.yaml
   conda activate playnano_env
   pip install -e .   # install playNano itself, from the project root

The environment file installs:

- Python 3.11 and the core scientific libraries (NumPy, SciPy, pandas, h5py, Pillow,
  Matplotlib, scikit-learn, scikit-image, OpenCV)
- AFM-specific tools: ``afmreader`` (installed with pip)
- GUI support via ``PySide6-Essentials`` (installed with pip)

It does not install playNano itself, which is why the last command is needed.

.. note::

   The environment uses the ``conda-forge`` channel for reliable binary support across platforms.

.. tip::

   If you modify ``environment.yaml``, you can update your environment with:

   .. code-block:: bash

      conda env update -f environment.yaml --prune

Dependencies
------------

This project requires the following Python packages:

- ``numpy``
- ``pandas``
- ``h5py``
- ``Pillow``
- ``matplotlib``
- ``opencv-python-headless``
- ``scipy``
- ``scikit-learn``
- ``scikit-image``
- ``python-dateutil``
- ``tifffile``
- ``pyyaml``
- ``PySide6-Essentials`` — for the GUI (the core Qt modules of PySide6)
- `AFMReader <https://github.com/AFM-SPM/AFMReader>`_ — for reading ``.jpk``, ``.spm`` and ``.asd`` files
