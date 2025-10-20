Installation
============

This page describes recommended installation paths for **playNano**.
Conda is the recommended default because it gives the most reliable binary
support for scientific packages and Qt/PySide.

The latest stable release is available on PyPI as the ``playnano`` package
as well as through GitHub.

System requirements
-------------------

- Python **3.10 - 3.12** (3.11 recommended)
- Linux, macOS, or Windows
- Internet connection for downloading packages

.. note::

   NumPy is currently pinned to ``<2.0`` for compatibility with some
   scientific libraries. See the :doc:`changelog` for updates.

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

- **PySide6 / Qt issues**
  If pip installation of PySide6 fails (common on some Windows setups), prefer the conda package:

  .. code-block:: bash

     conda install -c conda-forge pyside6

- **AFMReader**
  Required for reading some vendor formats (``.jpk``, ``.spm``, ``.asd``). If it is not available from PyPI in your environment, install it from GitHub:

  .. code-block:: bash

     pip install git+https://github.com/AFM-SPM/AFMReader.git

- **GIF export / metadata**
  Some input files must include metadata (e.g. ``line_rate``). If GIF export fails, check console logs for missing metadata.

Verification
^^^^^^^^^^^^

After installation, verify CLI and import:

.. code-block:: bash

   playnano --help

Check version from Python:

.. code-block:: bash

   python -c "import playNano; print(playNano.__version__)"

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

There are additionally optional dependencies that can be installed for particular tasks.
To work on the documentation, or run the example notebooks, you can install these as
"extras" using pip.

If you are developing or contributing, install the full set of extras, especially the
dev group (which includes linting and testing tools).

You can also install optional extras for docs, notebooks, or development:

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

This will install:

- Python 3.11 and core scientific libraries (NumPy, SciPy, Pillow, Matplotlib)
- AFM-specific tools: ``afmreader``
- GUI support via ``PySide6``
- Compatibility pins (e.g. ``h5py=3.8.*``) to avoid known issues

.. note::

   The environment uses the ``conda-forge`` channel for reliable binary support across platforms.

.. tip::

   If you modify ``environment.yaml``, you can update your environment with:

   .. code-block:: bash

      conda env update -f environment.yaml --prune
