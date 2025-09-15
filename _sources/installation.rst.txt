Installation
============

This page describes recommended installation paths for **playNano**.
Conda is the recommended default because it gives the most reliable binary
support for scientific packages and Qt/PySide.

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

Create a reproducible conda environment and install **playNano** in editable mode,
allowing you to make changes and test them immediately during development.

.. note::

   To use the recommended conda-based installation, first install either
   `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ or
   `Anaconda <https://www.anaconda.com/products/distribution>`_, which provide
   the ``conda`` package manager. Miniconda is lightweight and ideal for custom
   setups, while Anaconda includes a full suite of scientific packages out of the box.

.. code-block:: bash

   # 1) Create and activate a conda env
   conda create -n playnano python=3.11
   conda activate playnano

   # 2) Install the package (from the project root)
   pip install -e .

If you need PySide6 and prefer conda packages for Qt:

.. code-block:: bash

   conda install -c conda-forge pyside6

Install optional extras (examples):

.. code-block:: bash

   pip install -e ".[docs]"       # docs build dependencies (Sphinx, theme, nbsphinx)
   pip install -e ".[notebooks]"  # notebook/demo dependencies (Jupyter)

Alternative: pip + venv
^^^^^^^^^^^^^^^^^^^^^^^

If you prefer the standard library virtualenv workflow, use ``venv``:

.. code-block:: bash

   python -m venv .venv
   # Linux / macOS
   source .venv/bin/activate
   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1

   pip install -e .

Install via environment.yaml
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you prefer a reproducible setup using a pre-defined environment file, you can
use the provided ``environment.yaml`` to create a conda environment with all required
dependencies.

.. code-block:: bash

   conda env create -f environment.yml
   conda activate playnano_env

This will install:

- Python 3.11 and core scientific libraries (NumPy, SciPy, Pillow, Matplotlib)
- AFM-specific tools: ``afmreader``
- GUI support via ``PySide6``
- Compatibility pins (e.g. ``h5py=3.8.*``) to avoid known issues

.. note::

   The environment uses the ``conda-forge`` channel for reliable binary support across platforms.

.. tip::

   If you modify ``environment.yml``, you can update your environment with:

   .. code-block:: bash

      conda env update -f environment.yml --prune

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

Developer / contributor install
-------------------------------

Developer install (linting/tests/docs extras):

.. code-block:: bash

   pip install -e ".[dev,docs,notebooks]"

Run tests:

.. code-block:: bash

   pytest -q

Build the docs locally:

.. code-block:: bash

   make -C docs html
