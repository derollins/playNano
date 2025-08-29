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

Quick Install (recommended: conda)
----------------------------------

Create a reproducible conda environment and install **playNano** in editable
mode (so you can develop / iterate).

.. code-block:: bash

   # 1) Create and activate a conda env (use conda-forge for best binary support)
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
-----------------------

If you prefer the standard library virtualenv workflow, use ``venv``:

.. code-block:: bash

   python -m venv .venv
   # Linux / macOS
   source .venv/bin/activate
   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1

   pip install -e .

Environment YAML (conda) — example
----------------------------------

You can also provide an ``environment.yml`` for reproducible installs. Example:

.. code-block:: yaml

   name: playnano
   channels:
     - conda-forge
   dependencies:
     - python=3.11
     - numpy>=1.23,<2.0
     - pandas
     - h5py
     - pillow
     - tifffile
     - matplotlib
     - afmreader
     - python-dateutil>=2.8
     - scipy
     - scikit-learn
     - pyyaml>=6.0
     - pyside6>=6.5
     - pip
     - pip:
       - -e .

Install from Git / specific branch
----------------------------------

To install playNano directly from GitHub (useful for CI or testing a branch):

.. code-block:: bash

   pip install -e "git+https://github.com/derollins/playNano.git@main#egg=playNano"

Special notes & troubleshooting
-------------------------------

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
------------

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

Summary checklist
-----------------

- Use **conda** (conda-forge) as the first recommendation for users.
- Use editable install ``pip install -e .`` for development.
- Install ``pyside6`` via conda on Windows if you see issues.
- Install AFMReader from GitHub if not on PyPI.
