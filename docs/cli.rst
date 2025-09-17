Command Line Interface (CLI)
============================

The primary entrypoint for playNano is the :command:`playnano` command-line tool.
It provides interactive playback (GUI), batch processing/export, analysis runs,
and an interactive wizard for building pipelines.

General usage
-------------

.. code-block:: bash

   playnano <command> <input_file> [options]

Run :command:`playnano --help` to see global options and a list of subcommands.

Available subcommands
^^^^^^^^^^^^^^^^^^^^^

- ``play``    - Launch the interactive GUI viewer.
- ``process`` - Batch mode: apply filters and export images/bundles/GIFs.
- ``analyze`` - Run a pipeline of analysis modules (detection/tracking) on a stack and export results.
- ``wizard``  - Interactive REPL for constructing processing/analysis pipelines.
- ``env-info``- Print environment information useful for debugging.

The CLI maps to the Python entry point ``playNano.cli.entrypoint:main``.

Batch mode operations
---------------------

Processing and analysis can be run in a non-interactive batch mode without a GUI or wizard.

Batch processing mode
^^^^^^^^^^^^^^^^^^^^^

Apply filters and export without user interaction.

.. code-block:: bash

   playnano process /path/to/sample.h5-jpk \
     [--channel CHANNEL] \
     [--processing "remove_plane;row_median_align"] \
     [--processing-file pipeline.yaml] \
     [--export tif,npz,h5] \
     [--make-gif] \
     [--output-folder OUTPUT_DIR] \
     [--output-name BASE_NAME] \
     [--scale-bar-nm SCALE_BAR_INT] \
     [--zmin MINIMUM] \
     [--zmax MAXIMUM]

Primary options
~~~~~~~~~~~~~~~

- ``--channel`` (default: ``height_trace``) - channel name to load from the file.
- ``--processing`` - semi-colon delimited inline pipeline string.
- ``--processing-file`` - YAML/JSON file describing the processing pipeline.
- ``--export`` - comma-separated list of formats to write: ``tif``, ``npz``, ``h5``.
- ``--make-gif`` - write an animated GIF with current annotations.
- ``--output-folder`` / ``--output-name`` - export location and basename.
- ``--scale-bar-nm`` - integer length (nm) for scale bar in GIF (0 disables).
- ``--zmin`` / ``--zmax`` - initial z-range; can be a float or ``auto`` (1st/99th percentiles).

.. note::

   ``--processing`` and ``--processing-file`` are mutually exclusive.
   Use one or the other (or neither to run with no processing).

Processing pipeline schema
~~~~~~~~~~~~~~~~~~~~~~~~~~

Example pipeline YAML (see :doc:`processing` for details):

.. code-block:: yaml

   filters:
     - name: remove_plane
     - name: gaussian_filter
       sigma: 2.0
     - name: threshold_mask
       threshold: 2
     - name: polynomial_flatten
       order: 2

Batch analysis mode
^^^^^^^^^^^^^^^^^^^

Run an analysis pipeline on an loaded AFM stack and export the results.

.. code-block:: bash

   playnano analyze /path/to/processed_sample.h5 \
     [--channel CHANNEL] \
     (--analysis-steps "detect_particles:threshold=5;track_particles:max_distance=3.0" \
      | --analysis-file analysis.yaml) \
     [--output-folder OUTPUT_DIR] \
     [--output-name BASE_NAME]

Analysis options
~~~~~~~~~~~~~~~~

- ``--analysis-steps`` - semicolon-delimited inline steps (example below).
- ``--analysis-file`` - YAML/JSON file specifying the analysis pipeline.
- ``--output-folder`` / ``--output-name`` - export location and basename.

Inline example:

.. code-block:: bash

   playnano analyze sample_flat.h5 \
     --analysis-steps "detect_particles:threshold=4.5;track_particles:max_distance=2.5" \
     --output-folder ./analysis_results --output-name run1

Analysis pipeline schema
~~~~~~~~~~~~~~~~~~~~~~~~

Example YAML (see :doc:`analysis` for full reference):

.. code-block:: yaml

   analysis:
     - name: detect_particles
       mask_fn: mask_threshold
       threshold: 5
     - name: track_particles
       max_distance: 3.0

**Outputs**

- ``<output>.json`` - sanitized analysis record (suitable for downstream parsing).
- ``<output>.h5`` - full analysis bundle (if HDF5 export requested).

Wizard mode (interactive)
-------------------------

The wizard opens a small REPL to interactively build and run processing and
analysis pipelines. Useful for experimentation and creating shareable YAML
configurations.

.. code-block:: bash

   playnano wizard /path/to/sample.h5-jpk --output-folder ./results --output-name demo

Common REPL commands
^^^^^^^^^^^^^^^^^^^^

Processing pipeline commands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``add <filter>`` - add a processing step (REPL prompts for parameters).
- ``list`` - show the current processing pipeline steps.
- ``save <path.yaml>`` - save the current processing pipeline as YAML.
- ``run`` - execute the processing pipeline on the supplied input.

Analysis pipeline commands (prefixed with ``a``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``aadd <analysis_step>`` - add an analysis step (REPL prompts for parameters).
- ``alist`` - show the current analysis pipeline steps.
- ``asave <path.yaml>`` - save the current analysis pipeline as YAML.
- ``arun`` - run the analysis pipeline (runs any processing steps as configured if
    AFM stack contains only raw data).

Other utility commands
~~~~~~~~~~~~~~~~~~~~~~

- ``help`` - show available REPL commands.
- ``quit`` - exit the wizard (this is a single, global quit that closes the entire session).

Play (GUI) mode
---------------

Open the PySide6 GUI viewer:

.. code-block:: bash

   playnano play /path/to/sample.h5-jpk \
     [--channel CHANNEL] \
     [--processing "remove_plane;row_median_align"] \
     [--processing-file pipeline.yaml] \
     [--output-folder OUTPUT_DIR] \
     [--output-name BASE_NAME] \
     [--scale-bar-nm SCALE_BAR_INT] \
     [--zmin MINIMUM] \
     [--zmax MAXIMUM]

GUI highlights
^^^^^^^^^^^^^^

- Playback controls (play/pause, FPS slider, frame slider)
- Toggle raw vs processed views and apply processing on demand
- Z-scale histogram with draggable zmin/zmax lines and numeric spinboxes
- Export panel: NPZ, OME-TIFF, HDF5 and GIF export options
- Keyboard shortcuts: ``Space`` (play/pause), ``F`` (apply filters), ``R`` (toggle raw/processed), ``G`` (export GIF), ``E`` (export selected formats)

Notes about z-range
~~~~~~~~~~~~~~~~~~~

- ``--zmin`` / ``--zmax`` accept a float or the string ``auto`` (default).
  When ``auto`` is used values are computed as the 1st and 99th percentiles of the stack.

env-info
~~~~~~~~

Prints environment and dependency information to help debugging and issue reports:

.. code-block:: bash

   playnano env-info

Help & usage
------------

- Subcommand help:

  .. code-block:: bash

     playnano <subcommand> --help

  Example:

  .. code-block:: bash

     playnano process --help

- Global help:

  .. code-block:: bash

     playnano --help

Tips & troubleshooting
----------------------

- If the CLI command is not found, ensure your environment is activated and that
  you installed the package (``pip install -e .``) in that environment.

- If GIF export fails, check the input metadata (e.g. ``line_rate``) and the console logs.

- For Windows users, installing **PySide6** via ``conda install -c conda-forge pyside6`` can reduce platform-specific issues.

Links
-----

- :doc:`processing` - processing pipeline reference and YAML schema
- :doc:`analysis` - analysis pipeline reference and YAML schema
- :doc:`installation` - installation instructions and troubleshooting
