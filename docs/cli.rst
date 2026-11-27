Command Line Interface (CLI)
============================

The primary entrypoint for playNano is the :command:`playnano` command-line tool.
It provides interactive playback (GUI), batch processing/export, analysis runs,
and an interactive wizard for building pipelines.

General usage
-------------

.. code-block:: bash

   playnano [global options] <command> <input_file> [command options]

Run :command:`playnano --help` to see global options and a list of subcommands.

Available subcommands
^^^^^^^^^^^^^^^^^^^^^

- ``play``    - Launch the interactive GUI viewer.
- ``process`` - Batch mode: apply filters and export images/bundles/GIFs.
- ``analyze`` - Run a pipeline of analysis modules (detection/tracking) on a stack and export results.
- ``wizard``  - Interactive REPL for constructing processing/analysis pipelines.
- ``env-info`` - Print environment information useful for debugging.

The CLI maps to the Python entry point ``playnano.cli.entrypoint:main``.

Quick examples
^^^^^^^^^^^^^^

.. code-block:: bash

   playnano play data.h5-jpk
   playnano process data.ARIS --channel HeightTrace --processing "remove_plane" --make-gif
   playnano analyze processed.h5 --analysis-file analysis.yaml

The default channel, ``height_trace``, is a JPK channel name. For other formats, give the
file's own name with ``--channel``, for example ``HeightTrace`` for ``.aris`` files; a wrong
name gives an error listing the channels the file has.

Global options
--------------

These options apply to **all subcommands** (or can be used on their own).

``--version``
^^^^^^^^^^^^^

Print the installed playNano version and exit.

.. code-block:: bash

   playnano --version

Example output:

.. code-block:: text

   playnano 0.5.3

(The version shown is the one you have installed.)

This is useful for debugging, reporting issues, and reproducibility.

``--log-level``
^^^^^^^^^^^^^^^

Set the global logging verbosity for the CLI.

.. code-block:: bash

   playnano --log-level DEBUG play data.h5-jpk
   playnano --log-level WARNING process data.h5-jpk --make-gif

Allowed values are:

- ``DEBUG``   - Verbose output for development and debugging
- ``INFO``    - Standard operational output (default)
- ``WARNING`` - Only warnings and errors
- ``ERROR``   - Errors only

If not specified, the default log level is ``INFO``.

.. note::

   Global options must be specified **before** the subcommand name.

Batch mode operations
---------------------

Processing and analysis can be run in a non-interactive batch mode without a GUI or wizard.

Batch processing mode - ``process``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Apply filters and export without user interaction.

.. code-block:: bash

   playnano process /path/to/sample.h5-jpk \
     [--channel CHANNEL] \
     [--processing "remove_plane;row_median_align"] \
     [--processing-file pipeline.yaml] \
     [--export tif,npz,h5] \
     [--make-gif] \
     [--make-video mp4,avi] \
     [--make-sequence] \
     [--output-folder OUTPUT_DIR] \
     [--output-name BASE_NAME] \
     [--scale-bar-nm SCALE_BAR_INT] \
     [--zmin MINIMUM] \
     [--zmax MAXIMUM] \
     [--draw-ts] \
     [--cmap COLORMAP_NAME] \
     [--fps FRAMES_PER_SECOND]

Primary options
~~~~~~~~~~~~~~~

- ``--channel`` (default: ``height_trace``) - channel name to load from the file.
- ``--processing`` - semi-colon delimited inline pipeline string.
- ``--processing-file`` - YAML/JSON file describing the processing pipeline.
- ``--export`` - comma-separated list of formats to write: ``tif``, ``npz``, ``h5``.
- ``--make-gif`` - write an animated GIF with current annotations.
- ``--make-video`` - write an annotated video; specify formats (e.g. ``mp4``, ``avi``).
- ``--make-sequence`` - write an annotated image sequence (PNG files).
- ``--output-folder`` / ``--output-name`` - export location and basename.
- ``--scale-bar-nm`` - integer length (nm) for scale bar in GIF (0 disables).
- ``--zmin`` / ``--zmax`` - initial z-range; can be a float or ``auto`` (1st/99th percentiles).
- ``--cmap`` - name of colormap to use for animated visualisation export (default=afm_brown).
- ``--fps`` - frame rate, in frames per second, for GIF and video exports (default 5).
- ``--draw-ts`` - draw each frame's timestamp on the GIF, video or image sequence (off by default).

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
     - name: mask_threshold
       threshold: 2
     - name: polynomial_flatten
       order: 2

Batch analysis mode
^^^^^^^^^^^^^^^^^^^

Run an analysis pipeline on a loaded AFM stack and export the results.

.. code-block:: bash

   playnano analyze /path/to/processed_sample.h5 \
     [--channel CHANNEL] \
     (--analysis-steps "feature_detection:mask_fn=mask_threshold,threshold=5;particle_tracking:max_distance=3.0" \
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
     --analysis-steps "feature_detection:mask_fn=mask_threshold,threshold=4.5;particle_tracking:max_distance=2.5" \
     --output-folder ./analysis_results --output-name run1

Analysis pipeline schema
~~~~~~~~~~~~~~~~~~~~~~~~

Example YAML (see :doc:`analysis` for full reference):

.. code-block:: yaml

   analysis:
     - name: feature_detection
       mask_fn: mask_threshold
       threshold: 5
     - name: particle_tracking
       max_distance: 3.0

**Outputs**

Both files are always written:

- ``<output>.json`` - sanitized analysis record (suitable for downstream parsing).
- ``<output>.h5`` - full analysis record in HDF5.

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
- ``clear`` - add a step that clears the current mask.
- ``remove <index>`` - remove a step (indices start at 1, as shown by ``list``).
- ``move <old_index> <new_index>`` - move a step to a new position.
- ``list`` - show the current processing pipeline steps.
- ``save <path.yaml>`` - save the current processing pipeline as YAML.
- ``load <path.yaml>`` - replace the pipeline with one saved by ``save``.
- ``run`` - execute the processing pipeline on the supplied input.

Analysis pipeline commands (prefixed with ``a``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``aadd <analysis_step>`` - add an analysis step (REPL prompts for parameters), or
  ``aadd <step>:<param>=<value>,...`` to give the parameters inline.
- ``aremove <index>`` and ``amove <old_index> <new_index>`` - remove or move a step.
- ``alist`` - show the current analysis pipeline steps.
- ``asave <path.yaml>`` - save the current analysis pipeline as YAML.
- ``aload <path>`` - load an analysis pipeline from YAML or JSON.
- ``arun`` - run the analysis pipeline (runs any processing steps first if the
  AFM stack contains only raw data).

Other utility commands
~~~~~~~~~~~~~~~~~~~~~~

- ``help`` - show available REPL commands.
- ``quit`` (or ``exit``) - exit the wizard (this is a single, global quit that closes the entire session).

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
     [--zmax MAXIMUM] \
     [--cmap COLORMAP_NAME] \
     [--fps INITIAL_FRAMES_PER_SECOND]

GUI highlights
^^^^^^^^^^^^^^

- Playback controls (play/pause, FPS, frame slider)
- Toggle raw vs processed views and apply processing on demand
- Z-scale histogram with draggable zmin/zmax lines and numeric spinboxes
- Export panel: NPZ, OME-TIFF, HDF5 and animated export options
- Keyboard shortcuts: ``Space`` (play/pause), ``F`` (apply filters), ``R`` (toggle raw/processed), ``G`` (export GIF), ``E`` (export selected formats)

Notes about z-range
~~~~~~~~~~~~~~~~~~~

- ``--zmin`` / ``--zmax`` accept a float or the string ``auto`` (default).
  When ``auto`` is used values are computed as the 1st and 99th percentiles of the stack.
  In ``play``, a value that is neither is replaced by ``auto`` with a warning, since the
  limits can be changed in the window.


Find out more on the GUI docs page: :doc:`gui`

env-info
--------

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

- If an export fails, rerun with ``--log-level DEBUG`` to see the full error in the console log.

- For GUI installation problems, see the PySide6 notes in :doc:`installation`.

Links
-----

- :doc:`processing` - processing pipeline reference and YAML schema
- :doc:`analysis` - analysis pipeline reference and YAML schema
- :doc:`installation` - installation instructions and troubleshooting
