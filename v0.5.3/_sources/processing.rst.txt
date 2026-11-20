Processing
==========

Overview
--------

The ``playnano.processing`` subpackage provides tools for processing AFM time-series
data for viewing and analysis. It includes functions for levelling, filtering,
masking, alignment, and trimming. These
:doc:`operations <processing-operations-reference>` are modular and composable,
allowing reproducible pipelines tailored to specific datasets and goals.

Processing is coordinated by
:class:`~playnano.processing.pipeline.ProcessingPipeline`, which manages the
reproducible and provenance-tracked transformation of an
:class:`~playnano.afm_stack.AFMImageStack`. A pipeline consists of an ordered list
of steps, where each step defines an operation and its parameters. Steps are executed
sequentially, and all parameters, versions, and outputs are logged for traceability.

This guide covers:

- CLI / GUI usage and examples
- Pipeline structure and step types
- Built-in processing operations and plugin support
- Programmatic usage
- What the pipeline records

See also the :doc:`cli`, :doc:`gui`, and :doc:`analysis` pages for related workflows.

CLI / GUI Usage
---------------

Processing can be applied directly from the CLI (``process`` subcommand) or in the
interactive GUI (``play`` command). In both cases, processing steps are defined using
a semicolon-separated string or a YAML file.

The ``--processing`` argument accepts a semicolon-separated list of steps. Each step is
the name of an operation (a filter, mask, video operation or stack edit), or ``clear`` to
reset the mask. Parameters follow the name after a colon, as comma-separated
``key=value`` pairs. For example:

.. code-block:: bash

   playnano process ./tests/resources/sample_0.h5-jpk \
       --processing "remove_plane;mask_threshold:threshold=1.5;row_median_align;gaussian_filter:sigma=2.0" \
       --export tif,npz \
       --make-gif \
       --output-folder ./results \
       --output-name sample_processed

Parameter values are read as Python values: ``2``, ``-2``, ``2.5`` and ``1e3`` are
numbers, ``true``/``false`` are booleans, ``none`` is ``None``, and lists or tuples
are written in brackets, for example ``drop_frames:indices_to_drop=[0,1]`` or
``intersection_crop:pad=(2,3)``. Anything else is text. To pass text that looks
like a number, quote it: ``name='1'``. ``--analysis-steps`` and values typed into
the ``wizard`` follow the same rules.

Alternatively, pipelines can be defined in YAML for better readability and reuse:

.. code-block:: yaml

   filters:
     - name: remove_plane
     - name: mask_threshold
       threshold: 2
     - name: polynomial_flatten
       order: 2
     - name: gaussian_filter
       sigma: 2.0

In YAML files, write numbers in scientific notation with a dot, such as ``1.0e3``;
YAML reads ``1e3`` as text.

Run it via:

.. code-block:: bash

   playnano process ./tests/resources/sample_0.h5-jpk \
       --processing-file pipeline.yaml \
       --export tif,npz \
       --make-gif \
       --output-folder ./results \
       --output-name sample_processed

Interactive pipeline creation is supported via the ``wizard`` subcommand:

.. code-block:: bash

   playnano wizard ./tests/resources/sample_0.h5-jpk \
       --output-folder ./results \
       --output-name processed_sample

Use ``save <filename-to_save>.yaml`` within the wizard to export the constructed
pipeline as YAML for reuse with ``--processing-file``.

Pipeline Structure and Step Types
---------------------------------

A :class:`~playnano.processing.pipeline.ProcessingPipeline` organizes transformations
into sequential steps applied to an
:class:`~playnano.afm_stack.AFMImageStack`. Each step performs a specific task such
as filtering, masking, or alignment and can be configured with parameters. Steps are
executed in order, and results are tracked with metadata to ensure reproducibility.

After execution, the :attr:`~playnano.afm_stack.AFMImageStack.data` attribute is
updated with the final processed array.

Operation Types
^^^^^^^^^^^^^^^

- **Filters (2D frame operations)** — modify individual frames (e.g. flattening,
  smoothing). Filter functions accept 2D NumPy arrays and return 2D NumPy float arrays.
  Some filters can also accept a mask (if defined by a preceding mask operation)
  and masked regions are excluded automatically from the filter calculation.
- **Masks (2D binary operations)** — generate boolean masks to exclude regions from
  filters or analysis. Masks are combined via logical OR. Use ``clear`` to reset.
- **Video Processing (3D stack operations)** — apply transformations across full
  time-series stacks, such as alignment or drift correction. Operate on 3D arrays and
  may record metadata.
- **Stack Edits (AFMImageStack-level operations)** — remove frames (``drop_frames``,
  ``drop_frame_range``, ``select_frames``, ``drop_final_frames``). The helpers return
  the indices of the frames to drop, and the pipeline removes those frames and updates
  the metadata and timestamps to match.

Step Naming and Provenance
^^^^^^^^^^^^^^^^^^^^^^^^^^

Each step is named as ``step_<index>_<operation_name>`` and its output is stored in
:attr:`~playnano.afm_stack.AFMImageStack.processed` (for data) or
:attr:`~playnano.afm_stack.AFMImageStack.masks` (for binary masks).

Provenance information is stored in ``stack.provenance["processing"]`` (see
:attr:`~playnano.afm_stack.AFMImageStack.provenance`) and includes:

- ``steps`` — ordered list of step records (parameters, versions, timestamps, etc.)
- ``keys_by_name`` — maps operation names to their snapshot keys.

This ensures all transformations are traceable and reproducible.

Processing Operations
---------------------

Built-in Filters & Masks
^^^^^^^^^^^^^^^^^^^^^^^^

Several built-in filters and masks are available. Each takes a NumPy array and optional
parameters, returning either a processed float array (filters) or binary mask array
(masks).

See :doc:`processing-operations-reference` for a full list.

Plugins
^^^^^^^

Custom filters can be added via entry points under ``playnano.filters`` for 2D frame
operations or ``playnano.video_processing`` for 3D stack operations. Any callable
that accepts a 2D or 3D NumPy array and returns a processed array of the same
dimensionality can be registered by defining an entry point in the package's
``pyproject.toml``. This allows seamless integration of custom processing functions
into **playNano** processing pipelines, appearing alongside built-in operations.

Example ``pyproject.toml`` snippet:

.. code-block:: toml

   [project.entry-points."playnano.filters"]
   my_plugin = "my_pkg.module:my_filter"

   [project.entry-points."playnano.video_processing"]
   my_3D_plugin = "my_pkg.module:my_3D_filter"

Example plugin function:

.. code-block:: python

   def my_filter(frame: np.ndarray, **kwargs) -> np.ndarray:
       """Process a 2D array and return a filtered version."""

When the package with this plugin is installed in the same environment as **playNano**,
it appears in the same CLI/API list as the built-in filters.

A repository of optional **playNano** plugins is available at:
https://github.com/derollins/playNano-plugins

Several projects also have the required entry points and can be installed to add new
filters:

- **Python-NanoLocz-Library** (``pnanolocz``) — collection of AFM processing functions,
  including flattening and filtering with automated routines.
  https://github.com/derollins/Python-Nanolocz-Library

Building a Pipeline with the Wizard
-----------------------------------

The processing pipeline can be defined in the CLI and run in the CLI or the GUI.

The **playNano** wizard allows processing pipelines to be built interactively.
To launch this you use the ``wizard`` subcommand followed by a path to the file you
are processing and flags that define the output folder and file name (see :doc:`cli`).
Once built the pipeline can be saved as yaml file that can be used in future runs or
run immediately within the wizard.

Run the wizard with:

.. code-block:: bash

  playnano wizard ./tests/resources/sample_0.h5-jpk --output-folder ./results --output-name processed_sample

Once the data is loaded, use the ``add`` command followed by the name of a filter, mask
or video operation to add steps to the pipeline. The wizard will then prompt you to enter optional or
required parameters. Once the pipeline is complete use the ``save`` with the path to a ``.yaml``
file to save the pipeline.

Once constructed and saved the processing pipeline that has been built can be run with the
``run`` command which will run the processing pipeline, step-by-step, with the configured
parameters. The wizard will then ask if you would like to export the processed data as ``.npz``,
``.h5`` or ``.ome-tiff`` and then if you would like to generate a ``.gif``.

Programmatic Usage
------------------

Use the :class:`~playnano.processing.pipeline.ProcessingPipeline` class directly for
custom pipelines:

.. code-block:: python

   from playnano.afm_stack import AFMImageStack
   from playnano.processing.pipeline import ProcessingPipeline

   stack = AFMImageStack.load_data("data/sample.h5-jpk", channel="height_trace")

   pipeline = ProcessingPipeline(stack)
   pipeline.add_filter("remove_plane")
   pipeline.add_mask("mask_threshold", threshold=2.0)
   pipeline.add_filter("gaussian_filter", sigma=1.0)
   pipeline.run()   # updates stack.processed and stack.data

After execution, the processed frames are available via ``stack.data``, and intermediate
snapshots can be accessed through ``stack.processed``.


Saved data & exports
--------------------

The processing system supports exporting processed results and snapshots to:

- **OME-TIFF** - multi-frame TIFF, compatible with ImageJ/Fiji.
- **NPZ** - numpy zipped archive containing arrays and metadata.
- **HDF5** - self-contained bundle including data, processed snapshots and provenance.
- **GIF** - annotated animated GIF (frame timestamps are drawn when available).

Use the CLI flags ``--export``, ``--make-gif``, ``--output-folder`` and ``--output-name`` to
control export behaviour (See :doc:`cli` for CLI flag details).

What the pipeline records
^^^^^^^^^^^^^^^^^^^^^^^^^

After execution, the following are available:

- ``stack.processed`` — processed frame snapshots keyed by step name
- ``stack.masks`` — boolean masks keyed by step name
- ``stack.provenance["processing"]`` — full step records and mappings
- ``stack.provenance["environment"]`` — runtime metadata (Python/OS/package versions)

These enable complete reproducibility and intermediate inspection.

Inspecting Results Programmatically
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After running a processing pipeline, processed arrays, masks, and provenance information
are stored directly on the :class:`~playnano.afm_stack.AFMImageStack` object.
You can use the following commands to explore what was generated:

.. code-block:: python

   print(sorted(stack.processed.keys()))
   print(sorted(stack.masks.keys()))

   for step in stack.provenance["processing"]["steps"]:
       print(step["index"], step["step_type"], step["name"])

   for key in stack.provenance["processing"]["keys_by_name"].get("polynomial_flatten", []):
       arr = stack.processed[key]

.. code-block:: text

  ['raw', 'step_1_remove_plane', 'step_2_polynomial_flatten']
  ['step_3_mask_threshold']
  1 filter remove_plane
  2 filter polynomial_flatten
  3 mask mask_threshold

This output is for a pipeline of ``remove_plane``, ``polynomial_flatten`` and
``mask_threshold``: two filters and one mask, applied in sequence, plus the ``raw``
snapshot of the data before processing.
You can access a specific result directly:

.. code-block:: python

   flattened = stack.processed["step_2_polynomial_flatten"]
   print(flattened.shape)

which returns a NumPy array representing the processed frame or stack at that step.

Tips & Troubleshooting
----------------------

- If a ``raw`` snapshot is missing, check if it was loaded from an existing bundle.
- If a plugin does not appear in the CLI, verify that its entry point group is
  ``playnano.filters`` (2D frame operations) or ``playnano.video_processing``
  (3D stack operations).
- For large datasets, prefer exporting HDF5 bundles instead of large JSON logs.

See also
^^^^^^^^

- :doc:`processing-operations-reference` — list of all built-in operations
- :doc:`cli` — command-line usage
- :doc:`gui` — interactive GUI overview
- :doc:`exporting` — export formats and options
- :doc:`analysis` — analysis pipelines and provenance
