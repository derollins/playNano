Processing
==========

Overview
--------

The ``playNano.processing`` subpackage provides tools for preparing AFM time-series
data for viewing and analysis. It includes functions for levelling, filtering,
masking, alignment, and trimming. These
:doc:`operations <processing-operations-reference>` are modular and composable,
allowing reproducible pipelines tailored to specific datasets and goals.

Processing is coordinated by
:class:`~playNano.processing.pipeline.ProcessingPipeline`, which manages the
reproducible and provenance-tracked transformation of an
:class:`~playNano.afm_stack.AFMImageStack`. A pipeline consists of an ordered list
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

The ``--processing`` argument accepts a semicolon-separated list of steps. Each step
is either a data operation, a mask generator, with optional parameters passed via
colon-separated key-value pairs or ``clear`` that resets masks. For example:

.. code-block:: bash

   playnano process ./tests/resources/sample_0.h5-jpk \
       --processing "remove_plane;threshold_mask:threshold=1.5;row_median_align;gaussian_filter:sigma=2.0" \
       --export tif,npz \
       --make-gif \
       --output-folder ./results \
       --output-name sample_processed

Alternatively, pipelines can be defined in YAML for better readability and reuse:

.. code-block:: yaml

   filters:
     - name: remove_plane
     - name: threshold_mask
       threshold: 2
     - name: polynomial_flatten
       order: 2
     - name: gaussian_filter
       sigma: 2.0

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

A :class:`~playNano.processing.pipeline.ProcessingPipeline` organizes transformations
into sequential steps applied to an
:class:`~playNano.afm_stack.AFMImageStack`. Each step performs a specific task such
as filtering, masking, or alignment and can be configured with parameters. Steps are
executed in order, and results are tracked with metadata to ensure reproducibility.

After execution, the :attr:`~playNano.afm_stack.AFMImageStack.data` attribute is
updated with the final processed array.

Operation Types
^^^^^^^^^^^^^^^

- **Filters (2D frame operations)** — modify individual frames (e.g. flattening,
  smoothing). Accept 2D NumPy arrays and return float arrays. Masked regions (if
  defined by a preceding mask operation) are excluded automatically.
- **Masks (2D binary operations)** — generate boolean masks to exclude regions from
  filters or analysis. Masks are combined via logical OR. Use ``clear`` to reset.
- **Video Processing (3D stack operations)** — apply transformations across full
  time-series stacks, such as alignment or drift correction. Operate on 3D arrays and
  may record metadata.
- **Stack Edits (AFMImageStack-level operations)** — modify dataset structure (e.g.
  cropping, frame removal). Return a new 3D array, with metadata and timestamps
  updated automatically.

Step Naming and Provenance
^^^^^^^^^^^^^^^^^^^^^^^^^^

Each step is named as ``step_<index>_<operation_name>`` and its output is stored in
:attr:`~playNano.afm_stack.AFMImageStack.processed` (for data) or
:attr:`~playNano.afm_stack.AFMImageStack.masks` (for binary masks).

Provenance information is stored in
:attr:`~playNano.afm_stack.AFMImageStack.provenance["processing"]` and includes:

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

Custom filters can be added via entry points under ``playNano.filters``. Any callable
that accepts a 2D NumPy array and returns a processed array can be registered.

Example ``pyproject.toml`` snippet:

.. code-block:: toml

   [project.entry-points."playNano.filters"]
   my_plugin = "my_pkg.module:my_filter"

Example plugin function:

.. code-block:: python

   def my_filter(frame: np.ndarray, **kwargs) -> np.ndarray:
       """Process a 2D array and return a filtered version."""


When the plugin is installed, it appears in the same CLI/API list as the
built-in filters.

CLI / GUI Usage
---------------

The processing pipeline can defined in the CLI and run in the CLI or the GUI.

The **playNano** wizard allows processing pipelines to be built interactively.
To launch this you use the ``wizard`` subcommand followed by a path to the file you
are processing and flags that define the output folder and file name (see :doc:`cli`).
Once built the pipeline can be saved as yaml file that can be used in future runs or
run immediately within the wizard.

Run the wizard with:

.. code-block:: bash

  playnano wizard .test/resources/sample_0.h5-jpk --output-folder ./results --output-name processed_sample

Once the data is loaded, use the ``add`` command followed by the name of a filter, mask
or mask to add steps to the pipeline. The wizard will then prompt you to enter optional or
required parameters. Once the pipeline is complete use the ``save`` with the path to a ``.yaml``
file to save the pipeline.

Once constructed and saved the processing pipeline that has been built can be run with the
``run`` command which will run the processing pipeline, step-by-step, with the configured
parameters. The wizard will then ask if you would like to export the processed data as ``.npz``,
``.h5`` or ``.ome-tiff`` and then if you would like to generate a ``.gif``.

Programmatic usage
       """Process a 2D array and return a filtered version."""
       ...

Installed plugins appear alongside built-in filters in the CLI and GUI.

Programmatic Usage
------------------

Use the :class:`~playNano.processing.pipeline.ProcessingPipeline` class directly for
custom pipelines:

.. code-block:: python

   from playNano.afm_stack import AFMImageStack
   from playNano.processing.pipeline import ProcessingPipeline

   stack = AFMImageStack.load_afm_stack("data/sample.h5-jpk", channel="height_trace")

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
- **GIF** - annotated animated GIF (requires timing metadata for correct frame rates).

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
are stored directly on the :class:`~playNano.afm_stack.AFMImageStack` object.
You can use the following commands to explore what was generated:

.. code-block:: python

   print(sorted(stack.processed.keys()))
   print(sorted(stack.masks.keys()))

   for step in stack.provenance["processing"]["steps"]:
       print(step["index"], step["step_type"], step["name"])

   for key in stack.provenance["processing"]["keys_by_name"].get("polynomial_flatten", []):
       arr = stack.processed[key]

.. code-block:: text

  ['step_1_remove_plane', 'step_2_polynomial_flatten']
  ['step_3_threshold_mask']
  1 filter remove_plane
  2 filter polynomial_flatten
  3 mask threshold_mask

This indicates two filters and one mask were applied in sequence.
You can access a specific result directly:

.. code-block:: python

flattened = stack.processed["step_2_polynomial_flatten"]
print(flattened.shape)

which returns a NumPy array representing the processed frame or stack at that step.

Tips & Troubleshooting
----------------------

- If a ``raw`` snapshot is missing, check if it was loaded from an existing bundle.
- If a plugin does not appear in the CLI, verify that its entry point group is
  ``playNano.filters``.
- For large datasets, prefer exporting HDF5 bundles instead of large JSON logs.

See also
^^^^^^^^

- :doc:`processing-operations-reference` — list of all built-in operations
- :doc:`cli` — command-line usage
- :doc:`gui` — interactive GUI overview
- :doc:`exporting` — export formats and options
- :doc:`analysis` — analysis pipelines and provenance
