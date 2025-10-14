Processing
==========

The ``playNano.processing`` subpackage provides tools for flattening,
filtering and masking AFM image stacks. Processing is applied frame-by-frame
to stacks shaped ``(n_frames, height, width)`` so that snapshots and
per-step provenance are retained while preserving stack shape.

This page covers:
- quick start and CLI examples
- common filters and masks
- how to supply pipelines (inline or YAML)
- programmatic usage
- a concise summary of what the pipeline records

See also the :doc:`cli`, :doc:`gui` and :doc:`analysis` pages.

Quick start
-----------

Processing can be applied directly in a batch mode using the ``process`` subcommand.
This can be used to apply a series of filters and export the results in the CLI:

.. code-block:: bash

   playnano process ./tests/resources/sample_0.h5-jpk \
       --processing "remove_plane;threshold_mask:threshold=1.5;row_median_align;gaussian_filter:sigma=2.0" \
       --export tif,npz \
       --make-gif \
       --output-folder ./results \
       --output-name sample_processed

Or use a YAML pipeline file:

.. code-block:: yaml

   filters:
     - name: remove_plane
     - name: threshold_mask
       threshold: 2
     - name: polynomial_flatten
       order: 2
     - name: gaussian_filter
       sigma: 2.0

.. code-block:: bash

   playnano process ./tests/resources/sample_0.h5-jpk --processing-file pipeline.yaml

Concepts & behaviour
--------------------

- Processing operations are **2D functions** applied to each frame independently.
- Supported step types:
  - **Filters** - modify image data (flattening, smoothing, alignment).
  - **Masks** - boolean masks used to exclude regions from subsequent filters.
  - **Plugins** - third-party filters registered via entry points.
- The pipeline maintains snapshots for raw and intermediate results and records
  detailed provenance for reproducibility.
- After a pipeline run the pipeline **updates ``stack.data``** to the final
  processed array (so downstream code sees processed frames by default).

Built-in filters and masks
--------------------------

There are a number of built in functions that can be used to process AFM data.

These functions take a numpy array as a argument along with any parameters and
return a numpy array. In the case of the filters this is an array of floats while
the mask functions output a binary array.

Certain filters (e.g. ``remove_plane``, ``row_median_align``) support masked computation.
When a binary mask is provided, the operation is applied to the full image, but its internal
parameters are estimated only from unmasked pixels. This is useful when regions of the image
contain artifacts, noise, or irrelevant features that should not influence the operation,
but the correction itself must be applied globally (i.e. flattening based on background pixels).

The output of each function is saved as a step and the masks can  also be used in analysis
pipelines.

Filters
^^^^^^^

- ``remove_plane`` - fit and subtract a 2D plane (useful for tilt removal).
- ``polynomial_flatten`` - fit & subtract a 2D polynomial surface.
  - parameter: ``order`` (int, default: 2)
- ``row_median_align`` - subtract median per row to remove horizontal banding.
- ``zero_mean`` - subtract global mean (centres data around zero or background around zero if a foreground
    mask is applied).
- ``gaussian_filter`` - gaussian smoothing.
  - parameter: ``sigma`` (float, default: 1.0)

Masks
^^^^^

- ``mask_threshold`` - mask values above ``threshold``.
  - parameter: ``threshold`` (float, default: 0.0)
- ``mask_below_threshold`` - mask values below ``threshold``.
  - parameter: ``threshold`` (float, default: 0.0)
- ``mask_mean_offset`` - mask values beyond mean ± factor x std.
  - parameter: ``factor`` (float, default: 1.0)
- ``mask_morphological`` - threshold + morphological closing (structure size param).
  - parameter: ``threshold`` (float)
  - parameter: ``structure_size`` (int, default= 3)
- ``mask_adaptive`` - block-wise adaptive thresholding (``block_size``, ``offset``).
  - parameter: ``block_size`` (int, default: 5)
  - parameter: ``offset`` (float, default: 0.0)

.. note::
   Masks are combined using logical OR (new masks overlay previous ones).
   Use the ``clear`` step to reset masks.

Plugins
^^^^^^^

Extend the pipeline by registering filter functions via entry points under
``playNano.filters``. This can be any callable that accepts a 2D numpy array
with optional parameters and returns a processed 2D array.

Example `pyproject.toml` fragment:

.. code-block:: toml

   [project.entry-points."playNano.filters"]
   my_plugin = "my_pkg.module:my_filter"

Plugin signature:

.. code-block:: python

   def my_filter(frame: np.ndarray, **kwargs) -> np.ndarray:
       """
       Accepts a 2D array (frame) and returns a processed 2D array.
       """

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
------------------

The processing pipeline can be used programmatically via the
:class:`~playNano.processing.pipeline.ProcessingPipeline` class, which operates
on a :class:`~playNano.afm_stack.AFMImageStack` object. Use the ``add_filter()`` and
``add_mask()`` methods to build the pipeline step-by-step, and call ``run()``
to execute it.

Build and run a pipeline from Python:

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

After a run the following are available on the :class:`~playNano.afm_stack.AFMImageStack`:

- ``stack.processed`` : dict
  - Snapshots keyed by step name (see detailed section below) and raw data preserved in ``raw``.
- ``stack.masks`` : dict
  - Boolean mask snapshots keyed by step name.
- ``stack.provenance["processing"]`` : dict
  - ``steps`` : ordered list of per-step provenance records.
  - ``keys_by_name`` : mapping of step names to created snapshot keys.
- ``stack.provenance["environment"]`` : metadata about OS / Python / package versions.

These records enable reproducibility and inspection of intermediate results.

Advanced / Implementation details
---------------------------------

Snapshot key naming
^^^^^^^^^^^^^^^^^^^

- Processed snapshot keys use the pattern::

   step_<idx>_<step_name>

  where ``idx`` is 1-based step index and ``step_name`` is the invoked step
  name with spaces replaced by underscores. A ``"raw"`` snapshot is created
  automatically (if missing) before the first processing step.

- Mask snapshots are stored under ``stack.masks`` with similar keys. When
  masks are overlaid the new mask key concatenates the previous mask suffixes
  to preserve lineage.

Provenance record structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For each run ``stack.provenance["processing"]`` is rebuilt and contains:

- ``steps`` - list of dicts, each with fields:
  - ``index`` : int (1-based)
  - ``name`` : str (step name)
  - ``params`` : dict (keyword args passed)
  - ``timestamp`` : ISO-8601 UTC timestamp
  - ``step_type`` : ``"filter"``, ``"mask"``, ``"clear"`` or ``"plugin"``
  - ``version`` : optional version string if provided via a decorator or plugin metadata
  - ``function_module`` : Python module path (where the function lives)
  - *If mask*: ``mask_key`` and a concise ``mask_summary`` (shape/dtype)
  - *If filter/plugin*: ``processed_key`` and an ``output_summary``

- ``keys_by_name`` - dict mapping step name to ordered list of created keys.

Other notes
^^^^^^^^^^^

- Indexing in snapshot keys is **1-based** (``step_1_*`` is the first applied step).
- After pipeline completion ``stack.data`` is overwritten with the final processed
  array so that subsequent consumers use the processed frames by default.
- When you export a bundle (HDF5/NPZ) the provenance and snapshots are included.
- If you pass ``log_to`` to programmatic run helpers, large arrays are sanitized
  (summarized) for JSON-friendly logging.

Inspecting results programmatically
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # list snapshots
   print(sorted(stack.processed.keys()))
   print(sorted(stack.masks.keys()))

   # walk provenance
   for step in stack.provenance["processing"]["steps"]:
       print(step["index"], step["step_type"], step["name"], step.get("processed_key") or step.get("mask_key"))

   # retrieve results produced by a named step
   for key in stack.provenance["processing"]["keys_by_name"].get("polynomial_flatten", []):
       arr = stack.processed[key]
       # do stuff...

Tips & troubleshooting
----------------------

- If you expect a ``"raw"`` snapshot but do not see one, check whether you loaded
  an HDF5 bundle; bundles may already contain ``raw`` snapshots.
- If a plugin filter does not appear in the CLI, ensure the package is installed
  and exposes the entry point group ``playNano.filters``.
- For large stacks, avoid asking the pipeline to write the entire record as raw JSON
  (use the HDF5 bundle instead).

See also
^^^^^^^^

- :doc:`cli` - command-line reference
- :doc:`gui` - GUI behaviour and export options
- :doc:`analysis` - analysis pipeline and provenance for analysis steps

