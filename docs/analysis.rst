Analysis
========

The analysis system in **playNano** provides a provenance-aware pipeline
for running a variety of analysis modules including built in modules for
feature detection and particle tracking on AFM image stacks.
Analysis steps produce structured results (counts, tables, tracks, summaries)
that are stored on the AFM stack and recorded with full provenance for audit
and reproducibility.

This page covers:
- quick CLI and Python examples
- YAML schema for pipeline files
- how results are saved and inspected
- advanced implementation details (module loading, exact provenance fields)
- extending the system with custom modules

Quick start
-----------

Run an inline analysis pipeline from the CLI:

.. code-block:: bash

   playnano analyze data/processed_sample.h5 \
      --analysis-steps "feature_detection:mask_fn=mask_threshold,threshold=1;particle_tracking:max_distance=3" \
      --output-folder ./results      --output-name tracked_particles

Or load the pipeline from a YAML/JSON file:

.. code-block:: yaml

   analysis:
     - name: feature_detection
       mask_fn: mask_threshold
       threshold: 1
     - name: particle_tracking
       max_distance: 3

.. code-block:: bash

   playnano analyze data/processed_sample.h5 --analysis-file my_pipeline.yaml

Overview & behaviour
--------------------

- Analysis pipelines are conceptually similar to :doc:`processing`` pipelines but operate
  on derived results rather than on image arrays.
- Analysis **adds** results into ``stack.analysis`` (it does **not** replace
  ``stack.data``).
- Each analysis step is executed by an ``AnalysisModule`` (built-in or plugin).
- The pipeline records per-step provenance (parameters, timestamps, version and
  stored result keys) under ``stack.provenance['analysis']``.
- If ``log_to`` is provided to programmatic runs or the CLI, a **sanitised JSON**
  summary is written (arrays and complex objects are summarised for JSON friendliness).
- Some analysis modules have requirements such as being proceeded by cirtain other
  analysis modules, i.e. tracking modules require the detections of particles prior to
  tracking in order to have particles to track.

Available analysis modules
^^^^^^^^^^^^^^^^^^^^^^^^^^

See the generated list of installed modules:

.. include:: _generated/generated_module_list.rst


CLI usage
---------

The analysis pipeline can be run from the pipeline or programmatically but not currently from the GUI.

This is the general form of the analyze subcommand for CLI use:

.. code-block:: bash

   playnano analyze <input_file> \
      (--analysis-steps 'step1:arg=val;step2:arg=val' | --analysis-file pipeline.yaml) \
      [--channel CHANNEL] \
      [--output-folder OUTPUT_DIR] \
      [--output-name BASE_NAME]

Common options:

- ``--analysis-steps`` - semicolon-delimited inline pipeline string.
- ``--analysis-file`` - YAML or JSON file describing the pipeline (mutually exclusive with inline).
- ``--channel`` - channel to read (default: ``height_trace``).
- ``--output-folder`` / ``--output-name`` - control where exported results are written.

Since some analysis modules have several parameters it is often easier to generate a YAML using the wizard.


YAML schema
^^^^^^^^^^^

Top-level key must be ``analysis``:

.. code-block:: yaml

   analysis:
     - name: feature_detection
       mask_fn: mask_threshold
       threshold: 4.5
     - name: particle_tracking
       max_distance: 2.5
       min_length: 5

Each entry:

- **name** (str, required) - analysis module name.
- **parameters** - module-specific kwargs passed through to the module.

Validation notes
^^^^^^^^^^^^^^^^

- The ``analysis`` key is required.
- Each step must include ``name``.
- Unknown module names raise an error at runtime.
- Parameters are forwarded as keyword arguments and must match the module signature.


Programmatic usage
------------------

Construct and run a pipeline from Python:

.. code-block:: python

   from playNano.afm_stack import AFMImageStack
   from playNano.analysis.pipeline import AnalysisPipeline
   import yaml

   stack = AFMImageStack.load_afm_stack("data/processed_sample.h5", channel="height_trace")

   ap = AnalysisPipeline()
   ap.add("detect_particles", threshold=5)
   ap.add("track_particles", max_distance=3.0)

   # run and optionally write a sanitised JSON log
   record = ap.run(stack, log_to="analysis.json")

   # programmatic access
   print(list(stack.analysis.keys()))
   print(stack.provenance["analysis"]["steps"])

Outputs & exports
-----------------

- Results are saved on the AFM stack instance:

  - ``stack.analysis`` : dict of stored analysis results (keys use a step-based naming scheme).
  - ``stack.provenance["analysis"]`` : detailed provenance about the run.
  - ``stack.provenance["environment"]`` : runtime environment metadata (OS, Python, packages).

- CLI/utility functions may optionally export:
  - A sanitised JSON summary (human- and machine-readable).
  - HDF5 bundle with full data + provenance.

Inspecting results
------------------

Common programmatic patterns:

.. code-block:: python

   # 1) List all stored analysis keys
   print(sorted(stack.analysis.keys()))

   # 2) Walk step provenance
   for step in stack.provenance["analysis"]["steps"]:
       print(step["index"], step["name"], step["analysis_key"], step.get("version"))

   # 3) Access outputs for a named module
   for rec in stack.provenance["analysis"]["results_by_name"].get("detect_particles", []):
       key = rec["analysis_key"]
       outputs = rec["outputs"]
       # use outputs (dict/list/array) directly

Advanced / Implementation details
---------------------------------

Module loading
^^^^^^^^^^^^^^

- Modules are resolved first from the built-in registry
  (``playNano.analysis.BUILTIN_ANALYSIS_MODULES``), then via Python entry points
  in the ``playNano.analysis`` group. The first matching entry point is used.
- Loaded classes are instantiated and must subclass ``AnalysisModule``.
- Instantiated modules are cached on the pipeline instance to avoid repeated
  re-instantiation.

Result storage layout
^^^^^^^^^^^^^^^^^^^^^^

- Analysis result keys follow the pattern::

   step_<idx>_<module_name>

  where ``idx`` is 1-based and spaces are replaced by underscores. Each key in
  ``stack.analysis`` maps to the raw outputs returned by the module (could be a
  dict, array, DataFrame, etc.).

Provenance structure
^^^^^^^^^^^^^^^^^^^^^

After a run ``stack.provenance["analysis"]`` contains:

- ``steps`` - ordered list of per-step records. Each record includes:

  - ``index``: 1-based integer
  - ``name``: module name as invoked
  - ``params``: parameters passed to the module (keyword args)
  - ``timestamp``: ISO-8601 UTC timestamp of execution
  - ``version``: optional version string (if module specifies it)
  - ``analysis_key``: key used to store the outputs in ``stack.analysis``

- ``results_by_name`` - mapping from module name to a **list** of results produced by that module during the run. Each entry in the list is a dict with:

  - ``analysis_key`` - stored key in ``stack.analysis``
  - ``outputs`` - the raw outputs object stored under that key

- ``frame_times`` - result of ``stack.get_frame_times()`` (if present), else ``None``.

Extending with Custom Modules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can create your own analysis modules and register them as plugins.
See :doc:`custom_analysis_modules` for full details, including requirements,
examples, and best practices.

Notes
^^^^^

- The pipeline will create ``stack.provenance`` and ``stack.analysis`` if they do not exist.
- ``stack.provenance["environment"]`` is set if not already present (gathered via the system info util).
- When ``log_to`` is supplied, the pipeline writes a sanitised JSON summary using :func:`playNano.analysis.utils.common.sanitize_analysis_for_logging` (this is intended to produce small, JSON-friendly summaries suitable for logs).

Troubleshooting & tips
----------------------

- If a module name fails to resolve, ensure the module is listed in the built-in registry
  or that a plugin exposing the correct entry point is installed.
- For large or complex outputs prefer HDF5 export - sanitised JSON may truncate or summarise arrays.
- If analyses expect processed frames, run a processing pipeline first (see :doc:`processing`).

See also
^^^^^^^^

- :doc:`processing` - pre-processing & pipeline snapshots
- :doc:`cli` - command-line reference and examples
- :doc:`gui` - interactive playback & exporting
- :doc:`custom_analysis_modules` - writing and registering new analysis steps
