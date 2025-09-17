Custom Analysis Modules
=======================

You can extend **playNano** by writing your own analysis modules and registering
them as plugins. This allows you to integrate custom feature detectors, tracking
methods, or specialised statistics into the standard pipeline.

Stucture and Requirements
-------------------------

Custom modules must subclass :class:`playNano.analysis.base.AnalysisModule` and
implement two things:

- a ``name`` property returning a unique string identifier
- a ``run(stack, previous_results=None, **params) -> dict`` method

You may also define:

- a ``version`` class attribute (string) to help with provenance tracking
- a ``requires`` list to declare required previous analysis modules

Minimal Example
^^^^^^^^^^^^^^^

.. code-block:: python

   from playNano.analysis.base import AnalysisModule
   from playNano.afm_stack import AFMImageStack

   class MyModule(AnalysisModule):
       version = "0.1.0"

       @property
       def name(self) -> str:
           return "my_module"

       def run(self, stack: AFMImageStack, previous_results=None, **params):
           # Example: count total number of pixels in the dataset
           return {"count": stack.data.size}

Outputs must be returned as a dictionary mapping **string keys** to
results (arrays, dicts, numbers, etc.). These values are stored under
``stack.analysis`` and linked in the provenance log.

Declaring Dependencies
^^^^^^^^^^^^^^^^^^^^^^

If your module requires the outputs of another analysis step, you can
declare this using the ``requires`` class attribute:

.. code-block:: python

   class MyDependentModule(AnalysisModule):
       requires = ["detect_particles"]

       @property
       def name(self):
           return "my_dependent"

       def run(self, stack, previous_results=None, **params):
           particles = previous_results["detect_particles"]["coords"]
           return {"n_particles": len(particles)}

This ensures the pipeline provides access to upstream results.

This isn't fully fleshed out so if you encounter any isssue please raise and issue on GitHub.

Registering the Module
----------------------

Add an entry under ``[project.entry-points."playNano.analysis"]`` in
``pyproject.toml`` so the plugin system can discover your module:

.. code-block:: toml

   [project.entry-points."playNano.analysis"]
   my_module = "mypackage.mymodule:MyModule"

After installation (``pip install .``), playNano will automatically detect
your plugin.

Best Practices
--------------

- **Keep outputs JSON-friendly** if you expect to log or export to JSON.
  For large arrays, consider summarising or providing statistics.
- **Add a version string** (``version = "0.1.0"``) to make provenance records
  more reproducible.
- **Document parameters and outputs** with clear docstrings - they appear in
  generated module documentation.
- **Test modules in isolation** before adding them to pipelines.
- **Use consistent naming**: short, lowercase, underscores for ``name``.

Usage in a Pipeline
-------------------

Once installed, your module can be invoked just like a built-in one:

.. code-block:: bash

   playnano analyze data/sample.h5 \
       --analysis-steps "my_module:param1=42"

or programmatically:

.. code-block:: python

   from playNano.analysis.pipeline import AnalysisPipeline

   pipeline = AnalysisPipeline()
   pipeline.add("my_module", param1=42)
   pipeline.run(stack)

Debugging & Troubleshooting
---------------------------

- Use logging (``import logging``) within your module for debug output.
- Check ``stack.provenance["analysis"]`` after running a pipeline to confirm
  your module's results were recorded.
