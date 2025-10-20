Quickstart
==========

This short quickstart gets **playNano** running quickly (recommended: **conda**).
For full details see the linked pages (:doc:`installation`, :doc:`cli`,
:doc:`gui`, :doc:`processing`, :doc:`analysis`).

Before you start ensure you have a copy of the **playNano** source code, either clone the repository or downlaod a relase from github.

A simnple way to clone the **playNano** repository:

..clode-block:: bash
   git clone https://github.com/derollins/playNano.git   # Clones the repo to a folder called playNano

1. Create a conda environment (recommended)
-------------------------------------------

.. code-block:: bash

   # from the project root (where pyproject.toml / src/ live)
   conda create -n playnano python=3.12 -c conda-forge
   conda activate playnano

2. Install playNano (editable)
------------------------------

Navigate to the playNano project root (where ``pyproject.toml`` / ``src/`` live) and run:

.. code-block:: bash

   pip install -e .

Optional extras (docs, notebooks):

.. code-block:: bash

   pip install -e ".[docs]" ".[notebooks]"

.. note::

   The ``-e`` flag installs **playNano** in editable mode, allowing you to
   modify the source code and see changes immediately—recommended for development
   and experimentation.

3. Quick verification
---------------------

.. code-block:: bash

   playnano --help
   python -c "import playNano; print(playNano.__version__)"

4. Most common actions (one-liners)
-----------------------------------

Launch interactive GUI:
^^^^^^^^^^^^^^^^^^^^^^^

To open a sample file in the GUI, run:

.. code-block:: bash

   playnano play ./tests/resources/sample_0.h5-jpk  # Opens GUI with loaded file

This opens a sample AFM file when run in the project root. Change the path to your
own data to view other files.

Batch process, analyis and export (no GUI):
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For batch processing and analysis the processing and analyis pipelines are run through seperate commands.
To run these commands on example data, these commands can be run from the project root.

.. code-block:: bash

   playnano process ./tests/resources/sample_0.h5-jpk\
     --processing "remove_plane;mask_mean_offset:factor=1;row_median_align" \
     --export h5,tif,npz --make-gif --output-folder ./results --output-name sample_processed

Run analysis (detection + tracking):

.. code-block:: bash

   playnano analyze ./results/sample_processed.h5 \
     --analysis-steps "feature_detection:threshold=5;track_particles:max_distance=3"

5. Where to go next
-------------------

- Full installation instructions and platform notes: :doc:`installation`
- CLI reference and flags: :doc:`cli`
- GUI overview and shortcuts: :doc:`gui`
- Processing pipeline details + YAML schema: :doc:`processing`
- Exporting data and GIFs: :doc:`exporting`
- Analysis API and CLI usage: :doc:`analysis`
- Step-by-step Jupyter demo: :doc:`notebooks`
