Quickstart
==========

This short quickstart gets *playNano* running quickly (recommended: **conda**).
For full details see the linked pages (installation, CLI, GUI, processing, analysis).

1. Create a conda environment (recommended)
-------------------------------------------

.. code-block:: bash

   # from the project root (where pyproject.toml / src/ live)
   conda create -n playnano python=3.12 -c conda-forge
   conda activate playnano

2. Install playNano (editable)
------------------------------

.. code-block:: bash

   pip install -e .

Optional extras (docs, notebooks):

.. code-block:: bash

   pip install -e ".[docs]" ".[notebooks]"

3. Quick verification
---------------------

.. code-block:: bash

   playnano --help
   python -c "import playNano; print(playNano.__version__)"

4. Most common actions (one-liners)
-----------------------------------

Launch interactive GUI:

.. code-block:: bash

   playnano play path/to/sample.h5-jpk

Batch process + export (no GUI):

.. code-block:: bash

   playnano process path/to/sample.h5-jpk \
     --processing "remove_plane;mask_mean_offset:factor=1;row_median_align" \
     --export tif,npz --make-gif --output-folder ./results --output-name sample_processed

Run analysis (detection + tracking):

.. code-block:: bash

   playnano analyze data/processed_sample.h5 \
     --analysis-steps "detect_particles:threshold=5;track_particles:max_distance=3"

5. Where to go next
--------------------

- Full installation instructions and platform notes: :doc:`installation`
- CLI reference and flags: :doc:`cli`
- GUI overview and shortcuts: :doc:`gui`
- Processing pipeline details + YAML schema: :doc:`processing`
- Analysis API and CLI usage: :doc:`analysis`
- Step-by-step Jupyter demo: :doc:`notebooks`
