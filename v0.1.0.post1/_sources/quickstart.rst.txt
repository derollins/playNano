Quickstart
==========

This short quickstart gets **playNano** running quickly (recommended: **conda**) using
the CLI and GUI. For full details see the linked pages (:doc:`installation`, :doc:`cli`,
:doc:`gui`, :doc:`processing`, :doc:`analysis`).

1. Create a conda environment (recommended)
-------------------------------------------

Ensure you have Anaconda or Miniconda installed (see :doc:`installation` for links) and
open the terminal (Anaconda PowerShel Pront for Windows).

.. code-block:: bash

   conda create -n playnano_env python=3.11
   conda activate playnano_env

2. Install playNano from PyPi
-----------------------------

Install the latest release of **playNano** from PyPi using pip.

.. code-block:: bash

   pip install playnano

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
own data to view other files. There is a preset processing pipeline can can be
applied by pressing the "F" key or the "Apply Filters" button in the GUI. You can
find out more about using the GUI in :doc:`gui`.

Batch process, analyis and export (no GUI):
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For batch processing and analysis the processing and analyis pipelines are run through seperate commands.
To run these commands on example data, these commands can be run from the project root.

.. code-block:: bash

   playnano process ./tests/resources/sample_0.h5-jpk\
     --processing "remove_plane;mask_mean_offset:factor=1;row_median_align;polynomial_flatten:order=2" \
     --export h5,tif,npz --make-gif --output-folder ./results --output-name sample_processed

This will load demo data, apply a processing pipeline, export the processed data as an HDF5 file (``h5``), a
NumPy zipped archive (``npz``) and a multi-page OME-TIFF (``tif``) to the ``./results`` folder. It will also
generate an animated GIF (from ``--make-gif``) with scale bar and frame timestamp annotations.

.. note::
   ``_filtered`` is automatically appended to the output name when processing is applied.

Run analysis (detection + tracking):

.. code-block:: bash

   playnano analyze ./results/sample_processed_filtered.h5 \
     --analysis-steps "feature_detection:mask_fn=mask_mean_offset,factor=0.5,threshold=5;particle_tracking:max_distance=3"

5. Where to go next
-------------------

- Full installation instructions and platform notes: :doc:`installation`
- CLI reference and flags: :doc:`cli`
- GUI overview and shortcuts: :doc:`gui`
- Processing pipeline details + YAML schema: :doc:`processing`
- Analysis API and CLI usage: :doc:`analysis`
- Step-by-step Jupyter demo: :doc:`notebooks`
