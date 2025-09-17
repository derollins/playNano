Notebooks
=========

Within the `notebooks/` directory, you will find Jupyter notebooks that demonstrate how to use
*playNano* programmatically. These notebooks cover the entire workflow from loading and processing
data to analysis and export.

Overview
--------

**Current notebooks:**

- `playnano_demo_notebook.ipynb`: An overview of loading, processing, analysing, and exporting
  time-series AFM data using the *playNano* library API.
- `processing_demo.ipynb`: A step-by-step guide to applying processing filters and exploring and
  exporting results.

Running Notebooks
-----------------

To run the notebooks Jupyter must be installed. The required packages can be installed when
installing **playNano** with pip.

.. code-block:: bash

    pip install -e .[notebooks]

Once installed Jupyter can be launched from the command line.

.. code-block:: bash

    jupyter notebook

This will open a browser window where you can navigate to the `notebooks/` directory and open the
notebooks.

Run each cell in the notebooks sequentially to see the workflow in action. Intially example data from
the test folder is used however you can change the paths to examine your own data and modify the processing
 and analysis steps to begin to analyse your data.

The full API reference is available in the :doc:`api/modules` section of the documentation.
