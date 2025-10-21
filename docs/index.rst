playNano Documentation
======================

Welcome to **playNano**, a Python toolkit for loading, processing, analysing,
and exporting high-speed AFM (HS-AFM) time-series data.

*playNano* provides a reproducible, provenance-aware workflow for handling AFM
videos from raw instrument files to flattened, filtered, and analysed outputs.
The toolkit includes a command-line interface (CLI), an interactive GUI for veiwing
data, and a modular analysis system that supports both built-in and custom extensions.

.. image:: images/GUI_window.png
   :alt: playNano GUI
   :width: 420px
   :align: center

Welcome to **playNano** - a Python toolkit for loading, processing, analysing
and exporting high-speed AFM (HS-AFM) time-series data (`.h5-jpk`, `.jpk`,
`.spm`, `.asd`).
This documentation covers installation, command-line usage, the PySide6 GUI,
processing filters, data export, analysis pipelines, and the API reference.

Quick links
-----------

- :doc:`introduction` - overview of playNano's motivation, design, and core workflow
- :doc:`installation` - how to install playNano (pip / conda)
- :doc:`quickstart` - 1-minute example: open a file, apply a filter, export GIF
- :doc:`cli` - full command-line reference and examples
- :doc:`gui` - GUI walkthrough, keyboard shortcuts and export workflow
- :doc:`processing` - filters, masks and pipeline behaviour
   - :doc:`processing-operations-reference` - reference of all built-in processing operations
- :doc:`exporting` - exporting data and GIFs from CLI, GUI, or programmatically
- :doc:`analysis` - running analysis modules and provenance
   - :doc:`custom_analysis_modules` - creating and registering custom analysis modules
- :doc:`whats_new/v0.1.0` - highlights of the latest release
- :doc:`changelog` - release notes and history

Quickstart (example)
--------------------

.. code-block:: bash

   # show a file in the interactive GUI
   playnano play ./test/resources/sample_0.h5-jpk

.. note::
   See :doc:`quickstart` for step-by-step examples.

User Guide Overview
-------------------

The **User Guide** is divided into two parts:

**Getting started**
~~~~~~~~~~~~~~~~~~~
- :doc:`introduction` - overview of playNano's motivation, design, and core workflow.
- :doc:`quickstart` - a standalone, five-minute overview showing how to load,
  process, and export AFM data.
- :doc:`notebooks` - interactive Jupyter notebooks demonstrating typical
  workflows and parameter exploration.

**Practical Guides**
~~~~~~~~~~~~~~~~~~~~
- :doc:`installation` - detailed installation and environment setup.
- :doc:`cli` - running batch processing, exports, and automation from the command line.
- :doc:`gui` - exploring AFM stacks interactively and exporting results.
- :doc:`processing` - applying filters, masks, and flattening operations with
  provenance tracking.
- :doc:`analysis` - running feature detection and tracking pipelines.
- :doc:`custom_analysis_modules` - extending the analysis system with your
  own modules or research methods.
- :doc:`exporting` - saving processed data and analysis results in open formats.

Each guide expands on concepts introduced in the Quickstart, combining practical
examples with deeper technical reference.

Contents
--------

User Guide
~~~~~~~~~~

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   introduction
   quickstart
   notebooks
   installation
   processing
   processing-operations-reference
   exporting
   analysis
   custom_analysis_modules
   cli
   gui
   changelog

API Reference
~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/modules

What's New
~~~~~~~~~~

.. toctree::
   :maxdepth: 1
   :caption: What's New

   whats_new/index

Information and Support
-----------------------

- :doc:`changelog`
- GitHub: https://github.com/derollins/playNano
- Issues: https://github.com/derollins/playNano/issues
- Email: d.e.rollins@leeds.ac.uk
