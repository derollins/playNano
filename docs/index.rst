playNano Documentation
======================

Welcome to **playNano** - a Python toolkit for loading, processing, analysing
and exporting high-speed AFM (HS-AFM) time-series data (`.h5-jpk`, `.jpk`,
`.spm`, `.asd`).
This documentation covers installation, command-line usage, the PySide6 GUI,
processing filters, analysis pipelines, and the API reference.

.. image:: images/GUI_window.png
   :alt: playNano GUI
   :width: 420px
   :align: center

Quick links
-----------

- :doc:`installation` - how to install playNano (pip / conda)
- :doc:`quickstart` - 1-minute example: open a file, apply a filter, export GIF
- :doc:`cli` - full command-line reference and examples
- :doc:`gui` - GUI walkthrough, keyboard shortcuts and export workflow
- :doc:`processing` - filters, masks and pipeline behaviour
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
   See :doc:`quickstart` for step-by-step instructions.

Contents
--------

User Guide
~~~~~~~~~~

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   cli
   gui
   processing
   analysis
   custom_analysis_modules
   notebooks
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
