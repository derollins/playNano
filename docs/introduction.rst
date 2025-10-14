Introduction
============

Overview
--------

**playNano** is an open-source Python toolkit for time-aware processing and analysis
of atomic force microscopy (AFM) data.
It is designed to handle complete AFM time series—such as high-speed AFM
(HS-AFM) videos—as unified datasets rather than isolated frames.
The software provides a reproducible, provenance-tracked workflow for preparing,
visualising, and analysing AFM data within a modular Python environment.

Motivation and Design
---------------------

Conventional AFM image-processing tools often treat each frame independently,
making it difficult to perform consistent operations or analyses across time-series
data. **playNano** was developed to address this gap by introducing a
**pipeline-based**, **time-aware**, and **FAIR-compliant** approach built using Python
tools and libraries.

The design philosophy is to keep the architecture transparent and simple enough
for experimental scientists to use and extend, while ensuring every processing
and analysis step is fully recorded for reproducibility.
Each pipeline—whether for image processing or analysis—stores its operations
and parameters directly in the dataset's provenance, allowing complete
reconstruction of results at any stage.

Core Components
---------------

**playNano** is organised around four main stages:

1. **Loading** - Imports a sequence of AFM frames into an
   :class:`~playNano.afm_stack.AFMImageStack`, preserving timestamps and metadata.
   The following AFM file formats are supported:
   ``.jpk``, ``.asd``, ``.h5-jpk``, and ``.spm``.
2. **Processing** - Applies flattening, filtering, and frame-alignment operations
   to prepare data for analysis and export. Pipelines are defined as ordered
   lists of steps that run serially on the stack.
3. **Analysis** - Executes configurable pipelines of analysis modules, such as
   particle detection, clustering, and tracking. Modules can be combined to
   create custom workflows.
4. **Export** - Saves processed data and analysis results in multiple open
   formats for reuse, sharing, and publication.

----

Next Steps
----------

→ Continue to :doc:`quickstart` to load your first AFM file and run a basic
processing workflow.