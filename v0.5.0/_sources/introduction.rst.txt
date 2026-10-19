Introduction
============

Overview
--------

**playNano** is an open-source Python toolkit for time-aware :doc:`processing <processing>`
and :doc:`analysis <analysis>` of atomic force microscopy (AFM) data.
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
   :class:`~playnano.afm_stack.AFMImageStack`, capturing per-frame
   timestamps, durations, absolute start times, scan direction and pixel
   size where the source format records them, and whole-video acquisition
   settings (bidirectional scanning, etc.) alongside.
   The following AFM file formats are supported:
   ``.jpk``, ``.asd``, ``.aris``, ``.h5-jpk``, and ``.spm``.
   Additionally files exported from playNano can be re-loaded back into the programme.
2. **Processing** - Applies flattening, filtering, and frame-alignment operations
   to prepare data for analysis and export. Pipelines are defined as ordered
   lists of steps that run serially on the stack. See more: :doc:`processing`.
3. **Analysis** - Executes configurable pipelines of analysis modules, such as
   particle detection, clustering, and tracking. Modules can be combined to
   create custom workflows. See more: :doc:`analysis`.
4. **Export** - Saves processed data and analysis results in multiple open
   formats for reuse, sharing, and publication. See more: :doc:`exporting`.

----

Loading and Metadata
--------------------

All loaders emit a common per-frame metadata schema, regardless of
source format. Every frame carries a ``timestamp`` (seconds from frame
zero), a ``frame_pixel_size_nm`` (physical pixel size for that frame),
and (where the source format encodes them) a ``frame_duration_s``,
an absolute ``start_epoch_ms``, a ``scan_direction`` (``'topDown'`` /
``'bottomUp'``), and a ``line_rate``. Fields a format cannot provide
are stored as ``None`` rather than filled with a synthetic value.

Whole-video acquisition settings that do not change between frames
(such as whether the acquisition alternates slow-axis direction) are
recorded on the stack under ``stack.acquisition``, distinct from the
per-frame dicts.

This unified layout means downstream code in processing, analysis,
export modules, can read from ``stack.frame_metadata[i]["timestamp"]``,
``stack.acquisition["bidirectional"]``, and so on without branching on
format.

----

Next Steps
----------

→ Continue to :doc:`quickstart` to load your first AFM file and run a basic
processing workflow.