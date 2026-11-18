.. _exporting:

Exporting Data
==============

**playNano** supports saving processed AFM video stacks and analysis results
in several interoperable formats. Exports can include raw data, processed
frames, masks, provenance, and optional animated visualisations.

Overview
--------

Data can be exported using the **command-line interface (CLI)** or from Python
using the functions in :mod:`playnano.io.export_data` and
:mod:`playnano.io.gif_export`.

Exports fall into two categories:

- :ref:`export-animated` — annotated visual outputs for presentation and communication
  (:ref:`export-gif`, :ref:`export-video`, :ref:`export-image-sequence`).
- :ref:`export-data` — structured data bundles for analysis, archiving, and round-trip
  reconstruction (:ref:`export-ome-tiff`, :ref:`export-npz`, :ref:`export-hdf5`).

.. _format-comparison:

Format Comparison
-----------------

The following table summarises the key features of each export format:

+------------------+---------+----------------+-----------+---------------------------+-------------+
| Format           | Raw     | Processed Data | Masks     | Provenance / Metadata     | Annotations |
+==================+=========+================+===========+===========================+=============+
| GIF              | Yes     | Yes            | No        | No                        | Yes         |
+------------------+---------+----------------+-----------+---------------------------+-------------+
| MP4/AVI          | Yes     | Yes            | No        | No                        | Yes         |
+------------------+---------+----------------+-----------+---------------------------+-------------+
| PNG (folder)     | Yes     | Yes            | No        | No                        | Yes         |
+------------------+---------+----------------+-----------+---------------------------+-------------+
| OME-TIFF         | Yes     | Yes            | No        | Limited                   | No          |
+------------------+---------+----------------+-----------+---------------------------+-------------+
| NPZ              | Yes     | Yes            | Yes       | Full                      | No          |
+------------------+---------+----------------+-----------+---------------------------+-------------+
| HDF5             | Yes     | Yes            | Yes       | Full, hierarchical        | No          |
+------------------+---------+----------------+-----------+---------------------------+-------------+

.. _exporting-cli:

Using the CLI
-------------

Exports are most commonly produced when running the ``process`` subcommand.
You can specify one or more export formats and control filenames and output
directories with these options:

.. code-block:: bash

   playnano process input_folder --export npz hdf5 --make-gif \
       --output-folder exports --output-name test_stack

CLI Flags
^^^^^^^^^

- ``--export`` — Choose one or more formats (``tif``, ``npz``, ``hdf5``).
- ``--make-gif`` — Generate an annotated GIF (requires timing metadata).
- ``--output-folder`` — Destination directory (default: ``output/``).
- ``--output-name`` — Base name for exported files (default: derived from input).

The visual export annotations can be further customised using ``--scale-bar-nm``,
``--zmin``, and ``--zmax`` options; see :ref:`export-animated` for details.

The frame rate can be set using ``--fps``, the deafult without the tag is 5.0.

For full syntax, see :doc:`cli`.
Files can also be exported from the GUI window; see :doc:`gui` for details.

**Example**

.. code-block:: bash

   playnano process sample_data --processing "remove_plane" \
       --export tif npz hdf5 --output-folder exports --output-name test_stack

This processes an AFM stack using the ``remove_plane`` filter and exports
OME-TIFF, NPZ, and HDF5 bundles to ``exports/``.

**Example output structure**

.. code-block:: text

   exports/
    ├── test_stack_filtered.ome.tif
    ├── test_stack_filtered.npz
    ├── test_stack_filtered.h5
    └── test_stack_filtered.gif

Files include the suffix ``_filtered`` when exported from a processed stack.
Raw exports can also be produced using the ``--raw`` flag.

---

.. _export-animated:

Animated Exports
================

GIF, video, and image sequence exports are designed for presentation,
communication, and visual inspection. All three support frame annotations
(timestamps and scale bars) and share a common approach to colourmap and
z-scale configuration.

.. _export-colourmaps:

Colourmaps and Z-Scale
----------------------

All animated exports use ``afm_brown`` as the default colourmap — a perceptually
linear, monotone-lightness map developed for playNano that avoids the brightness
discontinuities and dark plateaus common in traditional AFM colourmaps. Any
``matplotlib``-registered colourmap can be used instead, including playNano's
native maps ``playnano_gold`` and ``classic_afm``.

The z-scale is automatically set to the 1st and 99th percentiles of the data
by default, but can be overridden with explicit ``zmin`` and ``zmax`` values.
Within the GUI, both the colourmap and z-scale can be adjusted interactively
before export, and the exported file reflects those settings.

See :ref:`colormaps` for full details on the native colourmaps.

---

.. _export-gif:

GIF Export
----------

**Format:** ``.gif`` (annotated animation)

:func:`playnano.io.gif_export.export_gif` creates annotated GIFs.

GIF exports provide a compact visualisation of the AFM video or processed
stack, suitable for quick inspection and presentation.

**Annotations**

- **Timestamps** — derived from frame metadata.
- **Scale bar** — physical calibration (default 100 nm) derived from per-frame pixel size metadata.

GIF Export Options (CLI)
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   playnano process input_folder --make-gif --output-name my_video \
       --scale-bar-nm 50 --zmin auto --zmax auto --cmap playnano_gold

**Flags**

- ``--make-gif`` — Generate an animated GIF after processing.
- ``--scale-bar-nm`` — Integer length of the scale bar in nanometres (default: 100).
- ``--zmin`` — Minimum z-scale value; float or ``auto`` (default: auto).
- ``--zmax`` — Maximum z-scale value; float or ``auto`` (default: auto).
- ``--cmap`` — Name of the colormap to use (default: DEFAULT_CMAP).
- ``--fps`` — Set the frame rate for the animation in seconds (default: 5.0).

GIFs can be combined with the usual ``--export`` options to produce NPZ, HDF5,
or TIFF bundles simultaneously.

**Example**

.. code-block:: bash

   playnano process sample_data --processing "remove_plane" \
       --make-gif --output-name test_stack --scale-bar-nm 150 --zmin 0 --zmax 20

Programmatic GIF Export
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from playnano.io.gif_export import export_gif

   export_gif(
       stack,
       make_gif=True,
       output_folder="exports",
       output_name="sample",
       scale_bar_nm=100,
       fps=6.0,
       cmap_name="classic_afm",  # any matplotlib-registered colormap
       zmin="auto",
       zmax="auto",
       draw_ts=False,
       draw_scale_bar=True,
   )

---

.. _export-video:

Video Export
------------

**Format:** ``.mp4`` or ``.avi`` (annotated video)

:func:`playnano.io.video_export.export_video` creates annotated videos.

Video exports provide a high-quality visualisation of the AFM video or
processed stack, suitable for publication and presentation. Videos can be exported in MP4 or AVI
format, and support the same annotations and colourmap options as GIFs.

**Annotations**

- **Timestamps** — derived from frame metadata.
- **Scale bar** — physical calibration (default 100 nm) derived from per-frame pixel size metadata.

Video Export Options (CLI)
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   playnano process input_folder --make-video --output-name my_video \
       --scale-bar-nm 50 --zmin auto --zmax auto --cmap playnano_gold

**Flags**

- ``--make-video`` — Generate an animated video after processing default is MP4 but can add a string to
  specify the format, i.e. ``--make-video avi``.
- ``--scale-bar-nm`` — Integer length of the scale bar in nanometres (default: 100).
- ``--zmin`` — Minimum z-scale value; float or ``auto`` (default: auto).
- ``--zmax`` — Maximum z-scale value; float or ``auto`` (default: auto).
- ``--cmap`` — Name of the colormap to use (default: DEFAULT_CMAP).
- ``--fps`` — Set the frame rate for the animation in seconds (default: 5.0).

Programmatic Video Export
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from playnano.io.video_export import export_video

   export_video(
       stack,
       make_video=True,
       output_folder="exports",
       output_name="sample",
       fmt="mp4", #or "avi"
       cmap_name="playnano_gold",  # any matplotlib-registered colormap
       scale_bar_nm=100,
       fps=6.0,
       zmin="auto",
       zmax="auto",
       draw_ts=True,
       draw_scale_bar=True,
   )

---

.. _export-image-sequence:

Image Sequence Export
---------------------

**Format:** ``.png`` (individual frames in a folder)

Image sequence exports save each frame of the AFM video or processed stack as
a separate PNG file. This is useful for extracting specific frames for
publications or for importing into external tools.

**Annotations**

- **Timestamps** — derived from frame metadata; drawn onto each frame if enabled in the GUI.
- **Scale bar** — physical calibration (default 100 nm) derived from per-frame pixel size metadata;
  drawn onto each frame if enabled in the GUI.

Image Sequence Export Options (CLI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   playnano process input_folder --make-sequence --output-name my_frames \
       --scale-bar-nm 50 --zmin auto --zmax auto --cmap playnano_gold

**Flags**

- ``--make-sequence`` — Generate an image sequence after processing (PNG).
- ``--scale-bar-nm`` — Integer length of the scale bar in nanometres (default: 100).
- ``--zmin`` — Minimum z-scale value; float or ``auto`` (default: auto).
- ``--zmax`` — Maximum z-scale value; float or ``auto`` (default: auto).
- ``--cmap`` — Name of the colormap to use (default: DEFAULT_CMAP).

Programmatic Image Sequence Export
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from playnano.io.image_sequence_export import export_image_sequence

   export_image_sequence(
       stack,
       make_sequence=True,
       output_folder="exports",
       output_name="sample",
       fmt="png", #or "jpg"
       cmap_name="viridis",  # any matplotlib-registered colormap
       scale_bar_nm=100,
       zmin=0,
       zmax=3,
       draw_ts=True,
       draw_scale_bar=True,
   )

---

.. _export-data:

Data Exports
============

HDF5, NPZ, and OME-TIFF exports are designed for data archiving, round-trip
reconstruction, and interoperability with external analysis tools. These formats
do not include visual annotations but carry structured data, masks, and provenance.

---

.. _export-ome-tiff:

OME-TIFF Export
---------------

**Format:** ``.ome.tif`` (single multi-frame image stack)

:func:`playnano.io.export_data.save_ome_tiff_stack` exports OME-TIFF files.

The OME-TIFF export stores the processed AFM video as a single stack of frames
in a format compatible with ImageJ, Fiji, Bio-Formats, and general image
analysis tools.

.. note::

   Since the OME-TIFF metadata schema only contains a single value for
   ``PhysicalSizeX`` / ``PhysicalSizeY`` for all images in a stack, all frames
   must share the same ``frame_pixel_size_nm`` value in their ``frame_metadata``
   dictionaries.

**Contents**

- Each frame stored as a TIFF plane.
- Global metadata including pixel size, timestamps, and channel name.
- OME-XML header describing acquisition.

**Use cases**

- Visualisation and measurement in ImageJ/Fiji.
- Conversion to other microscopy formats.
- Sharing of processed image stacks.

Programmatic OME-TIFF Export
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pathlib import Path
   from playnano.io.export_data import save_ome_tiff_stack

   save_ome_tiff_stack(Path("exports/stack_filtered.ome.tif"), stack)
   save_ome_tiff_stack(Path("exports/stack_raw.ome.tif"), stack, raw=True)

.. note::

   OME-TIFF exports contain only a single stack. Derived data such as masks or
   filtered layers are not included; use :ref:`export-hdf5` or :ref:`export-npz`
   for multi-layer outputs.

---

.. _export-npz:

NPZ Export
----------

**Format:** ``.npz`` (NumPy compressed archive)

:func:`playnano.io.export_data.save_npz_bundle` exports NPZ bundles.

**Contents**

- ``data`` — raw or processed image stack.
- ``processed__<step>`` — processed frame arrays.
- ``masks__<mask>`` — Boolean mask arrays.
- ``frame_metadata_json`` — per-frame metadata including timestamps and pixel sizes in nanometres.
- ``provenance_json`` — full processing and analysis history.
- ``pixel_size_nm`` — pixel size in nanometres.
- ``channel`` — data channel name.

**Example structure**

.. code-block:: text

   data
   processed__step_1_flatten
   masks__feature_mask
   frame_metadata_json
   provenance_json
   pixel_size_nm
   channel

Programmatic NPZ Export
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pathlib import Path
   from playnano.io.export_data import save_npz_bundle

   save_npz_bundle(Path("exports/stack_filtered.npz"), stack)
   save_npz_bundle(Path("exports/stack_raw.npz"), stack, raw=True)

**Reloading**

.. code-block:: python

   from playnano.io.data_loaders import load_npz_bundle

   stack = load_npz_bundle("exports/stack_filtered.npz")

---

.. _export-hdf5:

HDF5 Export
-----------

**Format:** ``.h5`` (hierarchical data container)

:func:`playnano.io.export_data.save_h5_bundle` exports HDF5 bundles.

**Contents**

- ``/data`` — raw or processed image stack.
- ``/processed/<step>`` — filtered or analysis layers.
- ``/masks/<mask>`` — Boolean masks.
- ``/frame_metadata_json`` — per-frame metadata including timestamps and pixel sizes in nanometres.
- ``/provenance_json`` — full processing and analysis history.

**Attributes**

- ``pixel_size_nm`` — pixel size in nanometres for the first frame (usually consistent across frames).
- ``channel`` — channel name.

**Example structure**

.. code-block:: text

   /data
   /processed/step_1_flatten
   /masks/feature_mask
   /frame_metadata_json
   /provenance_json
   .attrs:
     pixel_size_nm
     channel

Programmatic HDF5 Export
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pathlib import Path
   from playnano.io.export_data import save_h5_bundle

   save_h5_bundle(Path("exports/stack_filtered.h5"), stack)
   save_h5_bundle(Path("exports/stack_raw.h5"), stack, raw=True)

**Reloading**

.. code-block:: python

   from playnano.io.loaders import load_h5_bundle

   stack = load_h5_bundle("exports/stack_filtered.h5")

---

Advanced / Programmatic Usage
=============================

Unified Data Export
-------------------

Use :func:`~playnano.io.export_data.export_bundles` to export multiple formats
in a single call.

.. code-block:: python

   from pathlib import Path
   from playnano.io.export_data import export_bundles

   export_bundles(
       afm_stack=stack,
       output_folder=Path("exports"),
       base_name="my_stack",
       formats=["tif", "npz", "h5"],
       raw=False,
   )

Use ``raw=True`` to export unprocessed snapshots only.

---

Notes
-----

- NPZ and HDF5 exports contain processed layers, masks, timestamps, and full
  provenance for round-trip reconstruction.
- OME-TIFF is primarily for viewing in ImageJ/Fiji and does not include masks
  or processed layers.
- Use HDF5 for reproducible workflows and archiving.
- Use NPZ for lightweight interoperability within Python.
- GIFs and videos are primarily for communication, figures, and quick inspection.

---

Provenance Tracking
-------------------

All data exports except OME-TIFF include a provenance record that lists:

- Each processing step and its parameters.
- Input/output file names and timestamps.
- Versions of filters, plugins, and the playNano package.

Provenance is stored in ``provenance_json`` (NPZ/HDF5) or as an attribute in
``AFMImageStack.provenance``.
