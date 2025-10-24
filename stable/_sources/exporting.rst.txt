.. _exporting:

Exporting Data
==============

**playNano** supports saving processed AFM video stacks and analysis results
in several interoperable formats. Exports can include raw data, processed
frames, masks, provenance, and optional animated visualizations.

Overview
--------

Data can be exported using the **command-line interface (CLI)** or from Python
using the functions in :mod:`playNano.io.export_data` and
:mod:`playNano.io.gif_export`.

Supported export formats include:

- :ref:`export-ome-tiff` - Multi-frame image stack compatible with ImageJ/Fiji.
- :ref:`export-npz` - NumPy zipped archive with data arrays and metadata.
- :ref:`export-hdf5` - Hierarchical HDF5 bundle including data and provenance.
- :ref:`export-gif` - Annotated animation with timestamps, scale bar, and color map.

.. _format-comparison:

Format Comparison
-----------------

The following table summarises the key features of each export format:

+-------------+---------+----------------+-----------+---------------------------+
| Format      | Raw     | Processed Data | Masks     | Provenance / Metadata     |
+=============+=========+================+===========+===========================+
| OME-TIFF    | Yes     | Yes            | No        | Limited                   |
+-------------+---------+----------------+-----------+---------------------------+
| GIF         | Yes     | Yes            | No        | No                        |
+-------------+---------+----------------+-----------+---------------------------+
| NPZ         | Yes     | Yes            | Yes       | Full                      |
+-------------+---------+----------------+-----------+---------------------------+
| HDF5        | Yes     | Yes            | Yes       | Full, hierarchical        |
+-------------+---------+----------------+-----------+---------------------------+

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

- ``--export`` - Choose one or more formats (``tif``, ``npz``, ``hdf5``).
- ``--make-gif`` - Generate an annotated GIF (requires timing metadata).
- ``--output-folder`` - Destination directory (default: ``output/``).
- ``--output-name`` - Base name for exported files (default: derived from input).

The GIF annotations can be further customised using ``--scale-bar-nm``,
``--zmin``, and ``--zmax`` options; see :ref:`export-gif` for details.

For full syntax, see :doc:`cli`.
Files can also be exported from the GUI window, see :doc:`gui` for details.

**Example**

.. code-block:: bash

   playnano process sample_data --processing "remove_plane" --export tif npz hdf5 --output-folder exports --output-name test_stack

This processes an AFM stack using the ``remove_plane`` filter and exports OME-TIFF, NPZ, and HDF5 bundles to `exports/.

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

.. _export-ome-tiff:

OME-TIFF Export
---------------

**Format:** ``.ome.tif`` (single multi-frame image stack)

:func:`playNano.io.export_data.save_ome_tiff_stack` exports OME-TIFF files.

The OME-TIFF export stores the processed AFM video from the
:attr:`~playNano.afm_stack.AFMImageStack.data` attribute as a single stack
of frames in a format compatible with ImageJ, Fiji, Bio-Formats, and general
image analysis tools.

**Contents**

- Each frame stored as a TIFF plane
- Global metadata including pixel size, timestamps, and channel name
- OME-XML header describing acquisition

**Use cases**

- Visualisation and measurement in ImageJ/Fiji
- Conversion to other microscopy formats
- Sharing of processed image stacks

Programmatic TIFF Export
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pathlib import Path
   from playNano.io.export_data import save_ome_tiff_stack

   save_ome_tiff_stack(Path("exports/stack_filtered.ome.tif"), stack)
   save_ome_tiff_stack(Path("exports/stack_raw.ome.tif"), stack, raw=True)

.. note::

   OME-TIFF exports contain only a *single stack*.
   Derived data such as masks or filtered layers are not included;
   use :ref:`export-hdf5` or :ref:`export-npz` for multi-layer outputs.

---

.. _export-npz:

NPZ Export
----------

**Format:** ``.npz`` (NumPy compressed archive)

:func:`playNano.io.export_data.save_npz_bundle` exports NPZ bundles.

**Contents**

- ``data`` - raw or processed image stack
- ``processed__<step>`` - processed frame arrays
- ``masks__<mask>`` - Boolean mask arrays
- ``frame_metadata_json`` - per-frame metadata including timestamps
- ``provenance_json`` - full processing and analysis history
- ``pixel_size_nm`` - pixel size in nanometres
- ``channel`` - data channel name

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
   from playNano.io.export_data import save_npz_bundle

   save_npz_bundle(Path("exports/stack_filtered.npz"), stack)
   save_npz_bundle(Path("exports/stack_raw.npz"), stack, raw=True)

**Reloading**

.. code-block:: python

   from playNano.io.loaders import load_npz_bundle
   stack = load_npz_bundle("exports/stack_filtered.npz")

---

.. _export-hdf5:

HDF5 Export
-----------

**Format:** ``.h5`` (Hierarchical data container)

:func:`playNano.io.export_data.save_h5_bundle` exports HDF5 bundles.

**Contents**

- ``/data`` - raw or processed image stack
- ``/processed/<step>`` - filtered or analysis layers
- ``/masks/<mask>`` - Boolean masks
- ``/frame_metadata_json`` - per-frame metadata including timestamps
- ``/provenance_json`` - full processing and analysis history

**Attributes**

- ``pixel_size_nm`` - pixel size in nanometres
- ``channel`` - channel name

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
   from playNano.io.export_data import save_h5_bundle

   save_h5_bundle(Path("exports/stack_filtered.h5"), stack)
   save_h5_bundle(Path("exports/stack_raw.h5"), stack, raw=True)

**Reloading**

.. code-block:: python

   from playNano.io.loaders import load_h5_bundle
   stack = load_h5_bundle("exports/stack_filtered.h5")

---

.. _export-gif:

GIF Export
----------

**Format:** ``.gif`` (annotated animation)

:func:`playNano.io.gif_export.export_gif` creates annotated GIFs.

GIF exports provide a compact visualisation of the AFM video or processed
stack, with optional annotations.

**Annotations**

- **Timestamps** - derived from frame metadata
- **Scale bar** - physical calibration (default 100 nm)
- **Colour map** - default ``afmhot`` normalisation
- **Frame rate** - determined from timing metadata

GIF Export Options (CLI)
^^^^^^^^^^^^^^^^^^^^^^^^

When exporting GIFs, the following flags control the annotations and scaling:

.. code-block:: bash

   playnano process input_folder --make-gif --output-name my_video \
       --scale-bar-nm 100 --zmin auto --zmax auto

**Flags**

- ``--make-gif`` - Generate an animated GIF after processing.
- ``--scale-bar-nm`` - Integer length of the scale bar in nanometres (default: 100).
- ``--zmin`` - Minimum value of the z scale; can be a float or ``auto`` (default: auto).
- ``--zmax`` - Maximum value of the z scale; can be a float or ``auto`` (default: auto).

These options let you control the visual appearance of the GIF:

- **Scale bar** shows a physical size reference.
- **Z scale** sets the height (colour) mapping; auto uses min/max of the data.
- GIFs can be combined with the usual ``--export`` options to produce NPZ, HDF5, or TIFF bundles simultaneously.

**Example**

.. code-block:: bash

   playnano process sample_data --processing "remove_plane" \
       --make-gif --output-name test_stack --scale-bar-nm 150 --zmin 0 --zmax 20

Programmatic GIF Export
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from playNano.io.gif_export import export_gif
   export_gif(
       stack,
       make_gif=True,
       output_folder="exports",
       output_name="sample",
       scale_bar_nm=100,
       zmin="auto",
       zmax="auto",
       draw_ts=True,
   )

---

Advanced / Programmatic Usage
-----------------------------

Unified Export
^^^^^^^^^^^^^^

Use :func:`~playNano.io.export_bundles` to export multiple formats in one call.

.. code-block:: python

   from pathlib import Path
   from playNano.io.export_data import export_bundles

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

- NPZ and HDF5 exports contain processed layers, masks, timestamps, and
  full provenance for round-trip reconstruction.
- OME-TIFF is primarily for viewing in ImageJ/Fiji and does not include
  masks or processed layers.
- Use HDF5 for reproducible workflows and archiving.
- Use NPZ for lightweight interoperability within Python.
- GIFs are primarily for communication, figures, and quick inspection.

---

Provenance Tracking
-------------------

All exports except OME-TIFF include a provenance record that lists:

- Each processing step and its parameters
- Input/output file names and timestamps
- Versions of filters, plugins, and the playNano package

Provenance is stored in ``provenance_json`` (NPZ/HDF5) or as an attribute in
``AFMImageStack.provenance``.
