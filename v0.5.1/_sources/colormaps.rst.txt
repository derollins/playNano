.. _colormaps:

Colormaps
=========

**playNano** ships with three native colourmaps designed specifically for AFM
data. All three are registered globally on import as ``matplotlib`` colormaps,
making them available by name across the entire toolkit and in your own scripts
alongside built-in options such as ``afmhot`` and ``viridis``.

.. code-block:: python

   import playnano  # registers all three colourmaps on import

   import matplotlib.pyplot as plt
   plt.imshow(height_data, cmap="afm_brown")
   plt.imshow(height_data, cmap="playnano_gold")
   plt.imshow(height_data, cmap="classic_afm")

.. _colormap-overview:

Overview
--------

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Name
     - Character
     - L* range
     - Linearity (R²)
     - Best for
   * - ``afm_brown``
     - Near-black → brown → warm cream
     - 0.6 → 96.4
     - 0.996
     - General use, HS-AFM video
   * - ``playnano_gold``
     - Black → deep red → amber → white
     - 0.0 → 100.0
     - 0.978
     - High-contrast, feature-rich samples
   * - ``classic_afm``
     - Black → dark red → brown → cream
     - 0.0 → 99.8
     - 0.763
     - Continuity with legacy workflows

.. _colormap-preview:

.. figure:: images/native_colormaps.png
   :alt: playNano native colormaps
   :align: center
   :width: 500px

   The three native playNano colourmaps rendered across the full data range.

---

.. _colormap-afm-brown:

afm_brown
---------

``afm_brown`` is the default colourmap for all playNano visual exports and the GUI.
It was developed for this project to address the perceptual limitations of traditional
AFM colourmaps — in particular the dark plateau near zero that compresses
contrast in low-lying substrate regions, and the non-linear lightness
discontinuities that cause flickering in HS-AFM videos when the z-scale shifts
between frames.

The ramp runs from near-black through a warm brown mid-tone to a pale cream,
retaining the familiar orange-brown character of conventional AFM colour
schemes. Lightness (CIE L*) increases monotonically across the full range with
near-perfect linearity (R² = 0.996), meaning equal steps in height produce
equal steps in perceived brightness throughout the scale.

.. list-table::
   :widths: 35 65

   * - **L\* range**
     - 0.6 → 96.4
   * - **Linearity (R²)**
     - 0.996
   * - **Monotone**
     - Yes (zero non-monotone steps)
   * - **Mean ΔE\* per step**
     - 0.647
   * - **ΔE\* CoV**
     - 0.20
   * - **Perceptual character**
     - Perceptually linear

**When to use**

``afm_brown`` is the recommended default for most AFM data. Its perceptually
linear lightness ensures fine substrate detail is resolved at the dark end of
the scale, and its visual stability makes it well-suited to HS-AFM video
exports where z-scale consistency between frames is important. It is also similar
enough to traditional AFM colour schemes to be visually familiar and easily accepted
by users used to older AFM colour scales.

---

.. _colormap-playnano-gold:

playnano_gold
-------------

``playnano_gold`` is a high-contrast colourmap spanning the full luminance
range (L* 0-100), from pure black through deep red and gold to white. The
wider hue sweep and full dynamic range make it effective for samples with
complex topography and a broad distribution of feature heights, where maximum
separation between levels is desirable.

Lightness increases monotonically throughout, though the larger hue excursion
means perceived colour differences per step are less uniform than in
``afm_brown`` (ΔE\* CoV = 0.57). For static images and figures where contrast
takes priority over strict perceptual linearity, this is the stronger choice.

.. list-table::
   :widths: 35 65

   * - **L\* range**
     - 0.0 → 100.0
   * - **Linearity (R²)**
     - 0.978
   * - **Monotone**
     - Yes (zero non-monotone steps)
   * - **Mean ΔE\* per step**
     - 0.789
   * - **ΔE\* CoV**
     - 0.57
   * - **Perceptual character**
     - Monotone-lightness, high-contrast

**When to use**

``playnano_gold`` suits complex, feature-rich samples where the full height
range needs to be visually separated.

---

.. _colormap-classic-afm:

classic_afm
-----------

``classic_afm`` replicates the non-linear brown colourmap commonly encountered
in AFM software. It is included for continuity, allowing direct visual
comparison with figures and data produced in other packages, and is not
recommended for new work.

The lightness profile is non-linear (R² = 0.763), with a pronounced dark
plateau in the lower quarter of the range (L* reaches only 5.5 at the 25th
percentile) that suppresses contrast in low-height regions. This is the
behaviour that ``afm_brown`` was specifically designed to overcome.

.. list-table::
   :widths: 35 65

   * - **L\* range**
     - 0.0 → 99.8
   * - **Linearity (R²)**
     - 0.763
   * - **Monotone**
     - Yes (zero non-monotone steps)
   * - **Mean ΔE\* per step**
     - 0.738
   * - **ΔE\* CoV**
     - 0.50
   * - **Perceptual character**
     - Non-linear (legacy)

**When to use**

Use ``classic_afm`` only when reproducing the look of figures generated with
other AFM software, or when comparing directly with existing published data
that used a similar colourmap.

---

.. _colormap-usage:

Usage
-----

Programmatic
^^^^^^^^^^^^

All three colourmaps are available as ``matplotlib``
:class:`~matplotlib.colors.LinearSegmentedColormap` objects via
:mod:`playnano.utils.colormaps`.

.. code-block:: python

    import playnano  # register_custom_colormaps() is called on import

    # All three are now available by name for matplotlib functions like imshow():
    plt.imshow(height_data, cmap="afm_brown")
    plt.imshow(height_data, cmap="playnano_gold")
    plt.imshow(height_data, cmap="classic_afm")

    # Reversed versions are also registered automatically
    plt.imshow(height_data, cmap="afm_brown_r")

    # Additionally, they can be used within the playNano export functions
    from playnano.io.gif_export import export_gif

    export_gif(
       stack,
       make_gif=True,
       output_folder="exports",
       output_name="sample",
       scale_bar_nm=100,
       cmap_name="playnano_gold",  # or "afm_brown", "classic_afm"
       zmin="auto",
       zmax="auto",
       draw_ts=True,
      draw_scale_bar=False,
   )

CLI and GUI
^^^^^^^^^^^

The default colourmap for all animated exports (:ref:`export-gif`,
:ref:`export-video`, :ref:`export-image-sequence`) is ``afm_brown``. An
alternative colourmap can be set via the ``--cmap`` flag:

.. code-block:: bash

   playnano process sample_data --make-gif --cmap playnano_gold

In the GUI, the colourmap and z-scale can be adjusted interactively from the
display controls before export.

---

.. _colormap-background:

Background: perceptual linearity in AFM visualisation
------------------------------------------------------

Conventional AFM colourmaps such as Matplotlib's ``afmhot`` and the traditional
colormap reprduced in ``classic_afm``, were not designed with perceptual uniformity
in mind. Two common consequences are:

- **The black plateau** — a long stretch of near-zero lightness at the low end
  of the scale that compresses contrast and obscures fine substrate detail.
- **Flicker in HS-AFM video** — non-linear lightness kinks interact with
  small frame-to-frame changes in the data range, causing apparent brightness
  jumps between frames even when the underlying height values are stable.

Perceptual linearity (monotonically increasing L* at a near-constant rate)
addresses both issues. It does not require full perceptual uniformity in the
strict sense used to describe colourmaps such as ``viridis`` — where every
step produces an identical ΔE\* — but it eliminates the artefacts above while
remaining visually familiar to AFM users.

The table below compares the **playNano** native colourmaps with ``afmhot`` on the
key perceptual metrics:

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 20 15

   * - Colormap
     - L\* range
     - Linearity (R²)
     - Mean ΔE\*/step
     - ΔE\* CoV
   * - ``afm_brown``
     - 0.6 → 96.4
     - 0.996
     - 0.647
     - 0.20
   * - ``playnano_gold``
     - 0.0 → 100.0
     - 0.978
     - 0.789
     - 0.57
   * - ``classic_afm``
     - 0.0 → 99.8
     - 0.763
     - 0.738
     - 0.50
   * - ``afmhot`` (reference)
     - 0.0 → 100.0
     - 0.872
     - 1.004
     - 0.16