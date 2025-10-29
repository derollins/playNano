Processing Operations Reference
===============================

This page lists all built-in operations available in the `playNano.processing` pipeline
subpackage. Each operation is grouped by type and includes a brief description along with
its configurable parameters and default values.

Filters (2D Frame Operations)
-----------------------------

:mod:`playNano.processing.filters`

These functions operate on individual 2D frames and return a transformed array of floats.

Certain filters (e.g. ``remove_plane``, ``row_median_align``) support masked
computation. When a binary mask is provided (see below for mask generator steps), the
operation is applied to the full image, but its internal parameters are estimated only
from unmasked pixels. This is useful when regions of the image contain artifacts,
noise, or irrelevant features that should not influence the operation, but the
correction itself must be applied globally (i.e. flattening based on background
pixels).

This masking behaviour is handled automatically by the processing pipeline if a mask
operation has been added prior to the filter step. Programmatically there is a separate
module :mod:`playNano.processing.masked_filters` that contains masked variants of filters.

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - Name
     - Description
     - Parameters
   * - :func:`~playNano.processing.filters.remove_plane`
     - Removes a best-fit plane from the frame using least squares. Can be masked.
     - None
   * - :func:`~playNano.processing.filters.polynomial_flatten`
     - Fit and subtract a 2D plane (tilt removal). Can be masked.
     - order (int), default: 2
   * - :func:`~playNano.processing.filters.row_median_align`
     - Subtract row-wise median to remove horizontal banding. Can be masked.
     - None
   * - :func:`~playNano.processing.filters.zero_mean`
     - Subtract global mean (optionally masked). Can be masked.
     - None
   * - :func:`~playNano.processing.filters.gaussian_filter`
     - Apply Gaussian smoothing.
     - sigma (float), default: 1.0

Masks (2D Binary Operations)
----------------------------

:mod:`playNano.processing.mask_generators`

These functions generate boolean masks to exclude regions from filters (if a
:mod:`~playNano.processing.masked_filters` function is available) or be used in
analysis.

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - Name
     - Description
     - Parameters
   * - :func:`~playNano.processing.mask_generators.mask_threshold`
     - Mask values above a threshold.
     - threshold (float), default: 0.0
   * - :func:`~playNano.processing.mask_generators.mask_below_threshold`
     - Mask values below a threshold.
     - threshold (float), default: 0.0
   * - :func:`~playNano.processing.mask_generators.mask_mean_offset`
     - Mask values more than a factor * std above the mean.
     - factor (float), default: 1.0
   * - :func:`~playNano.processing.mask_generators.mask_morphological`
     - Applies binary closing to ``|data| > threshold`` using a square structuring element.
     - threshold (float), default: 3, structure_size (int), default: 3
   * - :func:`~playNano.processing.mask_generators.mask_adaptive`
     - Block-wise adaptive thresholding.
     - block_size (int), default: 15, offset (float), default: 0.0

These masks are combined using logical OR. Use the ``clear`` step to reset masks.

Video Processing (3D Stack Operations)
--------------------------------------

:mod:`playNano.processing.video_processing`

These functions operate on 3D stacks (n_frames, height, width) of AFM frames
for alignment, cropping and padding. Outputs include processed stacks and
metadata dictionaries.

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - Name
     - Description
     - Parameters
   * - :func:`~playNano.processing.video_processing.align_frames`
     - Align frames using cross-correlation to a single reference frame. Jump smoothing limits unrealistic displacements. Optional Gaussian pre-filtering improves correlation.
     - reference_frame (int, default: 0), method (str, default: "fft_cross_correlation"), mode (str, default: "pad"), debug (bool, default: False), max_shift (int, optional), pre_filter_sigma (float, optional), max_jump (int, optional)
   * - :func:`~playNano.processing.video_processing.rolling_frame_align`
     - Align frames using a rolling reference (average of last N aligned frames) with integer-pixel shifts. Optional jump smoothing and Gaussian pre-filtering.
     - window (int, default: 5), mode (str, default: "pad"), debug (bool, default: False), max_shift (int, optional), pre_filter_sigma (float, optional), max_jump (int, optional)
   * - :func:`~playNano.processing.video_processing.intersection_crop`
     - Crop aligned stack to largest common intersection region. Returns cropped stack and metadata.
     - stack (ndarray, 3D)
   * - :func:`~playNano.processing.video_processing.crop_square`
     - Crop aligned stack to the largest centered square. Returns metadata including original, intersection, new shapes and offset.
     - stack (ndarray, 3D)
   * - :func:`~playNano.processing.video_processing.replace_nan`
     - Replace NaN values in 2D/3D stacks using several strategies: zero, mean, median, global_mean, constant.
     - mode (str, default: "zero"), value (float, optional, required if mode="constant")

AFM Stack Editing (Frame Selection)
-----------------------------------

:mod:`playNano.processing.stack_edit`

These functions operate on 3D AFM stacks (n_frames, height, width) to remove
or select frames. Only `drop_frames` performs actual edits; the other functions
generate indices to drop for use with `drop_frames`. This is managed by the
:meth:`~playNano.processing.pipeline.ProcessingPipeline._handle_stack_edit_step`
method.

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - Name
     - Description
     - Parameters
   * - :func:`~playNano.processing.stack_edit.drop_frames`
     - Remove specific frames from a 3D array. Does not modify the input array.
     - data (ndarray, 3D), indices_to_drop (list of int)
   * - :func:`~playNano.processing.stack_edit.drop_frame_range`
     - Generate a list of frame indices to drop within a specified start (inclusive) to end (exclusive) range.
     - data (ndarray, 3D), start (int, inclusive), end (int, exclusive)
   * - :func:`~playNano.processing.stack_edit.select_frames`
     - Generate a list of frame indices to drop, keeping only the selected frames.
     - data (ndarray, 3D), keep_indices (list of int, frames to retain)
