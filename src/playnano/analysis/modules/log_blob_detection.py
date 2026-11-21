"""Analysis module that detects round features with a Laplacian of Gaussian filter."""

from typing import Any, Optional

from playnano.analysis.base import AnalysisModule


class LoGBlobDetectionModule(AnalysisModule):
    """
    Detect round features in each frame with a Laplacian of Gaussian (LoG) filter.

    Pipeline name: ``"log_blob_detection"``. Each frame is searched at several
    scales with :func:`skimage.feature.blob_log`, which gives each feature's
    position and size. Unlike
    :class:`~playnano.analysis.modules.feature_detection.FeatureDetectionModule`,
    no mask is needed.

    .. note::
       The output has no ``centroid`` or ``labeled_masks``, so it cannot yet be
       used by ``particle_tracking`` or the clustering modules, although they list
       ``log_blob_detection`` as an input.

    Examples
    --------
    A Gaussian bump centred at row 32, column 20:

    >>> import numpy as np
    >>> from playnano.afm_stack import AFMImageStack
    >>> from playnano.analysis.modules.log_blob_detection import (
    ...     LoGBlobDetectionModule,
    ... )
    >>> yy, xx = np.mgrid[:64, :64]
    >>> bump = 3.0 * np.exp(-((yy - 32) ** 2 + (xx - 20) ** 2) / (2 * 3.0**2))
    >>> stack = AFMImageStack(bump[None], 2.0, "height_trace", "demo.h5-jpk")
    >>> result = LoGBlobDetectionModule().run(stack, max_sigma=6)
    >>> blob = result["features_per_frame"][0][0]
    >>> blob["y"], blob["x"]
    (32.0, 20.0)
    """

    version = "0.1.0"

    @property
    def name(self) -> str:
        """Name of this module in a pipeline, ``"log_blob_detection"``."""
        return "log_blob_detection"

    def run(
        self,
        stack,
        previous_results: Optional[dict[str, Any]] = None,
        *,
        min_sigma: float = 1.0,
        max_sigma: float = 5.0,
        num_sigma: int = 10,
        threshold: float = 0.1,
        overlap: float = 0.5,
        include_radius: bool = True,
    ) -> dict[str, Any]:
        """
        Detect blobs in each frame of ``stack.data``.

        Parameters
        ----------
        stack : AFMImageStack
            The stack to analyse.
        previous_results : dict, optional
            Not used. Default is None.
        min_sigma : float, optional
            Smallest Gaussian width searched, in pixels. Default is 1.0.
        max_sigma : float, optional
            Largest Gaussian width searched, in pixels. Default is 5.0.
        num_sigma : int, optional
            Number of widths searched between ``min_sigma`` and ``max_sigma``.
            Default is 10.
        threshold : float, optional
            Smallest LoG response kept, in the units of the data (usually nm).
            Lower values detect fainter blobs. Default is 0.1.
        overlap : float, optional
            If two blobs overlap by more than this fraction of their area, the
            smaller is removed. Between 0 and 1. Default is 0.5.
        include_radius : bool, optional
            Add each blob's radius (``sigma * sqrt(2)``, in pixels). Default is
            True.

        Returns
        -------
        dict
            ``"features_per_frame"``
                For each frame, a list of dicts, one per blob, with
                ``frame_timestamp`` (s), ``y`` and ``x`` (pixels), ``sigma``
                (pixels) and, if ``include_radius``, ``radius`` (pixels).
            ``"summary"``
                ``total_frames``, ``total_blobs`` and ``avg_blobs_per_frame``.

        See Also
        --------
        skimage.feature.blob_log : The detection function used.
        """
        from skimage.feature import blob_log

        n_frames, H, W = stack.data.shape
        features_per_frame: list[list[dict]] = []
        total = 0

        for i in range(n_frames):
            frame = stack.data[i]
            ts = stack.time_for_frame(i)
            blobs = blob_log(
                frame,
                min_sigma=min_sigma,
                max_sigma=max_sigma,
                num_sigma=num_sigma,
                threshold=threshold,
                overlap=overlap,
            )
            # blob_log returns Nx3 array: (y, x, sigma)
            feats = []
            for y, x, sigma in blobs:
                d: dict[str, Any] = {
                    "frame_timestamp": ts,
                    "y": float(y),
                    "x": float(x),
                    "sigma": float(sigma),
                }
                if include_radius:
                    d["radius"] = float(sigma * (2**0.5))
                feats.append(d)
            features_per_frame.append(feats)
            total += len(feats)

        summary = {
            "total_frames": n_frames,
            "total_blobs": total,
            "avg_blobs_per_frame": total / n_frames if n_frames else 0,
        }
        return {
            "features_per_frame": features_per_frame,
            "summary": summary,
        }
