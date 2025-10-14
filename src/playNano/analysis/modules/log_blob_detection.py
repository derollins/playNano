"""
Module for LoG blob detection.

Detect "blobs" in each frame of an AFM image stack using the Laplacian-of-Gaussian
method. Provides automatic multi-scale blob detection and optional radius estimation.
"""

from typing import Any, Optional

from playNano.analysis.base import AnalysisModule


class LoGBlobDetectionModule(AnalysisModule):
    """
    Detect blobs in AFM image stacks using the Laplacian-of-Gaussian (LoG) method.

    This module applies multi-scale blob detection to each frame in an AFM image stack
    using the Laplacian-of-Gaussian algorithm from `skimage.feature.blob_log`. It
    supports automatic scale selection and optional estimation of blob radii.

    Methods
    -------
    run(stack, previous_results=None, *, min_sigma=1.0, max_sigma=5.0, num_sigma=10,
        threshold=0.1, overlap=0.5, include_radius=True)
        Detects blobs in each frame of the AFM image stack and returns per-frame
        features and a summary.

    Version
    -------
    0.1.0

    Examples
    --------
    >>> module = LoGBlobDetectionModule()
    >>> result = module.run(stack, min_sigma=1.0, max_sigma=5.0, num_sigma=10)
    >>> result['summary']['total_blobs']
    42
    """

    version = "0.1.0"

    @property
    def name(self) -> str:
        """
        Name of the analysis module.

        Returns
        -------
        str
            The string identifier for this module: "log_blob_detection".
        """
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
        Detect "blobs" in each frame via a Laplacian-of-Gaussian filter.

        Parameters
        ----------
        stack : AFMImageStack
            Must have stack.data of shape (n_frames, H, W).

        min_sigma, max_sigma : float
            Parameters passed to skimage.feature.blob_log.

        num_sigma : int
            Parameter passed to skimage.feature.blob_log.

        threshold : float
            Absolute intensity threshold for LoG response.

        overlap : float
            If two detected blobs overlap more than this fraction,
            only the larger is kept.

        include_radius : bool
            If True, append the estimated blob radius in each feature-dict.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys:

            - features_per_frame : list of list of dict
                Per-frame list of detected blobs. Each dict contains:
                - frame_timestamp : float
                - y, x : float
                - sigma : float
                - radius : float, optional (if include_radius=True)
            - summary : dict
                Aggregate metrics:
                - total_frames : int
                - total_blobs : int
                - avg_blobs_per_frame : float
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
