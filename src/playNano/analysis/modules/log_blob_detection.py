"""
Module for LoG blob detection.

Module: LoGBlobDetectionModule
Detect “blobs” in each frame of an AFM image stack using the Laplacian-of-Gaussian
method.

Provides automatic multi-scale blob detection and optional radius estimation.
"""

from typing import Any, Optional

from playNano.analysis.base import AnalysisModule


class LoGBlobDetectionModule(AnalysisModule):
    @property
    def name(self) -> str:
        """
        Name of the analysis module.

        Returns
        -------
        str
            The string identifier for this module: "dbscan_clustering".
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
        Detect “blobs” in each frame via a Laplacian-of-Gaussian filter.

        Parameters
        ----------
        stack : AFMImageStack
            Must have stack.data of shape (n_frames, H, W).
        min_sigma, max_sigma, num_sigma : float|int
            Parameters passed to skimage.feature.blob_log.
        threshold : float
            Absolute intensity threshold for LoG response.
        overlap : float
            If two detected blobs overlap more than this fraction,
            only the larger is kept.
        include_radius : bool
            If True, append the estimated blob radius in each feature-dict.

        Returns
        -------
        {
          "features_per_frame": List[List[dict]],
          "summary": {
             "total_frames": int,
             "total_blobs": int,
             "avg_blobs_per_frame": float
          }
        }

        Each feature-dict contains at least:
          - "frame_timestamp": float
          - "y", "x": blob center coordinates
          - "sigma": scale at which it was detected
          - **optional** "radius": sqrt(2) * sigma  (if include_radius=True)
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
