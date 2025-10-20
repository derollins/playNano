"""
Module for linking particle features across frames to build trajectories.

This module defines the ParticleTrackingModule, which links features
detected in sequential frames of an AFM image stack using nearest-neighbor
matching based on feature centroids.

Features are matched across frames if they lie within a specified maximum
distance. Tracks are formed by chaining these matches over time.

Each resulting track includes:
    - A unique track ID
    - A list of frames where the particle appears
    - A list of point indices referencing the original features
    - A list of centroids describing the particle's positions

Optionally, per-track masks are extracted from the labeled feature masks.
"""

from typing import Any, Optional, Sequence

import numpy as np

from playNano.afm_stack import AFMImageStack
from playNano.analysis.base import AnalysisModule


class ParticleTrackingModule(AnalysisModule):
    """
    Link detected features frame-to-frame to produce particle trajectories.

    This module links features detected by a prior featuredetection module
    using nearest-neighbor centroid matching across adjacent frames. A new
    track is created for each unmatched feature.

    Version
    -------
    0.1.0
    """

    version = "0.1.0"

    @property
    def name(self) -> str:
        """
        Name of the analysis module.

        Returns
        -------
        str
            Unique identifier: "particle_tracking".
        """
        return "particle_tracking"

    requires = ["feature_detection", "log_blob_detection"]

    def run(
        self,
        stack: AFMImageStack,
        previous_results: Optional[dict[str, Any]] = None,
        detection_module: str = "feature_detection",
        coord_key: str = "features_per_frame",
        coord_columns: Sequence[str] = ("centroid_x", "centroid_y"),
        max_distance: float = 5.0,
        **params,
    ) -> dict[str, Any]:
        """
        Track particles across frames using nearest-neighbor association.

        Parameters
        ----------
        stack : AFMImageStack
            The input AFM image stack.

        previous_results : dict[str, Any], optional
            Must contain results from a detection module, including:

            - coord_key (e.g., "features_per_frame"): list of dicts with per-frame
              features
            - "labeled_masks": per-frame mask of label regions

        detection_module : str, optional
            Which module to read features from (default: "feature_detection").

        coord_key : str, optional
            Key in previous_results[detection_module] containing per-frame feature dicts
            (default: "features_per_frame").

        coord_columns : Sequence[str], optional
            Keys to extract coordinates from each feature; falls back to "centroid" if
            needed. Default is ("centroid_x", "centroid_y")).

        max_distance : float, optional
            Maximum allowed movement per frame in coordinate units (default: 5.0).

        Returns
        -------
        dict
            Dictionary with keys:

            - tracks : list of dict
                Per-track dictionaries containing:

                - id : int
                  Track ID

                - frames : list of int
                  Frame indices

                - point_indices : list of int
                  Indices into features_per_frame

                - centroids : list of tuple[float, float]
                  (x, y) positions of the tracked points

            - track_masks : dict[int, np.ndarray]
              Last mask per track

            - n_tracks : int
              Total number of tracks
        """

        if previous_results is None:
            raise RuntimeError(f"{self.name!r} requires previous results to run.")

        # Auto-detect the most recent available detection module

        if previous_results is None:
            raise RuntimeError(f"{self.name!r} requires previous results to run.")

        available = [mod for mod in reversed(self.requires) if mod in previous_results]
        if not available:
            raise RuntimeError(
                f"{self.name!r} requires one of {self.requires}, but none were found in previous results."  # noqa
            )

        detection_module = available[0]

        fd_out = previous_results[detection_module]
        feats = fd_out[coord_key]  # List[List[dict]]
        masks = fd_out["labeled_masks"]  # List[np.ndarray]

        n_frames = len(feats)
        tracks = []
        next_track_id = 0

        # List of tuples (track_id, last_coord)
        active_tracks = []

        def extract_coords(f: dict) -> tuple[float, float]:
            try:
                return tuple(float(f[k]) for k in coord_columns)
            except KeyError:
                c = f.get("centroid")
                if not c or len(c) < 2:
                    raise KeyError(
                        f"Missing coordinate keys {coord_columns} and fallback 'centroid'"  # noqa
                    ) from None
                return tuple(c[:2])

        for t in range(n_frames):
            this_feats = feats[t]
            assigned = set()
            new_active = []

            # Match existing tracks to nearest features
            for trk_id, last_coord in active_tracks:
                best = None
                best_dist = max_distance
                best_idx = None

                for i, f in enumerate(this_feats):
                    if i in assigned:
                        continue
                    coords = extract_coords(f)
                    dist = np.hypot(
                        coords[0] - last_coord[0], coords[1] - last_coord[1]
                    )
                    if dist < best_dist:
                        best_dist, best, best_idx = dist, coords, i

                if best is not None:
                    track = tracks[trk_id]
                    track["frames"].append(t)
                    track["coords"].append(best)
                    track["point_indices"].append(best_idx)
                    assigned.add(best_idx)
                    new_active.append((trk_id, best))

            # Start new tracks for unmatched detections
            for i, f in enumerate(this_feats):
                if i in assigned:
                    continue
                coords = extract_coords(f)
                trk = {
                    "id": next_track_id,
                    "frames": [t],
                    "coords": [coords],
                    "point_indices": [i],
                }
                tracks.append(trk)
                new_active.append((next_track_id, coords))
                next_track_id += 1

            active_tracks = new_active

        # Generate per-track masks from last known frame/feature
        track_masks = {}
        for trk in tracks:
            t_last = trk["frames"][-1]
            i_last = trk["point_indices"][-1]
            label = feats[t_last][i_last].get("label")
            if label is not None:
                track_masks[trk["id"]] = masks[t_last] == label

        return {
            "tracks": tracks,
            "track_masks": track_masks,
            "n_tracks": len(tracks),
        }
