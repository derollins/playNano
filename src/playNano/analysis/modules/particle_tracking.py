"""
Module for threshold based feature detection.

Module: ParticleTrackingModule
Detect "features" in each frame of an AFM image stack with a image mask.
"""

from typing import Any, Optional

import numpy as np

from playNano.afm_stack import AFMImageStack
from playNano.analysis.base import AnalysisModule


class ParticleTrackingModule(AnalysisModule):
    """
    Link detected features frame-to-frame to produce particle trajectories.

    Requires that a prior “feature_detection” (or equivalent) step has
    produced per-frame “centroids”: a list of (x, y) arrays.

    Attributes
    ----------
    requires : list[str]
        Names of analysis modules whose outputs (in previous_results) we depend on.

    Version
    -------
    0.1.0
        Initial implementation.
    """

    version = "0.1.0"

    @property
    def name(self) -> str:
        """
        Name of the analysis module.

        Returns
        -------
        str
            The string identifier for this module: "dbscan_clustering".
        """
        return "particle_tracking"

    # Declare that we need cooridinate output from a previous module,
    # i.e. feature_detection.
    requires = ["feature_detection", "log_blob_detection"]

    def run(
        self,
        stack: AFMImageStack,
        previous_results: Optional[dict[str, Any]] = None,
        max_distance: float = 5.0,
        **params,
    ) -> dict[str, Any]:
        """
        Link detections frame-to-frame by nearest neighbor.

        Parameters
        ----------
        previous_results : dict
            Must contain at least one entry with key "feature_detection"
            mapping to a dict that has "centroids": list of (x,y) per frame.
        max_distance : float
            Maximum allowed jump (in pixels/nm) between frames.

        Returns
        -------
        dict with keys:
          - "tracks": list of arrays, one per particle: [[(frame, x, y), …], …]
          - "labels":  np.ndarray of shape (n_frames, H, W)  # optional
        """
        if previous_results is None or "feature_detection" not in previous_results:
            raise RuntimeError(
                f"{self.name!r} requires output from 'feature_detection' - please add it before tracking."  # noqa
            )

        fd_out = previous_results["feature_detection"]
        feats = fd_out["features_per_frame"]  # List of lists of dicts
        masks = fd_out["labeled_masks"]  # List of 2D label arrays

        n_frames = len(feats)
        tracks = (
            []
        )  # each track: dict {id:int, frames: [t0,t1…], centroids: […], labels: […]}
        next_track_id = 0

        # active_tracks: list of (track_id, last_centroid)
        active_tracks = []

        for t in range(n_frames):
            this_feats = feats[t]  # e.g. [ {centroid:(y,x), label:…}, … ]
            assigned = set()
            new_active = []

            # 2) first, try to match each existing track by nearest‐neighbor
            for trk_id, last_cent in active_tracks:
                best = None
                best_dist = max_distance
                best_idx = None

                for i, f in enumerate(this_feats):
                    if i in assigned:
                        continue
                    dist = np.hypot(
                        f["centroid"][0] - last_cent[0], f["centroid"][1] - last_cent[1]
                    )
                    if dist < best_dist:
                        best_dist, best, best_idx = dist, f, i

                if best is not None:
                    # append to existing track
                    track = tracks[trk_id]
                    track["frames"].append(t)
                    track["centroids"].append(best["centroid"])
                    track["labels"].append(best["label"])
                    assigned.add(best_idx)
                    new_active.append((trk_id, best["centroid"]))
                # else: track ends here

            # 3) any unassigned detections start brand‐new tracks
            for i, f in enumerate(this_feats):
                if i in assigned:
                    continue
                trk = {
                    "id": next_track_id,
                    "frames": [t],
                    "centroids": [f["centroid"]],
                    "labels": [f["label"]],
                }
                tracks.append(trk)
                new_active.append((next_track_id, f["centroid"]))
                next_track_id += 1

            active_tracks = new_active

        # 4) Optionally, extract per‐track masks
        track_masks = {}
        for trk in tracks:
            # for simplicity, just store the mask of the last appearance:
            t_last = trk["frames"][-1]
            lbl = trk["labels"][-1]
            mask_t = masks[t_last] == lbl
            track_masks[trk["id"]] = mask_t

        return {
            "tracks": tracks,
            "track_masks": track_masks,
            "n_tracks": len(tracks),
        }
