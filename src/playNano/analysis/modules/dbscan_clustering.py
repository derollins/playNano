"""
DBSCAN clustering on features over the entire stack in 3D (x, y, time).

This module extracts feature points from a previous analysis step, optionally
normalizes them, applies DBSCAN, and returns clusters (with noise as label -1
omitted or optionally retained), cluster cores, and a summary.

Parameters
----------
coord_key : str
    Key in previous_results containing `features_per_frame`.
coord_columns : Sequence[str]
    Which keys in each feature-dict to use (e.g. ("x","y")).
use_time : bool
    If True and coord_columns length is 2, append frame time as the third dimension.
eps : float
    The maximum distance between two samples for them to be considered as in the
    same neighborhood (in normalized units if `normalise=True`).
min_samples : int
    The number of samples in a neighborhood for a point to be considered as a core
    point.
normalise : bool
    If True, min-max normalize each axis before clustering.
time_weight : float | None
    If given, multiply the time axis by this weight.
**dbscan_kwargs
    Forwarded to sklearn.cluster.DBSCAN.
"""

from typing import Any, Optional, Sequence

import numpy as np
from sklearn.cluster import DBSCAN

from playNano.analysis.base import AnalysisModule


class DBSCANClusteringModule(AnalysisModule):
    @property
    def name(self) -> str:
        return "dbscan_clustering"

    requires = ["feature_detection", "log_blob_detection"]

    def run(
        self,
        stack,
        previous_results: Optional[dict[str, Any]] = None,
        *,
        detection_module: str = "feature_detection",
        coord_key: str = "features_per_frame",
        coord_columns: Sequence[str] = ("centroid_x", "centroid_y"),
        use_time: bool = True,
        eps: float = 0.3,
        min_samples: int = 5,
        normalise: bool = True,
        time_weight: Optional[float] = None,
        **dbscan_kwargs: Any,
    ) -> dict[str, Any]:
        if previous_results is None or detection_module not in previous_results:
            raise RuntimeError(f"{self.name!r} requires output from {detection_module}")

        per_frame = previous_results[detection_module][coord_key]
        points, metadata = [], []
        for f_idx, feats in enumerate(per_frame):
            t = stack.time_for_frame(f_idx)
            for p_idx, feat in enumerate(feats):
                try:
                    coords = [float(feat[c]) for c in coord_columns]
                except KeyError:
                    cent = feat.get("centroid")
                    if not cent or len(cent) < len(coord_columns):
                        raise KeyError(
                            f"Missing keys {coord_columns} in feature"
                        ) from None
                    coords = [float(cent[0]), float(cent[1])]
                if use_time and len(coords) == 2:
                    coords.append(float(t))
                points.append(coords)
                metadata.append((f_idx, p_idx))

        if not points:
            dim = 3 if (use_time and len(coord_columns) == 2) else len(coord_columns)
            return {
                "clusters": [],
                "cluster_centers": np.empty((0, dim)),
                "summary": {"n_clusters": 0, "members_per_cluster": {}},
            }

        data = np.array(points)
        # normalize
        if normalise:
            mins, maxs = data.min(0), data.max(0)
            spans = maxs - mins
            spans[spans == 0] = 1.0
            data = (data - mins) / spans
            if time_weight is not None and data.shape[1] == 3:
                data[:, 2] *= time_weight

        # run DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples, **dbscan_kwargs)
        labels = clustering.fit_predict(data)

        # compute 'cluster centers' as mean of points in each cluster
        unique_labels = sorted(set(labels) - {-1})
        centers = []
        members = {}
        clusters_out = []
        for cid in unique_labels:
            idxs = np.where(labels == cid)[0].tolist()
            subset = data[idxs]
            center = subset.mean(axis=0)
            if normalise:
                if time_weight is not None and center.size == 3:
                    center[2] /= time_weight
                center = center * spans + mins
            centers.append(center)
            frames, p_inds, coords_list = [], [], []
            for idx in idxs:
                f_idx, p_idx = metadata[idx]
                frames.append(f_idx)
                p_inds.append(p_idx)
                coords_list.append(tuple(data[idx].tolist()))
            clusters_out.append(
                {
                    "id": cid,
                    "frames": frames,
                    "point_indices": p_inds,
                    "coords": coords_list,
                }
            )
            members[cid] = len(idxs)

        summary = {"n_clusters": len(unique_labels), "members_per_cluster": members}

        return {
            "clusters": clusters_out,
            "cluster_centers": np.array(centers),
            "summary": summary,
        }
