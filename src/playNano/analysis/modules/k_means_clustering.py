"""
K-Means clustering on features over the entire stack in 3D (x, y, time).

This module extracts a point-cloud from per-frame feature dictionaries
(e.g. coordinates + timestamps), optionally normalizes each axis to [0,1],
applies K-Means with a user-supplied k, then returns cluster assignments,
centers (in original units), and a summary.

Parameters
----------
coord_key : str
    Key in previous_results whose value is `features_per_frame`
    (list of lists of dicts).
coord_columns : Sequence[str]
    Which keys in each feature-dict to use (e.g. ("x","y")).
use_time : bool
    If True and coord_columns length is 2, append frame time as the third dimension.
k : int
    Number of clusters.
normalise : bool
    If True, min-max normalize each axis before clustering.
time_weight : float | None
    If given, multiply the time axis by this weight.
**kmeans_kwargs
    Forwarded to sklearn.cluster.KMeans.
"""

from typing import Any, Optional, Sequence

import numpy as np
from sklearn.cluster import KMeans

from playNano.analysis.base import AnalysisModule


class KMeansClusteringModule(AnalysisModule):
    @property
    def name(self) -> str:
        return "k_means_clustering"

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
        k: int,
        normalise: bool = True,
        time_weight: Optional[float] = None,
        **kmeans_kwargs: Any,
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
        # normalize each column
        if normalise:
            mins, maxs = data.min(0), data.max(0)
            spans = maxs - mins
            spans[spans == 0] = 1.0
            data = (data - mins) / spans
            if time_weight is not None and data.shape[1] == 3:
                data[:, 2] *= time_weight

        # run KMeans
        km = KMeans(n_clusters=k, **kmeans_kwargs)
        labels = km.fit_predict(data)
        centers = km.cluster_centers_.copy()

        # undo weighting/normalization on centers
        if normalise:
            if time_weight is not None and centers.shape[1] == 3:
                centers[:, 2] /= time_weight
            centers = centers * spans + mins

        # format output
        clusters_out, members = [], {}
        for cid in range(k):
            idxs = np.where(labels == cid)[0].tolist()
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

        summary = {"n_clusters": k, "members_per_cluster": members}
        return {
            "clusters": clusters_out,
            "cluster_centers": centers,
            "summary": summary,
        }
