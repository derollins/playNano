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
    """
    DBSCAN clustering of features across an AFMImageStack in (x, y, time) space.

    This module extracts coordinates from per-frame features, optionally adds time
    as a third dimension, normalizes the space, and applies DBSCAN clustering.
    It returns clusters with point metadata, core point means as cluster centers,
    and a summary of cluster sizes.

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
            The string identifier for this module: "dbscan_clustering".
        """
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
        """
        Perform DBSCAN clustering on detected features in (x, y[, t]) space.

        Parameters
        ----------
        stack : AFMImageStack
            The input stack with `.data` and `.time_for_frame()` method.

        previous_results : dict[str, Any], optional
            Output from previous analysis steps. Must contain features under
            the given `detection_module` and `coord_key`.

        detection_module : str
            Which module's output to use from `previous_results`. Default
            is "feature_detection".

        coord_key : str
            Key in `previous_results[detection_module]` containing the list
            of per-frame features. Default is "features_per_frame".

        coord_columns : Sequence[str]
            Keys to extract coordinates from each feature. If missing, will fall back
            to `centroid` tuple. Default is ("centroid_x", "centroid_y").

        use_time : bool
            Whether to append frame timestamp as a third coordinate. Dafaulr is True.

        eps : float
            Maximum distance for neighborhood inclusion (in normalized units if
            `normalise=True`). Default is 0.3.

        min_samples : int
            Minimum number of points in a neighborhood to form a core point.
            Default is 5.

        normalise : bool
            If True, normalize coordinate axes to [0, 1] range before clustering.
            Default is True.

        time_weight : float or None, optional
            Scaling factor for the time axis (after normalization). If None,
            no weighting is applied.

        **dbscan_kwargs : dict
            Additional keyword arguments forwarded to `sklearn.cluster.DBSCAN`.

        Returns
        -------
        dict[str, Any]
            Output dictionary with the following keys:

                - "clusters": list of dicts, one per cluster, containing:
                    - "id": cluster ID (int)
                    - "frames": list of frame indices
                    - "point_indices": list of feature indices within frames
                    - "coords": list of 2D or 3D coordinates (post-normalization)
                - "cluster_centers": np.ndarray of shape (n_clusters, D)
                    Mean location of each cluster in original coordinate units.
                - "summary": dict with:
                    - "n_clusters": total number of clusters found
                    - "members_per_cluster": dict of cluster ID to count
        """
        if previous_results is None:
            raise RuntimeError(f"{self.name!r} requires previous results to run.")

        # Auto-detect the most recent available detection module
        if detection_module not in previous_results:
            available = [
                mod for mod in reversed(self.requires) if mod in previous_results
            ]
            if not available:
                raise RuntimeError(
                    f"{self.name!r} requires one of {self.requires}, but none were found in previous results."  # noqa
                )
            detection_module = available[0]

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
