"""
Module for clustering particles using the X- means algorithmn.

This module implements the X-means clustering algorithm, an extension of the K-means
algorithm that automatically determines the optimal number of clusters by evaluating
splits using the Bayesian Information Criterion (BIC).

The X-means algorithm improves upon K-means by:
- Dynamically estimating the number of clusters.
- Using a local search to refine cluster boundaries.
- Applying model selection criteria to avoid overfitting.

Reference:
Pelleg, D., & Moore, A. W. (2000). X-means: Extending K-means with Efficient
Estimation of the Number of Clusters.
Carnegie Mellon University. http://www.cs.cmu.edu/~dpelleg/download/xmeans.pdf

"""

import logging
from typing import Any, Optional, Sequence

import numpy as np
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.xmeans import xmeans

from playNano.analysis.base import AnalysisModule

logger = logging.getLogger(__name__)


class XMeansClusteringModule(AnalysisModule):
    """
    Cluster features over the entire stack in 3D (x, y, time) using X-Means.

    You must supply a per-frame feature list (e.g. from `feature_detection`) under
    `coord_key`.  By default we extract each feature's 'centroid' → (x,y) and append
    its frame timestamp as t, producing a (3,) point.  X-Means then finds the
    optimal number of clusters in [min_k,max_k].

    Parameters
    ----------
    coord_key : str
        Key in previous_results containing `features_per_frame`: a list of lists
        of dicts, each dict holding at least one of:
          - explicit `coord_columns`: e.g. ("x","y","t")
          - a `centroid` tuple and use_time=True (to auto-append time).
    coord_columns : Sequence[str]
        Names of the keys in each feature-dict to use as your feature vector.
        Can be length 2 (x,y) or 3 (x,y,t).  If you give only (x,y), pass
        `use_time=True` to append timestamps automatically.
    use_time : bool
        If True and coord_columns is length 2, each feature will become
        (x,y,t) where t = stack.time_for_frame(frame_idx).
    min_k : int
        Minimum number of clusters to try.
    max_k : int
        Maximum number of clusters to try.
    **xmeans_kwargs
        Forwarded to the pyclustering `xmeans(...).process()` call.

    Returns
    -------
    dict with keys:
      - clusters: list of {
            id: int,
            frames: [frame indices…],
            coords: [(x,y,t)…],
            point_indices: [index-within-frame…],
        }
      - cluster_centers: np.ndarray of shape (n_clusters, D)
      - summary: {n_clusters: int, members_per_cluster: {id:count…}}

    Attributes
    ----------
    requires : list[str]
        Names of analysis modules whose outputs this module depends on.
    """

    @property
    def name(self) -> str:
        return "x_means_clustering"

    # Declare that we need cooridinate output from a previous module,
    # i.e. feature_detection.
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
        min_k: int = 1,
        max_k: int = 10,
        normalise: bool = True,
        time_weight: Optional[float] = None,
        **xmeans_kwargs: Any,
    ) -> dict[str, Any]:
        # 1) Dependency check
        if previous_results is None or detection_module not in previous_results:
            raise RuntimeError(
                f"{self.name!r} requires output from {detection_module} - please add it before clustering."  # noqa
            )

        fd_out = previous_results[detection_module]
        per_frame = fd_out[coord_key]  # List of lists of dicts
        # 2) Build point list + mapping metadata
        points, metadata = [], []
        for f_idx, feats in enumerate(per_frame):
            tval = stack.time_for_frame(f_idx)
            for p_idx, feat in enumerate(feats):
                # Extract base coords
                try:
                    coords = [float(feat[c]) for c in coord_columns]
                except KeyError:
                    # fallback: centroid tuple
                    cent = feat.get("centroid")
                    if cent and len(cent) >= len(coord_columns):
                        coords = [float(cent[0]), float(cent[1])]
                    else:
                        raise KeyError(
                            f"Missing keys {coord_columns} in feature"
                        ) from None
                # Append time dim if requested & not already present
                if use_time and len(coords) == 2:
                    coords.append(float(tval))
                if len(coords) == 3:
                    logger.info("Clusteing in three dimentions, x, y and time.")
                else:
                    logger.info("Clusteing in two dimentions, x and y.")
                points.append(coords)
                metadata.append((f_idx, p_idx))
        # 3) Handle no-features
        dim = (
            3
            if (use_time and coord_columns and len(coord_columns) == 2)
            else len(coord_columns)
        )
        if not points:
            return {
                "clusters": [],
                "cluster_centers": np.empty((0, dim)),
                "summary": {"n_clusters": 0, "members_per_cluster": {}},
            }

        data = np.array(points)

        # 4) Min- max normalise if requested.
        # Min-max normalize each column to [0,1]
        if normalise:
            mins = data.min(axis=0)
            maxs = data.max(axis=0)
            spans = maxs - mins
            spans[spans == 0] = 1.0
            data = (data - mins) / spans

            # If the user supplied a time_weight, apply it to the time column only
            if time_weight is not None and data.shape[1] == 3:
                # assume columns are [x, y, t]
                data[:, 2] = data[:, 2] * time_weight

        # 5) X-Means
        # Initialise centers for min_k clusters
        init_centers = kmeans_plusplus_initializer(data, min_k).initialize()
        xm = xmeans(data.tolist(), init_centers, kmax=max_k, **xmeans_kwargs)
        xm.process()
        cluster_idxs = xm.get_clusters()
        centers = np.array(xm.get_centers())

        # Reverse the normalization & weighting on centers
        if normalise:
            # undo time weight
            if time_weight not in (None, 0.0) and centers.shape[1] == 3:
                centers[:, 2] = centers[:, 2] / time_weight
            # un-normalize all dims
            centers = centers * spans + mins

        # 6) Format outputs
        clusters_out, members = [], {}
        for cid, idxs in enumerate(cluster_idxs):
            frames, coords_list, p_inds = [], [], []
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

        summary = {"n_clusters": len(cluster_idxs), "members_per_cluster": members}
        return {
            "clusters": clusters_out,
            "cluster_centers": centers,
            "summary": summary,
        }
