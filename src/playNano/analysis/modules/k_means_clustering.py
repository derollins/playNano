"""
K-Means clustering on features over the entire stack in 3D (x, y, time).

This module extracts a point-cloud from per-frame feature dictionaries
(e.g. coordinates + timestamps), optionally normalizes each axis to [0,1],
applies K-Means with a user-supplied k, then returns cluster assignments,
cluster centers (in original coordinate units), and a summary.

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
    """
    Cluster features across all frames using K-Means in 2D or 3D (x, y, [time]).

    Extracts point coordinates from per-frame features, applies optional normalization
    and time weighting, then performs K-Means clustering. Returns cluster assignments,
    centers in original scale, and a summary report.

    Parameters
    ----------
    coord_key : str
        Key in previous_results pointing to 'features_per_frame' structure.

    coord_columns : Sequence[str]
        Keys to extract coordinates from each feature (e.g. ("x", "y")).

    use_time : bool
        If True, appends frame timestamp as a third clustering dimension.

    k : int
        Number of clusters to fit.

    normalise : bool
        If True, normalize each axis to [0, 1] before clustering.

    time_weight : float or None
        Optional multiplier for time axis after normalization.

    **kmeans_kwargs
        Additional keyword arguments passed to sklearn.cluster.KMeans.

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
            The string identifier for this module.
        """
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
        """
        Perform K-Means clustering on features extracted from a stack.

        Constructs a coordinate array from features (x, y[, t]), optionally applies
        normalization and time weighting, and fits k-means to assign clusters.

        Parameters
        ----------
        stack : AFMImageStack
            The input image stack providing frame times and data context.

        previous_results : dict[str, Any], optional
            Dictionary containing outputs from previous analysis steps.
            Must contain the selected detection_module and coord_key.

        detection_module : str
            Key identifying which previous module's output to use.
            Default is "feature_detection".

        coord_key : str
            Key under the detection module that holds per-frame feature dicts.
            Default is "features_per_frame".

        coord_columns : Sequence[str]
            Keys to extract from each feature for clustering coordinates.
            If missing, fallback to the "centroid" tuple is attempted.
            Default is ("centroid_x", "centroid_y")

        use_time : bool
            If True and `coord_columns` is 2D, append frame timestamp as third
            dimension. Default is True.

        k : int
            Number of clusters to compute.

        normalise : bool
            Whether to min-max normalize each axis of the feature points before
            clustering. Default is True.

        time_weight : float or None, optional
            Weighting factor for time axis (applied after normalization).
            Only used if time is included as a third dimension.

        **kmeans_kwargs : dict
            Additional arguments forwarded to `sklearn.cluster.KMeans`.

        Returns
        -------
        dict
            A dictionary with the following keys:

            - "clusters" : list of dicts, each with:
                - id : int
                - frames : list of int
                - point_indices : list of int
                - coords : list of tuple
                    The normalized coordinates used in clustering for each point
                    in the cluster (e.g., (x, y[, t])).

            - "cluster_centers" : ndarray of shape (k, D)
                Cluster centers in original coordinate units.
            - "summary" : dict
                - "n_clusters" : int
                - "members_per_cluster" : dict mapping cluster id to point count

        Raises
        ------
        RuntimeError
            If the required `detection_module` output is not found in previous_results.

        KeyError
            If the required coordinate keys are missing in any feature dictionary.
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
        # normalize each column
        if normalise:
            mins, maxs = data.min(0), data.max(0)
            spans = maxs - mins
            spans[spans == 0] = 1.0
            data = (data - mins) / spans
            if time_weight is not None and data.shape[1] == 3:
                data[:, 2] *= time_weight

        # run KMeans
        km = KMeans(n_clusters=k, random_state=42, **kmeans_kwargs)
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
