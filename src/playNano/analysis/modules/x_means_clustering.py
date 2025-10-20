"""
Module for X-Means clustering as part of the playNano analysis pipeline.

This module implements a version of the X-Means clustering algorithm,
an extension of K-Means that estimates the optimal number of clusters using the
Bayesian Information Criterion (BIC).

Based on:
Pelleg, D., & Moore, A. W. (2000). X-means: Extending K-means with Efficient
Estimation of the Number of Clusters. Carnegie Mellon University.
http://www.cs.cmu.edu/~dpelleg/download/xmeans.pdf

"""

import logging
from typing import Any, Optional, Sequence

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from playNano.afm_stack import AFMImageStack
from playNano.analysis.base import AnalysisModule

logger = logging.getLogger(__name__)


class XMeansClusteringModule(AnalysisModule):
    """
    Cluster features using the X-Means algorithm over (x, y[, t]) coordinates.

    This module clusters spatial (and optionally temporal) feature coordinates extracted
    from an AFM stack using an X-Means algorithm implemented in pure Python.

    Parameters
    ----------
    coord_key : str
        Key in previous_results[detection_module] to find feature list.

    coord_columns : Sequence[str]
        Names of feature dictionary keys to use for coordinates
        (e.g. centroid_x, centroid_y).

    use_time : bool
        Whether to append frame timestamps as the third coordinate.

    min_k : int
        Initial number of clusters (minimum).

    max_k : int
        Maximum number of clusters to allow.

    normalise : bool
        Whether to min-max normalize coordinate space before clustering.

    time_weight : float, optional
        Multiplier for time dimension (after normalization).

    Returns
    -------
    dict
        Dictionary with clustering results:
        - clusters: list of {id, frames, point_indices, coords}
        - cluster_centers: (K, D) ndarray in original units
        - summary: {n_clusters: int, members_per_cluster: dict}

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
        return "x_means_clustering"

    requires = ["feature_detection", "log_blob_detection"]

    def run(
        self,
        stack: AFMImageStack,
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
        replicates: int = 3,
        max_iter: int = 300,
        bic_threshold: float = 0.0,
    ) -> dict[str, Any]:
        """
        Perform X-Means clustering on features extracted from an AFM stack.

        This method extracts (x, y[, t]) coordinates from detected features,
        optionally normalizes and time-weights them, and applies the X-Means algorithm
        to automatically select the number of clusters based on the BIC score.

        Parameters
        ----------
        stack : AFMImageStack
            The input image stack providing frame timing and metadata context.

        previous_results : dict[str, Any], optional
            Dictionary containing outputs from previous analysis steps.
            Must contain the selected detection_module and coord_key.

        detection_module : str
            Key identifying which previous modules output to use.
            Default is "feature_detection".

        coord_key : str
            Key under the detection module that holds per-frame feature dicts.
            Default is "features_per_frame".

        coord_columns : Sequence[str]
            Keys to extract from each feature for clustering coordinates.
            If missing, will fall back to using the "centroid" tuple if available.
            Defaults is ("centroid_x", "centroid_y").

        use_time : bool
            If True and `coord_columns` only gives 2D coordinates, appends the
            frame timestamp as a third dimension. Default is True.

        min_k : int
            Initial number of clusters to start with. Default is 1.

        max_k : int
            Maximum number of clusters allowed. Defalut is 10.

        normalise : bool
            Whether to normalize the feature coordinate axes to the [0, 1] range
            before clustering. Default is True.

        time_weight : float or None, optional
            Multiplicative factor applied to the time axis (after normalization).
            Used only if time is included as a third coordinate.

        replicates : int
            Number of times to run k-means internally to choose the best split.
            Default is 3.

        max_iter : int
            Maximum number of iterations for each k-means call.
            Default is 300.

        bic_threshold : float
            Minimum improvement in BIC required to split a cluster.
            Default is 0.0 (any improvement allows a split).

        Returns
        -------
        dict
            A dictionary with the following keys:

            - "clusters" : list of dicts, each with:
                - id : int
                - frames : list of int
                - point_indices : list of int
                - coords : list of tuple (normalized x, y, [t])
            - "cluster_centers" : ndarray of shape (k, D)
                Final cluster centers in original (denormalized) coordinates.
            - "summary" : dict
                - "n_clusters" : int
                - "members_per_cluster" : dict mapping cluster ID to point count.

        Raises
        ------
        RuntimeError
            If the required detection module output is missing from previous_results.

        KeyError
            If the expected coordinate keys are missing from any feature dictionary.
        """
        # Validate input dependencies
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

        fd_out = previous_results[detection_module]
        per_frame = fd_out[coord_key]

        # Extract and format data points
        points, metadata = [], []
        for f_idx, feats in enumerate(per_frame):
            tval = stack.time_for_frame(f_idx)
            for p_idx, feat in enumerate(feats):
                try:
                    coords = [float(feat[c]) for c in coord_columns]
                except KeyError:
                    cent = feat.get("centroid")
                    if cent and len(cent) >= len(coord_columns):
                        coords = [float(cent[0]), float(cent[1])]
                    else:
                        raise KeyError(
                            f"Missing keys {coord_columns} in feature."
                        ) from None

                if use_time and len(coords) == 2:
                    coords.append(float(tval))

                points.append(coords)
                metadata.append((f_idx, p_idx))

        if not points:
            dim = 3 if use_time else len(coord_columns)
            return {
                "clusters": [],
                "cluster_centers": np.empty((0, dim)),
                "summary": {"n_clusters": 0, "members_per_cluster": {}},
            }

        data = np.array(points)

        # Normalize
        if normalise:
            mins, maxs = data.min(axis=0), data.max(axis=0)
            spans = maxs - mins
            spans[spans == 0] = 1.0
            data = (data - mins) / spans
            if time_weight is not None and data.shape[1] == 3:
                data[:, 2] *= time_weight

        # Run X-means
        labels, centers = core_xmeans(
            data,
            init_k=min_k,
            max_k=max_k,
            min_cluster_size=2,
            distance="sqeuclidean",
            replicates=replicates,
            max_iter=max_iter,
            bic_threshold=bic_threshold,
        )

        # Undo normalization on centers
        if normalise:
            if time_weight not in (None, 0.0) and centers.shape[1] == 3:
                centers[:, 2] /= time_weight
            centers = centers * spans + mins

        # Format output
        clusters_out, members = [], {}
        for cid in np.unique(labels):
            if cid < 0:
                continue
            idxs = np.where(np.atleast_1d(labels == cid))[0]
            frames, coords_list, p_inds = [], [], []
            for idx in idxs:
                f_idx, p_idx = metadata[idx]
                frames.append(f_idx)
                p_inds.append(p_idx)
                coords_list.append(tuple(data[idx]))
            clusters_out.append(
                {
                    "id": int(cid),
                    "frames": frames,
                    "point_indices": p_inds,
                    "coords": coords_list,
                }
            )
            members[int(cid)] = len(idxs)

        return {
            "clusters": clusters_out,
            "cluster_centers": centers,
            "summary": {"n_clusters": len(members), "members_per_cluster": members},
        }


def core_xmeans(
    data: np.ndarray,
    init_k: int,
    max_k: int,
    min_cluster_size: int,
    distance: str,
    replicates: int,
    max_iter: int,
    bic_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Core X-Means loop.

    Parameters are equivalent to those in `run` above.
    """
    k = init_k
    centers = initialize_centers(data, k)

    while k <= max_k:
        km = KMeans(
            n_clusters=k, n_init=replicates, max_iter=max_iter, random_state=42
        ).fit(data)
        labels = km.labels_
        centers = km.cluster_centers_

        new_centers = []
        split_occurred = False

        for j in range(k):
            pts = data[labels == j]
            if len(pts) < 2:
                new_centers.append(centers[j])
                continue

            km2 = KMeans(
                n_clusters=2, n_init=replicates, max_iter=max_iter, random_state=42
            ).fit(pts)
            labels2, centers2 = km2.labels_, km2.cluster_centers_

            if (
                sum(labels2 == 0) < min_cluster_size
                or sum(labels2 == 1) < min_cluster_size
            ):
                new_centers.append(centers[j])
                continue

            bic_parent = compute_bic(pts, centers[j : j + 1])
            bic_children = sum(
                compute_bic(pts[labels2 == lab], centers2[lab : lab + 1])
                for lab in [0, 1]
            )

            if bic_children - bic_parent > bic_threshold:
                new_centers.extend(centers2)
                split_occurred = True
            else:
                new_centers.append(centers[j])

        if not split_occurred or len(new_centers) > max_k:
            break

        centers = np.vstack(new_centers)
        k = len(centers)

    final_dists = cdist(data, centers, metric="sqeuclidean")
    final_labels = np.argmin(final_dists, axis=1)
    return final_labels, centers


def compute_bic(points: np.ndarray, center: np.ndarray) -> float:
    """Compute Bayesian Information Criterion for a cluster.

    Parameters
    ----------
    points : np.ndarray
        Points in the cluster.
    center : np.ndarray
        Cluster center (shape (1, D)).

    Returns
    -------
    float
        BIC value.
    """
    n, p = points.shape
    if n <= 1:
        return -np.inf
    sse = np.sum((points - center) ** 2)
    var = sse / (n - 1)
    if var <= 0:
        var = np.finfo(float).eps
    ll = -0.5 * n * p * np.log(2 * np.pi * var) - 0.5 * sse / var
    num_params = p + 1
    penalty = 0.5 * num_params * np.log(n)
    return ll - penalty


def initialize_centers(points: np.ndarray, k: int) -> np.ndarray:
    """Initialize k centers using a k-means++-like heuristic."""
    n = points.shape[0]
    centers = [points[np.random.choice(n)]]
    for _ in range(1, k):
        dists = np.min(cdist(points, np.vstack(centers), "sqeuclidean"), axis=1)
        probs = dists / dists.sum()
        cumprobs = np.cumsum(probs)
        r = np.random.rand()
        idx = np.searchsorted(cumprobs, r)
        centers.append(points[idx])
    return np.vstack(centers)
