"""
Analysis module that groups detected features with X-means.

X-means finds the number of clusters itself: it repeatedly runs k-means and
splits clusters while splitting improves the Bayesian information criterion
(BIC). Based on:

Pelleg, D. and Moore, A. W. (2000). X-means: Extending K-means with Efficient
Estimation of the Number of Clusters. *Proceedings of the 17th International
Conference on Machine Learning*, 727-734.
"""

import logging
from typing import Any, Optional, Sequence

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from playnano.afm_stack import AFMImageStack
from playnano.analysis.base import AnalysisModule

logger = logging.getLogger(__name__)


class XMeansClusteringModule(AnalysisModule):
    """
    Group detected features from all frames with X-means, which picks the number.

    Pipeline name: ``"x_means_clustering"``. It needs the output of
    ``feature_detection`` from an earlier step.

    Each detected feature becomes one point. By default its coordinates are the
    feature's ``centroid``, (row, col) in pixels, with the frame time in seconds
    added as a third coordinate. With ``normalise`` (the default), each coordinate
    is rescaled to the range 0 to 1 before clustering.

    Starting with ``min_k`` clusters, each round runs k-means, then tries to split
    each cluster of 2 or more points in two. A cluster is split if the two halves,
    each of at least 2 points, have a higher combined BIC than the cluster by more
    than ``bic_threshold``. The next round runs k-means again, from scratch, with
    the new number of clusters. This stops when no cluster splits or splitting
    would give more than ``max_k`` clusters. The random seed of each k-means run
    is fixed, so results are repeatable.

    Examples
    --------
    Two particles over 4 frames, one moving to the right:

    >>> import numpy as np
    >>> from playnano.afm_stack import AFMImageStack
    >>> from playnano.analysis.modules.feature_detection import (
    ...     FeatureDetectionModule,
    ... )
    >>> data = np.zeros((4, 64, 64))
    >>> for t in range(4):
    ...     data[t, 10:16, 10 + t : 16 + t] = 3.0
    ...     data[t, 40:46, 40:46] = 3.0
    >>> stack = AFMImageStack(
    ...     data,
    ...     pixel_size_nm=2.0,
    ...     channel="height_trace",
    ...     file_path="demo.h5-jpk",
    ...     frame_metadata=[{"timestamp": 0.5 * t} for t in range(4)],
    ... )
    >>> detected = FeatureDetectionModule().run(
    ...     stack, mask_fn="mask_threshold", threshold=1.0
    ... )
    >>> previous = {"feature_detection": detected}
    >>> from playnano.analysis.modules.x_means_clustering import (
    ...     XMeansClusteringModule,
    ... )
    >>> result = XMeansClusteringModule().run(stack, previous, use_time=False)
    >>> result["cluster_centers"].round(2)
    array([[12.5, 15. ],
           [42.5, 42.5],
           [12.5, 13. ]])

    The still particle is one cluster; the moving one is split in two, because
    its positions spread more widely than one cluster explains well.
    """

    version = "0.1.0"

    @property
    def name(self) -> str:
        """Name of this module in a pipeline, ``"x_means_clustering"``."""
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
        Cluster the detected features with X-means.

        Parameters
        ----------
        stack : AFMImageStack
            The stack; used for the frame times.
        previous_results : dict
            Outputs of earlier steps. Must hold a detection output.
        detection_module : str, optional
            Name of the detection step to use. If it has not run,
            ``log_blob_detection`` is tried, then ``feature_detection``. Default is
            ``"feature_detection"``.
        coord_key : str, optional
            Key of the per-frame feature lists in the detection output. Default is
            ``"features_per_frame"``.
        coord_columns : list or tuple of str, optional
            Keys of each feature's coordinates. If a feature lacks them, the first
            two values of its ``centroid``, (row, col), are used instead.
            ``feature_detection`` has no ``centroid_x`` or ``centroid_y``, so this is
            what happens with the default. Default is
            ``("centroid_x", "centroid_y")``.
        use_time : bool, optional
            Add each feature's frame time, in seconds, as a third coordinate (only
            when there are two coordinates). Default is True.
        min_k : int, optional
            Number of clusters to start with. Must not be more than ``max_k``.
            Default is 1.
        max_k : int, optional
            Largest number of clusters allowed. Default is 10.
        normalise : bool, optional
            Rescale each coordinate to the range 0 to 1 (minimum to maximum over
            all points) before clustering. Default is True.
        time_weight : float, optional
            Multiply the rescaled time coordinate by this, so time counts more (above
            1) or less (below 1) than position. Must be positive. Only used with
            ``normalise`` and a time coordinate. Default is None (no weighting).
        replicates : int, optional
            Number of times each k-means run is started; the best is kept. Default
            is 3.
        max_iter : int, optional
            Largest number of iterations of each k-means run. Default is 300.
        bic_threshold : float, optional
            How much a split must raise the BIC to be accepted. Higher values give
            fewer clusters. Default is 0.0.

        Returns
        -------
        dict
            ``"clusters"``
                One dict per cluster, with ``id`` (int), and for each member point
                ``frames`` (frame index), ``point_indices`` (index in that frame's
                feature list) and ``coords`` (the point's coordinates as clustered,
                so rescaled if ``normalise``).
            ``"cluster_centers"``
                Array of shape (n_clusters, n_coordinates): the centre of each
                cluster, in the original units (pixels, and seconds for time).
            ``"summary"``
                ``n_clusters`` and ``members_per_cluster`` (cluster ID to number of
                points).

        Raises
        ------
        RuntimeError
            If ``previous_results`` holds no detection output.
        KeyError
            If the detection output lacks ``coord_key``, or a feature has neither
            ``coord_columns`` nor ``centroid``.
        ValueError
            If ``min_k`` is more than the number of points.

        Notes
        -----
        If ``min_k`` is more than ``max_k``, no clustering is done: the result has
        ``min_k`` centres picked at random from the points.
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
    """
    Run X-means on an array of points.

    Parameters
    ----------
    data : numpy.ndarray
        Points, shape (n_points, n_coordinates).
    init_k : int
        Number of clusters to start with.
    max_k : int
        Largest number of clusters allowed.
    min_cluster_size : int
        Smallest number of points in each half of a split.
    distance : str
        Not used; distances are always squared Euclidean.
    replicates : int
        Number of starts of each k-means run.
    max_iter : int
        Largest number of iterations of each k-means run.
    bic_threshold : float
        How much a split must raise the BIC to be accepted.

    Returns
    -------
    labels : numpy.ndarray
        Index of each point's cluster: the nearest centre.
    centers : numpy.ndarray
        Cluster centres, shape (n_clusters, n_coordinates).
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
    """
    Return the Bayesian information criterion (BIC) of one cluster; higher is better.

    The cluster is modelled as a spherical Gaussian around ``center``.

    Parameters
    ----------
    points : numpy.ndarray
        The cluster's points, shape (n_points, n_coordinates).
    center : numpy.ndarray
        The cluster's centre, shape (1, n_coordinates).

    Returns
    -------
    float
        The BIC, or ``-inf`` for fewer than 2 points.
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
    """
    Pick ``k`` starting centres from the points, spread out like k-means++.

    Uses NumPy's global random generator, which is not seeded.

    Parameters
    ----------
    points : numpy.ndarray
        Points, shape (n_points, n_coordinates).
    k : int
        Number of centres.

    Returns
    -------
    numpy.ndarray
        The centres, shape (k, n_coordinates).
    """
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
