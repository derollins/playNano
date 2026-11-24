"""
Analysis module that groups detected features with DBSCAN, over all frames.

DBSCAN groups points that have enough close neighbours and finds the number of
clusters itself. Points in sparse regions are treated as noise and left out.
"""

from typing import Any, Optional, Sequence

import numpy as np
from sklearn.cluster import DBSCAN

from playnano.analysis.base import AnalysisModule


class DBSCANClusteringModule(AnalysisModule):
    """
    Group detected features from all frames into clusters with DBSCAN.

    Pipeline name: ``"dbscan_clustering"``. It needs the output of
    ``feature_detection`` from an earlier step.

    Each detected feature becomes one point. By default its coordinates are the
    feature's ``centroid``, (row, col) in pixels, with the frame time in seconds
    added as a third coordinate. With ``normalise`` (the default), each coordinate
    is rescaled to the range 0 to 1 before clustering.

    A point with at least ``min_samples`` points (itself included) within ``eps``
    is a core point; clusters are core points within ``eps`` of each other plus
    their neighbours. Points in no cluster (noise) are not returned.
    Each cluster's centre is the mean of its points.

    Because coordinates are rescaled, ``eps`` is a fraction of the range of each
    coordinate. With a time coordinate, frames further apart than ``eps`` of the
    recording cannot be neighbours, so with few frames ``eps`` may need to be
    larger than the default, or ``time_weight`` below 1.

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
    >>> from playnano.analysis.modules.dbscan_clustering import (
    ...     DBSCANClusteringModule,
    ... )
    >>> result = DBSCANClusteringModule().run(
    ...     stack, previous, eps=0.5, min_samples=2
    ... )
    >>> result["summary"]
    {'n_clusters': 2, 'members_per_cluster': {0: 4, 1: 4}}
    >>> result["cluster_centers"].round(2)
    array([[12.5 , 14.  ,  0.75],
           [42.5 , 42.5 ,  0.75]])

    See Also
    --------
    sklearn.cluster.DBSCAN : The clustering used.
    """

    version = "0.1.0"

    @property
    def name(self) -> str:
        """Name of this module in a pipeline, ``"dbscan_clustering"``."""
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
        Cluster the detected features with DBSCAN.

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
        eps : float, optional
            Largest distance between two neighbouring points, in rescaled units if
            ``normalise``. Default is 0.3.
        min_samples : int, optional
            Number of points, itself included, a point needs within ``eps`` to be a
            core point. Default is 5.
        normalise : bool, optional
            Rescale each coordinate to the range 0 to 1 (minimum to maximum over
            all points) before clustering. Default is True.
        time_weight : float, optional
            Multiply the rescaled time coordinate by this, so time counts more (above
            1) or less (below 1) than position. Must be positive. Only used with
            ``normalise`` and a time coordinate. Default is None (no weighting).
        **dbscan_kwargs
            Passed to :class:`sklearn.cluster.DBSCAN`, for example ``metric``.

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
            # Plain int, not np.int64: it is used as a dict key below, and JSON
            # only accepts str, int, float, bool or None as keys.
            cid = int(cid)
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
