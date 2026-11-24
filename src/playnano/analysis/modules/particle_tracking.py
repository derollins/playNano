"""
Analysis module that links detected features from frame to frame into tracks.

Each feature is linked to the nearest feature of an existing track that is
close enough; features that are not linked start new tracks. Tracks can
continue across frames where the particle was not detected.

.. versionadded:: 0.2.0

See Also
--------
playnano.analysis.modules.feature_detection : Detects the features to track.
"""

from typing import Any, Optional, Sequence

import numpy as np

from playnano.afm_stack import AFMImageStack
from playnano.analysis.base import AnalysisModule


class ParticleTrackingModule(AnalysisModule):
    """
    Link features detected in successive frames into tracks.

    Pipeline name: ``"particle_tracking"``. It needs the output of
    ``feature_detection`` from an earlier step.

    In each frame, tracks are extended in the order they started: each takes
    the nearest feature not yet taken that is closer than ``max_distance`` to
    its last position. This is quick but not globally optimal: when particles
    come close, an older track can take a feature a newer one fits better.
    Features left over start new tracks.

    Version 0.2.0 added ``max_missing`` and ``distance_scale``. See :meth:`run`
    for the parameters and outputs.

    Examples
    --------
    >>> import numpy as np
    >>> from playnano.afm_stack import AFMImageStack
    >>> data = np.zeros((3, 64, 64))
    >>> for t in range(3):
    ...     data[t, 10 + t : 16 + t, 10:16] = 3.0  # a 6 x 6 pixel particle
    >>> stack = AFMImageStack(
    ...     data,
    ...     pixel_size_nm=2.0,
    ...     channel="height_trace",
    ...     file_path="demo.h5-jpk",
    ...     frame_metadata=[{"timestamp": 0.5 * t} for t in range(3)],
    ... )
    >>> from playnano.analysis.modules.feature_detection import (
    ...     FeatureDetectionModule,
    ... )
    >>> from playnano.analysis.modules.particle_tracking import (
    ...     ParticleTrackingModule,
    ... )
    >>> detected = FeatureDetectionModule().run(
    ...     stack, mask_fn="mask_threshold", threshold=1.0
    ... )
    >>> result = ParticleTrackingModule().run(
    ...     stack, {"feature_detection": detected}, max_distance=5
    ... )
    >>> track = result["tracks"][0]
    >>> track["frames"], track["coords"]
    ([0, 1, 2], [(12.5, 12.5), (13.5, 12.5), (14.5, 12.5)])
    """

    version = "0.2.0"

    @property
    def name(self) -> str:
        """Name of this module in a pipeline, ``"particle_tracking"``."""
        return "particle_tracking"

    requires = ["feature_detection", "log_blob_detection"]

    def _get_detection_outputs(
        self,
        previous_results: dict[str, Any],
        *,
        detection_module: str,
        coord_key: str,
    ) -> tuple[list[list[dict]], list[np.ndarray]]:
        """
        Retrieve per-frame features and labeled masks from a detection module.

        Parameters
        ----------
        previous_results : dict[str, Any]
            Results from earlier pipeline steps.
        detection_module : str
            Preferred detection module name. If not available, the most recent
            available module from ``self.requires`` is used.
        coord_key : str
            Key in the detection output containing per-frame features.

        Returns
        -------
        feats : list[list[dict]]
            Per-frame feature dicts.
        masks : list[np.ndarray]
            Per-frame labeled masks.

        Raises
        ------
        RuntimeError
            If no suitable detection module output is available, or required
            keys are missing.
        """
        if detection_module in previous_results:
            chosen = detection_module
        else:
            available = [m for m in reversed(self.requires) if m in previous_results]
            if not available:
                raise RuntimeError(
                    f"{self.name!r} requires one of {self.requires}, but none were found in previous results."  # noqa
                )
            chosen = available[0]

        fd_out = previous_results[chosen]

        if coord_key not in fd_out:
            raise RuntimeError(
                f"{self.name!r} expected detection output {chosen!r} to contain {coord_key!r}."  # noqa
            )
        if "labeled_masks" not in fd_out:
            raise RuntimeError(
                f"{self.name!r} expected detection output {chosen!r} to contain 'labeled_masks'."  # noqa
            )

        return fd_out[coord_key], fd_out["labeled_masks"]

    def _extract_coords(
        self, f: dict, coord_columns: Sequence[str]
    ) -> tuple[float, float]:
        """
        Extract 2D coordinates from a feature dictionary.

        Parameters
        ----------
        f : dict
            Feature dictionary.
        coord_columns : Sequence[str]
            Keys to use for coordinates. If missing, falls back to ``f["centroid"]``.

        Returns
        -------
        coords : tuple[float, float]
            (coord0, coord1) coordinate pair.

        Raises
        ------
        KeyError
            If neither ``coord_columns`` nor a valid ``centroid`` entry is present.
        """
        try:
            return tuple(float(f[k]) for k in coord_columns)
        except KeyError:
            c = f.get(
                "centroid"
            )  # 'centroid' is an output of playnano.analysis.modules.feature_detection
            if not c or len(c) < 2:
                raise KeyError(
                    f"Missing coordinate keys {coord_columns} and fallback 'centroid'"
                ) from None
            return tuple(c[:2])

    def _scaled_threshold(self, base: float, dt: int, mode: str) -> float:
        """
        Scale a base distance threshold by a frame gap.

        Parameters
        ----------
        base : float
            Base threshold (applied when dt == 1).
        dt : int
            Frame difference since the last detection (>= 1).
        mode : str
            One of {"constant", "linear", "sqrt"}.

        Returns
        -------
        float
            Scaled threshold.

        Raises
        ------
        ValueError
            If ``mode`` is not supported.
        """
        if mode == "constant":
            return base
        if mode == "linear":
            return base * dt
        if mode == "sqrt":
            return base * float(np.sqrt(dt))
        raise ValueError(
            "distance_scale must be one of {'constant','sqrt','linear'}; "
            f"got {mode!r}"
        )

    def _densify_track(self, trk: dict[str, Any]) -> None:
        """
        Convert sparse detections in a track to a dense within-span representation.

        The track's ``frames``, ``coords`` and ``point_indices`` arrays are replaced
        in-place so they span from the first to last detection (inclusive). Missing
        detections within the span are represented by ``None`` entries.

        Parameters
        ----------
        trk : dict[str, Any]
            Track dictionary with sparse lists:
            - frames : list[int]
            - coords : list[tuple[float, float]]
            - point_indices : list[int]

        Returns
        -------
        None
            The input track dict is modified in-place.
        """
        det_frames = trk["frames"]
        det_coords = trk["coords"]
        det_idxs = trk["point_indices"]

        start = det_frames[0]
        end = det_frames[-1]
        span_frames = list(range(start, end + 1))

        coords_dense: list[Optional[tuple[float, float]]] = [None] * len(span_frames)
        idx_dense: list[Optional[int]] = [None] * len(span_frames)

        for f, c, i in zip(det_frames, det_coords, det_idxs, strict=False):
            j = f - start
            coords_dense[j] = c
            idx_dense[j] = i

        trk["frames"] = span_frames
        trk["coords"] = coords_dense
        trk["point_indices"] = idx_dense

    def _last_detection(self, trk: dict[str, Any]) -> Optional[tuple[int, int]]:
        """
        Find the last valid (frame, point_index) pair in a dense track.

        Parameters
        ----------
        trk : dict[str, Any]
            Track dictionary containing ``frames`` and ``point_indices`` lists.

        Returns
        -------
        (frame_idx, point_idx) : tuple[int, int] or None
            Last detection pair, or None if no detections are present.
        """
        for t_f, i_f in zip(
            reversed(trk["frames"]), reversed(trk["point_indices"]), strict=False
        ):
            if i_f is not None:
                return t_f, i_f
        return None

    def run(
        self,
        stack: AFMImageStack,
        previous_results: Optional[dict[str, Any]] = None,
        detection_module: str = "feature_detection",
        coord_key: str = "features_per_frame",
        coord_columns: Sequence[str] = ("centroid_x", "centroid_y"),
        max_distance: float = 5.0,
        max_missing: int = 0,
        distance_scale: str = "constant",  # {"constant","sqrt","linear"}
        **params,
    ) -> dict[str, Any]:
        """
        Link the features detected in each frame into tracks.

        Parameters
        ----------
        stack : AFMImageStack
            The stack; not used directly.
        previous_results : dict
            Outputs of earlier steps. Must hold a detection output with the
            ``coord_key`` list and ``"labeled_masks"``.
        detection_module : str, optional
            Name of the detection step to use. If it has not run,
            ``log_blob_detection`` is tried, then ``feature_detection``. Default is
            ``"feature_detection"``.
        coord_key : str, optional
            Key of the per-frame feature lists in the detection output. Default is
            ``"features_per_frame"``.
        coord_columns : list or tuple of str, optional
            Keys of each feature's two coordinates. If a feature lacks them, its
            ``centroid`` is used instead. ``feature_detection`` has no
            ``centroid_x`` or ``centroid_y``, so with the default, coordinates are
            ``centroid``: (row, col), in pixels. Default is
            ``("centroid_x", "centroid_y")``.
        max_distance : float, optional
            Distance a particle can move between frames and still be linked, in
            the same units as the coordinates (pixels for ``feature_detection``).
            Only moves strictly shorter than this are linked. Default is 5.0.
        max_missing : int, optional
            Number of frames in a row a track can go without a feature before it
            ends. Default is 0: a track ends at its first missed frame.
        distance_scale : {"constant", "linear", "sqrt"}, optional
            How ``max_distance`` grows after missed frames, with ``dt`` the number
            of frames since the track's last feature: ``"constant"`` keeps it,
            ``"linear"`` multiplies it by ``dt`` and ``"sqrt"`` by ``sqrt(dt)``.
            Default is ``"constant"``.
        **params
            Not used; ignored.

        Returns
        -------
        dict
            ``"tracks"``
                One dict per track, with:

                - ``id`` (int): from 0, in the order tracks started.
                - ``frames`` (list of int): every frame from the track's first
                  feature to its last.
                - ``point_indices`` (list of int or None): the index of the
                  track's feature in each frame's list in the detection output,
                  or None for a missed frame.
                - ``coords`` (list of tuple or None): the feature's coordinates in
                  each frame, or None for a missed frame.
            ``"track_masks"``
                Maps each track ID to a boolean mask of its last feature, if the
                feature has a ``label``.
            ``"n_tracks"``
                The number of tracks.

        Raises
        ------
        RuntimeError
            If there is no detection output, or it lacks ``coord_key`` or
            ``"labeled_masks"``.
        ValueError
            If ``max_distance`` or ``max_missing`` is negative, or
            ``distance_scale`` is not recognised.
        KeyError
            If a feature has neither ``coord_columns`` nor ``centroid``.
        """

        if previous_results is None:
            raise RuntimeError(f"{self.name!r} requires previous results to run.")

        if max_missing < 0:
            raise ValueError("max_missing must be >= 0")
        if max_distance < 0:
            raise ValueError("max_distance must be >= 0")

        feats, masks = self._get_detection_outputs(
            previous_results,
            detection_module=detection_module,
            coord_key=coord_key,
        )

        n_frames = len(feats)
        tracks = []
        next_track_id = 0

        # List of dictionaries, each: {"id": int, "last_coord": (x,y),
        # "last_frame": int, "missing": int}
        active_tracks: list[dict[str, Any]] = []

        for t in range(n_frames):
            this_feats = feats[t]
            assigned: set[int] = set()
            new_active: list[dict[str, Any]] = []

            # Match existing tracks to nearest features
            for state in active_tracks:
                trk_id = state["id"]
                last_coord = state["last_coord"]
                last_frame = state["last_frame"]
                missing = state["missing"]

                dt = t - last_frame  # time since last detection (in frames)
                dist_thresh = self._scaled_threshold(max_distance, dt, distance_scale)

                best = None
                best_idx = None
                best_dist = dist_thresh

                for i, f in enumerate(this_feats):
                    if i in assigned:
                        continue
                    coords = self._extract_coords(f, coord_columns)
                    dist = np.hypot(
                        coords[0] - last_coord[0], coords[1] - last_coord[1]
                    )
                    if dist < best_dist:
                        best_dist, best, best_idx = dist, coords, i

                if best is not None:
                    track = tracks[trk_id]  # track id equals index in tracks list
                    track["frames"].append(t)
                    track["coords"].append(best)
                    track["point_indices"].append(best_idx)

                    assigned.add(best_idx)

                    new_active.append(
                        {
                            "id": trk_id,
                            "last_coord": best,
                            "last_frame": t,
                            "missing": 0,
                        }
                    )
                else:
                    missing += 1
                    if missing <= max_missing:
                        new_active.append(
                            {
                                "id": trk_id,
                                "last_coord": last_coord,
                                "last_frame": last_frame,
                                "missing": missing,
                            }
                        )
                    # else: retire (drop from active)

            # Start new tracks for unmatched detections
            for i, f in enumerate(this_feats):
                if i in assigned:
                    continue
                coords = self._extract_coords(f, coord_columns)
                trk = {
                    "id": next_track_id,
                    "frames": [t],
                    "coords": [coords],
                    "point_indices": [i],
                }
                tracks.append(trk)
                new_active.append(
                    {
                        "id": next_track_id,
                        "last_coord": coords,
                        "last_frame": t,
                        "missing": 0,
                    }
                )
                next_track_id += 1

            active_tracks = new_active

        for trk in tracks:
            self._densify_track(trk)

        # Generate per-track masks from last known frame/feature
        track_masks = {}
        for trk in tracks:
            last_det = self._last_detection(trk)
            if last_det is None:
                continue
            t_last, i_last = last_det
            label = feats[t_last][i_last].get("label")
            if label is not None:
                track_masks[trk["id"]] = masks[t_last] == label

        return {
            "tracks": tracks,
            "track_masks": track_masks,
            "n_tracks": len(tracks),
        }
