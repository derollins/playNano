"""
Module for linking particle features across frames to build trajectories.

This module defines the ParticleTrackingModule, which links features
detected in sequential frames of an AFM image stack using nearest-neighbor
matching based on feature coordinates.

Features are matched across frames if they lie within a specified maximum
distance. Tracks are formed by chaining these matches over time.

Each resulting track includes:
    - A unique track ID
    - A list of frames spanned by the track (first→last detection)
    - A list of point indices aligned with frames; missing detections are None
    - A list of coordinates describing the particle's positions

Optionally, per-track masks are extracted from the labeled feature masks.

See Also
--------
playnano.analysis.modules.feature_detection : Mask based particle detection method.
playnano.analysis.modules.log_blob_detection : LoG-based particle detection method.

.. versionadded:: 0.2.0

Author
------
Daniel E. Rollins (d.e.rollins@leeds.ac.uk) / GitHub: derollins

AI Transparency Note
--------------------
AI-based tools were used for limited typing/formatting assistance
and for debugging, refactoring, and documentation suggestions. All code paths,
algorithms, and final behaviour were reviewed and validated by the author.
"""

from typing import Any, Optional, Sequence

import numpy as np

from playnano.afm_stack import AFMImageStack
from playnano.analysis.base import AnalysisModule


class ParticleTrackingModule(AnalysisModule):
    """
    Link detected features frame-to-frame to produce particle trajectories.

    This module links features detected by a prior feature detection module
    using nearest-neighbor coordinate matching across adjacent frames. A new
    track is created for each unmatched feature.

    Version
    -------
    0.2.0

    Version 0.2.0 allows tracks to continue across missing detections
    and supports scaling the distance threshold with time since last detection.
    """

    version = "0.2.0"

    @property
    def name(self) -> str:
        """
        Name of the analysis module.

        Returns
        -------
        str
            Unique identifier: "particle_tracking".
        """
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
        Track particles across frames using nearest-neighbor association.

        Parameters
        ----------
        stack : AFMImageStack
            The input AFM image stack.

        previous_results : dict[str, Any], optional
            Must contain results from a detection module, including:

            - coord_key (e.g., "features_per_frame"): list of dicts with per-frame
              features
            - "labeled_masks": per-frame mask of label regions

        detection_module : str, optional
            Which module to read features from (default: "feature_detection").

        coord_key : str, optional
            Key in previous_results[detection_module] containing per-frame feature
            dicts (default: "features_per_frame").

        coord_columns : Sequence[str], optional
            Keys to extract coordinates from each feature; falls back to "centroid" if
            needed. Default is ("centroid_x", "centroid_y").

        max_distance : float, optional
            Maximum allowed movement per frame in coordinate units (default: 5.0).

        max_missing : int, optional
            Maximum allowed consecutive missing detections before terminating a track.
            Default is 0 (no missing allowed).

        distance_scale : str, optional
            How to scale max_distance with time since last detection in frames:
            "constant" (default), "linear" (scale linearly with time), or "sqrt" (scale
            with square root of time).

        Returns
        -------
        dict
            Dictionary with keys:

            - tracks : list of dict
                Per-track dictionaries containing:

                - id : int
                  Track ID

                - frames : list[int]
                  Frame indices spanned by track

                - point_indices : list[Optional[int]]
                  Indices into features_per_frame

                - coords : list[Optional[tuple[float, float]]]
                  Coordinates (coord0, coord1) aligned with frames; None indicates
                  missing detection.

            - track_masks : dict[int, np.ndarray]
              Last mask per track

            - n_tracks : int
              Total number of tracks
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
