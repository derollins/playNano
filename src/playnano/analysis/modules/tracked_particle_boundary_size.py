# mypy: disable-error-code=type-arg
"""
Particle boundary size analysis module for the playNano analysis pipeline.

This module computes a per-particle boundary size metric over time by consuming
the output of ``particle_region_extraction``.

For each tracked particle and frame, the maximum dimension of the (tight)
bounding box is recorded.  Optionally, a threshold may be applied to classify
each time point into a discrete state (e.g. compact vs extended).

The module is designed to integrate with the playNano ``AnalysisPipeline`` and
records results in a provenance-aware, serializable format suitable for
downstream analysis and plotting.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from playnano.analysis.base import AnalysisModule


class BoundarySizeModule(AnalysisModule):
    """
    Measure particle boundary size over time from pre-extracted particle regions.

    This module consumes output from ``ParticleRegionExtractionModule``
    (registered as ``"particle_region_extraction"`` in the pipeline).  It
    expects that module to have been run with ``include_bbox=True``.

    The boundary size metric is derived from the tight bounding box:

    ``max_dim = max(bbox_height, bbox_width)``

    Optionally, a threshold can be applied to derive a discrete state variable
    (e.g. compact vs extended).

    Parameters accepted in :meth:`run`
    ------------------------------------
    extraction_module : str, default ``"particle_region_extraction"``
        Name of the upstream extraction module in the pipeline.
    threshold : float or None
        If provided, each frame is classified as ``state = int(max_dim > threshold)``.
    measure : str, default ``"bbox_max_dim"``
        Metric name (currently only ``"bbox_max_dim"`` is supported).

    Notes
    -----
    - If a region entry is ``None`` (missing detection) or its bounding box is
      ``None``, ``max_dim`` is recorded as ``np.nan``.
    - If the upstream extraction module was run without ``include_bbox=True``,
      a ``RuntimeError`` is raised.
    """

    version = "0.1.0"

    @property
    def name(self) -> str:
        """Module name for registration in the analysis pipeline."""
        return "tracked_particle_boundary_size"

    requires = ["particle_region_extraction"]

    def run(
        self,
        stack: Any,
        previous_results: Optional[Dict[str, Any]] = None,
        *,
        extraction_module: str = "particle_region_extraction",
        threshold: Optional[float] = None,
        measure: str = "bbox_max_dim",
    ) -> Dict[str, Any]:
        """
        Execute the boundary size analysis.

        Parameters
        ----------
        stack : AFMImageStack
            Input image stack (not used directly; present for pipeline API
            compatibility).
        previous_results : dict, optional
            Results from earlier pipeline modules.  Must contain output from
            ``extraction_module`` with ``include_bbox=True``.
        extraction_module : str, default ``"particle_region_extraction"``
            Key identifying the region extraction module output in
            ``previous_results``.
        threshold : float or None, optional
            Threshold applied to ``max_dim`` to compute a binary state
            variable.  If ``None``, no state classification is performed.
        measure : str, default ``"bbox_max_dim"``
            Name of the boundary size metric.  Currently only
            ``"bbox_max_dim"`` is supported.

        Returns
        -------
        dict
            Keys:

            ``measure`` : str
                Name of the boundary size metric used.
            ``threshold`` : float or None
                Threshold used for state classification, or ``None``.
            ``per_track`` : list of dict
                Per-track time series.  Each dict has:

                - ``"track_id"`` : int
                - ``"frames"`` : list of int
                - ``"timestamps"`` : list of float
                - ``"max_dim"`` : list of float (``np.nan`` where absent)
                - ``"state"`` : list of int/float  *(only if threshold given)*

            ``flat_table`` : list of dict
                Row-per-frame representation suitable for ``pandas.DataFrame``.
            ``plot_hints`` : dict
                Suggested plotting helper function names.
            ``summary`` : dict
                Bookkeeping counters.
        """
        if previous_results is None:
            raise RuntimeError(f"{self.name!r} requires previous results to run.")

        if measure != "bbox_max_dim":
            raise ValueError(
                f"{self.name!r} only supports measure='bbox_max_dim' right now."
            )

        if extraction_module not in previous_results:
            raise RuntimeError(
                f"{self.name!r} requires extraction_module={extraction_module!r} "
                f"to be present in previous_results. "
                f"Add ParticleRegionExtractionModule before this module in the pipeline."  # noqa: E501
            )

        ext_out = previous_results[extraction_module]

        # Validate that bbox data is present
        ext_summary = ext_out.get("summary", {})
        if not ext_summary.get("include_bbox", True):
            raise RuntimeError(
                f"{self.name!r} requires the upstream {extraction_module!r} to have "
                f"been run with include_bbox=True."
            )

        rows: List[Dict[str, Any]] = []
        per_track: List[Dict[str, Any]] = []
        n_missing = 0

        for trk in ext_out.get("per_track", []):
            track_id = int(trk["track_id"])
            frames = trk.get("frames", [])
            timestamps = trk.get("timestamps", [])
            regions = trk.get("regions", [])

            track_max_dim: List[float] = []
            track_state: Optional[List[Any]] = [] if threshold is not None else None

            for frame_idx, ts, region in zip(frames, timestamps, regions, strict=False):
                max_dim = self._max_dim_from_region(region)

                if np.isnan(max_dim):
                    n_missing += 1

                track_max_dim.append(float(max_dim))

                label_val = (
                    region.get("label", np.nan) if region is not None else np.nan
                )

                row: Dict[str, Any] = {
                    "track_id": track_id,
                    "label": label_val,
                    "frame": frame_idx,
                    "timestamp": ts,
                    "max_dim": float(max_dim) if not np.isnan(max_dim) else np.nan,
                }

                if threshold is not None:
                    state_val: Any = (
                        np.nan if np.isnan(max_dim) else int(max_dim > threshold)
                    )
                    row["state"] = state_val
                    if track_state is not None:
                        track_state.append(state_val)

                rows.append(row)

            trk_rec: Dict[str, Any] = {
                "track_id": track_id,
                "frames": frames,
                "timestamps": timestamps,
                "max_dim": track_max_dim,
            }
            if threshold is not None:
                trk_rec["state"] = track_state

            per_track.append(trk_rec)

        return {
            "measure": measure,
            "threshold": threshold,
            "per_track": per_track,
            "flat_table": rows,
            "plot_hints": {
                "single_track": "playnano_plugins.plotting.plot_boundary_over_time",
                "multi_track": "playnano_plugins.plotting.plot_boundary_over_time_multiple_tracks",  # noqa: E501
            },
            "summary": {
                "n_tracks": len(ext_out.get("per_track", [])),
                "n_rows": len(rows),
                "n_missing_region_measurements": n_missing,
                "state_included": threshold is not None,
            },
        }

    @staticmethod
    def _max_dim_from_region(region: Optional[Dict[str, Any]]) -> float:
        """
        Compute ``max_dim`` from a region dict's ``bbox_tight``.

        Returns ``np.nan`` if the region is ``None`` or has no bounding box.
        """
        if region is None:
            return np.nan
        bbox = region.get("bbox_tight")
        if bbox is None:
            return np.nan
        minr, minc, maxr, maxc = bbox
        return float(max(maxr - minr, maxc - minc))
