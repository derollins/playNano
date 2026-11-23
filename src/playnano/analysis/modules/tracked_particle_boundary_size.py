# mypy: disable-error-code=type-arg
"""
Analysis module that measures the size of each tracked particle over time.

The size is the longer side of each region's bounding box, as found by
``particle_region_extraction``. A threshold can split the time points into two
states, for example compact and extended.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from playnano.analysis.base import AnalysisModule


class BoundarySizeModule(AnalysisModule):
    """
    Measure each tracked particle's size, frame by frame.

    Pipeline name: ``"tracked_particle_boundary_size"``. It needs the output of
    ``particle_region_extraction``, run with ``include_bbox=True`` (the
    default). The size, ``max_dim``, is the longer side of the region's tight
    bounding box, in pixels. See :meth:`run` for the parameters and outputs.
    """

    version = "0.1.0"

    @property
    def name(self) -> str:
        """Name of this module in a pipeline, ``"tracked_particle_boundary_size"``."""
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
        Measure the size of each tracked particle in each frame.

        Parameters
        ----------
        stack : AFMImageStack
            The stack; not used directly.
        previous_results : dict
            Outputs of earlier steps. Must hold the output of
            ``extraction_module``.
        extraction_module : str, optional
            Name of the region extraction step. Default is
            ``"particle_region_extraction"``.
        threshold : float, optional
            If given, each frame is also given a ``state``: 1 if ``max_dim`` is
            greater than ``threshold`` (in pixels), otherwise 0. Default is None.
        measure : str, optional
            The size measure. Only ``"bbox_max_dim"`` is available. Default is
            ``"bbox_max_dim"``.

        Returns
        -------
        dict
            ``"measure"``, ``"threshold"``
                The values used.
            ``"per_track"``
                One dict per track, with ``track_id``, ``frames``, ``timestamps``
                (s), ``max_dim`` (pixels) and, with a threshold, ``state``.
                Frames where the particle was not found have ``max_dim`` NaN and
                ``state`` NaN.
            ``"flat_table"``
                The same values with one dict per track and frame, with
                ``track_id``, ``label``, ``frame``, ``timestamp``, ``max_dim`` and,
                with a threshold, ``state``; ready for :class:`pandas.DataFrame`.
            ``"plot_hints"``
                Names of plotting functions in ``playnano_plugins``, a package that
                is not part of playNano.
            ``"summary"``
                ``n_tracks``, ``n_rows``, ``n_missing_region_measurements`` and
                ``state_included``.

        Raises
        ------
        RuntimeError
            If ``previous_results`` is None, lacks ``extraction_module``, or the
            extraction ran with ``include_bbox=False``.
        ValueError
            If ``measure`` is not ``"bbox_max_dim"``.
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
