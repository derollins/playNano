"""
Particle-based postprocessing helpers.

These functions take the raw outputs of feature detection and
particle tracking modules and turn them into tabular data,
plots, and CSV/HDF5 exports.
"""

from pathlib import Path
from typing import Any, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def flatten_tracks(
    tracking_outputs: Mapping[str, Any],
    detection_outputs: Mapping[str, Any],
) -> pd.DataFrame:
    """
    Build a long-form DataFrame with one row per (frame, feature).

    Parameters
    ----------
    tracking_outputs : dict
        The dict returned by `ParticleTrackingModule.run()`, e.g.
        {
          "tracks": [
             {"id": 0, "frames": [...], "centroids": [...], "labels": [...]},
             ...
          ],
          "track_masks": {...},  # optional
          "n_tracks": int,
        }
    detection_outputs : dict
        The dict returned by `FeatureDetectionModule.run()`, e.g.
        {
          "features_per_frame": List[List[dict]],
          "labeled_masks": List[np.ndarray],
          "summary": {...},
        }

    Returns
    -------
    pd.DataFrame
        Columns:
          - track_id  (int)
          - frame     (int)
          - timestamp (float)
          - label     (int)
          - centroid_x  (float)
          - centroid_y  (float)
          - area        (int)
          - mean, min, max  (float)
    """
    rows = []
    # Build a quick lookup: (frame_idx, label) -> detection dict
    det_index = {}
    for frame_idx, feats in enumerate(detection_outputs["features_per_frame"]):
        for feat in feats:
            det_index[(frame_idx, feat["label"])] = feat

    for tr in tracking_outputs["tracks"]:
        tid = tr["id"]
        for frame_idx, label in zip(tr["frames"], tr["labels"], strict=False):
            feat = det_index.get((frame_idx, label), {})
            rows.append(
                {
                    "track_id": tid,
                    "frame": frame_idx,
                    "timestamp": feat.get("frame_timestamp", np.nan),
                    "label": label,
                    "centroid_x": feat.get("centroid", (np.nan, np.nan))[0],
                    "centroid_y": feat.get("centroid", (np.nan, np.nan))[1],
                    "area": feat.get("area", np.nan),
                    "mean_intensity": feat.get("mean", np.nan),
                    "min_intensity": feat.get("min", np.nan),
                    "max_intensity": feat.get("max", np.nan),
                }
            )

    df = pd.DataFrame(rows)
    return df


def plot_tracks_3d(
    df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    save_to: Optional[Path] = None,
    cmap: str = "tab10",
) -> plt.Axes:
    """
    Scatter the particle centroids in a 3D plot (x, y, time), colored by track_id.

    Parameters
    ----------
    df : pandas.DataFrame
        As returned by `flatten_tracks`, must contain columns
        ['centroid_x','centroid_y','timestamp','track_id'].
    ax : matplotlib Axes, optional
        If provided, must be a 3D Axes (`projection='3d'`). Otherwise a new
        figure/axes is created.
    save_to : Path, optional
        If given, will save `fig.savefig(save_to)`.
    cmap : str, optional
        Matplotlib colormap name for discrete track colors.

    Returns
    -------
    ax : matplotlib Axes
        The 3D axes containing the scatter.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    track_ids = df["track_id"].unique()
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(track_ids)))

    for tid, c in zip(track_ids, colors, strict=False):
        sub = df[df["track_id"] == tid]
        ax.scatter(
            sub["centroid_x"],
            sub["centroid_y"],
            sub["timestamp"],
            label=f"track {tid}",
            color=c,
        )

    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    ax.set_zlabel("Time (s)")
    ax.legend()

    if save_to:
        ax.get_figure().savefig(save_to, dpi=150)

    return ax


def export_particle_csv(df: pd.DataFrame, out_path: Path) -> None:
    """
    Write the flattened track DataFrame to CSV.

    Parameters
    ----------
    df : pandas.DataFrame
    out_path : Path
        Path to write the .csv file (will create parent dirs).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
