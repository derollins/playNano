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


def flatten_particle_features(
    object_outputs: Mapping[str, Any],
    detection_outputs: Mapping[str, Any],
    *,
    object_key: str = "tracks",
    object_id_field: str = "track_id",  # or "cluster_id"
    frame_key: str = "frames",
    label_key: str = "labels",
) -> pd.DataFrame:
    """
    Build a long-form DataFrame from a particle-based analysis result.

    Parameters
    ----------
    object_outputs : dict
        Output from a particle analysis module (e.g. 'tracks' or 'clusters').
    detection_outputs : dict
        Output from feature detection, must include 'features_per_frame'.
    object_key : str
        Key in object_outputs (e.g. "tracks" or "clusters").
    object_id_field : str
        Name to use for object identifier column in the output
        (e.g. "track_id" or "cluster_id").
    frame_key : str
        Key in each object pointing to a list of frame indices.
    label_key : str
        Key in each object pointing to a list of detection labels.

    Returns
    -------
    pd.DataFrame
        Flattened table with detection + object metadata per frame.
    """
    rows = []
    det_index = {}
    for frame_idx, feats in enumerate(detection_outputs["features_per_frame"]):
        for feat in feats:
            det_index[(frame_idx, feat["label"])] = feat

    for obj in object_outputs[object_key]:
        oid = obj["id"]
        for frame_idx, label in zip(obj[frame_key], obj[label_key], strict=False):
            feat = det_index.get((frame_idx, label), {})
            rows.append(
                {
                    object_id_field: oid,
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

    return pd.DataFrame(rows)


def plot_particle_labels_3d(
    df: pd.DataFrame,
    object_id_field: str = "track_id",
    ax: Optional[plt.Axes] = None,
    save_to: Optional[Path] = None,
    cmap: str = "tab10",
) -> plt.Axes:
    """
    Plot particle ids in 3D (x, y, time), colored by object ID.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain ['centroid_x','centroid_y','timestamp', object_id_field]
    object_id_field : str
        Column to use for color grouping (e.g. "track_id", "cluster_id")
    ax : matplotlib Axes, optional
        A 3D Axes to draw into, or None to create a new one.
    save_to : Path, optional
        If given, save the figure to file.
    cmap : str
        Colormap name for particle group colors.

    Returns
    -------
    ax : Axes
        The 3D axes used.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    ids = df[object_id_field].unique()
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(ids)))

    for oid, c in zip(ids, colors, strict=False):
        sub = df[df[object_id_field] == oid]
        ax.scatter(
            sub["centroid_x"],
            sub["centroid_y"],
            sub["timestamp"],
            label=f"{object_id_field} {oid}",
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
