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
    grouping_output: Mapping[str, Any],
    detection_output: Mapping[str, Any],
    *,
    object_key: Optional[str] = None,
    object_id_field: str = "cluster_id",
    frame_key: str = "frames",
    index_key: str = "point_indices",
) -> pd.DataFrame:
    """
    Build a DataFrame linking each grouping analysis results to detected features.

    Each object (e.g. cluster or track) is linked to its corresponding detected
    feature metadata using (frame index, point index) pairs to locate features in
    the output of a feature detection step and merges metadata into one flattened
    table.

    Parameters
    ----------
    grouping_output : dict
        Dictionary from a grouping module (e.g. clustering or tracking).
        Must contain a list of group objects under the `object_key`, where each
        object has lists of `frames` and `point_indices`.

    detection_output : dict
        Dictionary from a detection module (e.g. feature_detection), which must
        contain the key 'features_per_frame': a list of feature dicts per frame.

    object_key : str, optional
        Key in `grouping_output` pointing to the list of group objects.
        Default is "clusters".

    object_id_field : str, optional
        Column name to use in the output DataFrame to identify the group,
        e.g., "cluster_id" or "track_id". Default is "cluster_id".

    frame_key : str, optional
        Key in each group object listing the frames the object appears in.
        Default is "frames".

    index_key : str, optional
        Key in each group object listing the per-frame point indices (used
        to match detections in `features_per_frame`). Default is "point_indices".

    Returns
    -------
    pd.DataFrame
        Flattened DataFrame linking features to group membership.
        Includes feature metadata and:
        - object_id_field (e.g. "cluster_id")
        - frame
        - timestamp
        - label
        - centroid_x, centroid_y
        - area
        - mean_intensity
        - min_intensity
        - max_intensity
    """
    if object_key is None:
        if "tracks" in grouping_output:
            object_key = "tracks"
            object_id_field = "track_id"
        elif "clusters" in grouping_output:
            object_key = "clusters"
            object_id_field = "cluster_id"
        else:
            raise ValueError("Unable to autodetect object_key. Please specify.")

    features_per_frame = detection_output.get("features_per_frame", [])
    rows = []

    for obj in grouping_output.get(object_key, []):
        cid = obj["id"]
        frames = obj.get(frame_key)
        point_indices = obj.get(index_key)
        if frames is None or point_indices is None:
            raise KeyError(
                f"Grouping objects must have '{frame_key}' and '{index_key}' lists"
            )

        for frame_idx, pt_idx in zip(frames, point_indices, strict=False):
            # Defensive: skip if index out of range
            if frame_idx >= len(features_per_frame):
                continue
            frame_features = features_per_frame[frame_idx]
            if pt_idx >= len(frame_features):
                continue
            feat = frame_features[pt_idx]

            # Build row dict
            row = row = {
                object_id_field: cid,
                "frame": frame_idx,
                "timestamp": feat.get("frame_timestamp", np.nan),
                "label": feat.get("label", None),
                "centroid_x": feat["centroid"][0],  # x
                "centroid_y": feat["centroid"][1],  # y
                "area": feat.get("area", np.nan),
                "mean": feat.get("mean", np.nan),
                "min": feat.get("min", np.nan),
                "max": feat.get("max", np.nan),
            }
            rows.append(row)

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
