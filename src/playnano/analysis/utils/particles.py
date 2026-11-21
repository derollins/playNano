"""Turn tracks or clusters of detected features into a table, plot or CSV file."""

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
    Make a table with one row per feature in each track or cluster.

    Each object (a track or cluster) lists the frame and index of each of its
    features. These are looked up in the detection output to get each
    feature's measurements.

    Parameters
    ----------
    grouping_output : dict
        Output of a tracking or clustering module, for example
        :class:`~playnano.analysis.modules.particle_tracking.ParticleTrackingModule`.
        Each object must have an ``"id"`` and the lists named by ``frame_key``
        and ``index_key``.
    detection_output : dict
        Output of a detection module, with ``"features_per_frame"``: a list,
        for each frame, of feature dicts with at least ``"centroid"``.
    object_key : str, optional
        Key of the list of objects in ``grouping_output``. If None, ``"tracks"``
        is used if present (and the ID column is called ``"track_id"``),
        otherwise ``"clusters"`` (``"cluster_id"``), overriding
        ``object_id_field``. Default is None.
    object_id_field : str, optional
        Name of the ID column. Used only if ``object_key`` is given. Default
        is ``"cluster_id"``.
    frame_key : str, optional
        Key of each object's list of frame indices. Default is ``"frames"``.
    index_key : str, optional
        Key of each object's list of feature indices within those frames.
        Entries that are None (frames where a track has no feature) are
        skipped. Default is ``"point_indices"``.

    Returns
    -------
    pandas.DataFrame
        One row per feature, with the columns ``object_id_field``, ``frame``,
        ``timestamp`` (s), ``label``, ``centroid_x`` and ``centroid_y``
        (pixels), ``area`` (pixels), and ``mean``, ``min`` and ``max`` (height,
        usually in nm). Missing measurements are NaN. Frame or feature indices
        out of range are skipped.

    Raises
    ------
    ValueError
        If ``object_key`` is None and ``grouping_output`` has neither
        ``"tracks"`` nor ``"clusters"``.
    KeyError
        If an object lacks the ``frame_key`` or ``index_key`` list.
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
            # Dense tracks can include missing points
            if pt_idx is None:
                continue

            # Defensive: skip if index out of range
            if frame_idx < 0 or frame_idx >= len(features_per_frame):
                continue
            frame_features = features_per_frame[frame_idx]
            if pt_idx < 0 or pt_idx >= len(frame_features):
                continue
            feat = frame_features[pt_idx]

            # Build row dict
            row = {
                object_id_field: cid,
                "frame": frame_idx,
                "timestamp": feat.get("frame_timestamp", np.nan),
                "label": feat.get("label", None),
                # Follow scikit‑image’s convention for coordinates, row, col i.e. y, x
                "centroid_x": feat["centroid"][1],  # col (x)
                "centroid_y": feat["centroid"][0],  # row (y)
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
    Plot each feature's position against time in 3D, coloured by track or cluster.

    Parameters
    ----------
    df : pandas.DataFrame
        A table from :func:`flatten_particle_features`, with the columns
        ``centroid_x``, ``centroid_y``, ``timestamp`` and ``object_id_field``.
    object_id_field : str, optional
        Column that identifies each track or cluster. Default is
        ``"track_id"``.
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D, optional
        3D axes to draw on. If None, a new figure is created. Default is None.
    save_to : str or pathlib.Path, optional
        If given, save the figure to this file at 150 dpi. Default is None.
    cmap : str, optional
        Name of the matplotlib colormap for the objects. Default is
        ``"tab10"``.

    Returns
    -------
    mpl_toolkits.mplot3d.axes3d.Axes3D
        The axes drawn on, with x and y in pixels and time in seconds.
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
    Write a table from :func:`flatten_particle_features` to a CSV file.

    Parameters
    ----------
    df : pandas.DataFrame
        The table to write; the index is not written.
    out_path : pathlib.Path
        The CSV file. Missing folders are created. Must be a Path, not a str.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
