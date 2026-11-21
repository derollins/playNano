"""Per-frame summaries of detected features, as a table or histogram."""

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd


def frame_summary_to_dataframe(
    features_per_frame: Sequence[Sequence[dict]],
) -> pd.DataFrame:
    """
    Summarise the features detected in each frame, one row per frame.

    Parameters
    ----------
    features_per_frame : list of list of dict
        The ``"features_per_frame"`` output of a feature detection module,
        such as ``feature_detection``. Each feature dict must have ``"area"``
        and ``"mean"``.

    Returns
    -------
    pandas.DataFrame
        One row per frame, with the columns:

        - ``frame_index`` (int): from 0.
        - ``n_features`` (int): number of features.
        - ``total_area`` and ``mean_area`` (float): feature area, in pixels.
        - ``mean_intensity`` (float): the average of the features' mean
          heights, usually in nm.

        ``mean_area`` and ``mean_intensity`` are 0.0 for a frame with no
        features.
    """
    rows = []
    for i, feats in enumerate(features_per_frame):
        areas = [f["area"] for f in feats]
        intensities = [f["mean"] for f in feats]
        rows.append(
            {
                "frame_index": i,
                "n_features": len(feats),
                "total_area": sum(areas),
                "mean_area": float(pd.Series(areas).mean()) if areas else 0.0,
                "mean_intensity": (
                    float(pd.Series(intensities).mean()) if intensities else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_frame_histogram(
    df: pd.DataFrame,
    column: str,
    ax: Optional[plt.Axes] = None,
    save_to: Optional[Path] = None,
    bins: int = 20,
) -> plt.Axes:
    """
    Plot a histogram of one column of a per-frame summary.

    Parameters
    ----------
    df : pandas.DataFrame
        A table from :func:`frame_summary_to_dataframe`.
    column : str
        The column to plot, for example ``"n_features"``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, a new figure is created. Default is None.
    save_to : str or pathlib.Path, optional
        If given, save the figure to this file at 150 dpi. Default is None.
    bins : int, optional
        Number of histogram bins. Default is 20.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.
    """
    if ax is None:
        fig, ax = plt.subplots()

    ax.hist(df[column], bins=bins, edgecolor="black")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    ax.set_title(f"Histogram of {column}")

    if save_to:
        ax.get_figure().savefig(save_to, dpi=150)
    return ax
