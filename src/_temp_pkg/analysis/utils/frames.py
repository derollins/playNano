"""
Frame-based post-processing helpers.

E.g. summarize per-frame statistics or produce histograms.
"""

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd


def frame_summary_to_dataframe(
    features_per_frame: Sequence[Sequence[dict]],
) -> pd.DataFrame:
    """
    Build a DataFrame with one row per frame.

    Summarises number of features, total area, mean intensity, etc.

    Parameters
    ----------
    features_per_frame : list of lists of dict
        As returned by `FeatureDetectionModule.run()["features_per_frame"]`.

    Returns
    -------
    pd.DataFrame
        Columns:
          - frame_index (int)
          - n_features  (int)
          - total_area  (int)
          - mean_area   (float)
          - mean_intensity (float)
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
    Plot a histogram of a per-frame summary metric.

    Parameters
    ----------
    df : pandas.DataFrame
        As returned by `frame_summary_to_dataframe`.
    column : str
        Which column to histogram (e.g. 'n_features').
    ax : matplotlib Axes, optional
    save_to : Path, optional
    bins : int
        Number of histogram bins.

    Returns
    -------
    ax : matplotlib Axes
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
