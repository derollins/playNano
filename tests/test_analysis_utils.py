"""Tests for analysis utils"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from playNano.analysis.utils import common, frames, particles

matplotlib.use("Agg")  # Use a non-interactive backend suitable for testing


# --- Common Utils ---


def test_numpy_encoder_serializes_ndarray():
    data = {"arr": np.array([1, 2, 3])}
    json_str = json.dumps(data, cls=common.NumpyEncoder)
    assert json_str == '{"arr": [1, 2, 3]}'


def test_numpy_encoder_raises_for_unserializable():
    class Dummy:
        pass

    data = {"obj": Dummy()}
    with pytest.raises(TypeError):
        json.dumps(data, cls=common.NumpyEncoder)


# Sample nested record
sample_record = {
    "metadata": {"experiment": "test", "version": 1.0},
    "results": {"values": [1, 2, 3], "array": np.array([4.5, 5.5])},
}


def test_export_to_hdf5_creates_file():
    """export_to_hdf5 creates an HDF5 file on disk."""
    with TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test.h5"
        common.export_to_hdf5(sample_record, out_path)
        assert out_path.exists()


def test_export_to_hdf5_structure_and_values():
    """export_to_hdf5 writes correct structure and values."""
    with TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test.h5"
        common.export_to_hdf5(sample_record, out_path)
        with h5py.File(out_path, "r") as f:
            root = f["analysis_record"]
            assert "metadata" in root
            assert "results" in root

            # Check scalar attributes

            assert root["metadata"]["experiment"].attrs["value"] == "test"
            assert root["metadata"]["version"].attrs["value"] == 1.0

            # Check array values
            values_ds = root["results"]["values"]
            values_ds = values_ds[
                "values"
            ]  # Access the actual dataset inside the group

            values = [json.loads(v) for v in values_ds[:]]
            assert values == sample_record["results"]["values"]


# --- Frame Utils ---

# Mock input data
tracking_outputs = {
    "tracks": [
        {"id": 1, "frames": [0, 1], "centroids": [(5, 5), (6, 6)], "labels": [10, 11]}
    ],
    "n_tracks": 1,
}

detection_outputs = {
    "features_per_frame": [
        [
            {
                "label": 10,
                "frame_timestamp": 0.0,
                "centroid": (5.0, 5.0),
                "area": 100,
                "mean": 1.0,
                "min": 0.5,
                "max": 1.5,
            }
        ],
        [
            {
                "label": 11,
                "frame_timestamp": 1.0,
                "centroid": (6.0, 6.0),
                "area": 110,
                "mean": 1.1,
                "min": 0.6,
                "max": 1.6,
            }
        ],
    ],
    "labeled_masks": [],
    "summary": {},
}


def test_flatten_tracks_returns_dataframe():
    """Test flatten_tracks returns a DataFrame with expected columns."""
    df = particles.flatten_particle_features(tracking_outputs, detection_outputs)
    expected_cols = {
        "track_id",
        "frame",
        "timestamp",
        "label",
        "centroid_x",
        "centroid_y",
        "area",
        "mean_intensity",
        "min_intensity",
        "max_intensity",
    }
    assert isinstance(df, pd.DataFrame)
    assert expected_cols.issubset(df.columns)


def test_plot_tracks_3d_returns_axes():
    """Test plot_tracks_3d returns a matplotlib Axes object."""
    df = particles.flatten_particle_features(tracking_outputs, detection_outputs)
    ax = particles.plot_particle_labels_3d(df)
    assert isinstance(ax, plt.Axes)


def test_plot_tracks_3d_saves_file():
    """Test plot_tracks_3d saves a file if save_to is provided."""
    df = particles.flatten_particle_features(tracking_outputs, detection_outputs)
    with TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "plot.png"
        particles.plot_particle_labels_3d(df, save_to=out_path)
        assert out_path.exists()


def test_export_particle_csv_creates_file():
    """Test export_particle_csv writes a CSV file to disk."""
    df = particles.flatten_particle_features(tracking_outputs, detection_outputs)
    with TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "tracks.csv"
        particles.export_particle_csv(df, out_path)
        assert out_path.exists()
        loaded = pd.read_csv(out_path)
        assert not loaded.empty


# --- Frame Utils ---

# Mock input data
mock_features_per_frame = [
    [{"area": 10, "mean": 1.0}, {"area": 20, "mean": 2.0}],
    [{"area": 15, "mean": 1.5}],
    [],
]


def test_frame_summary_to_dataframe_structure():
    """frame_summary_to_dataframe returns expected DataFrame structure."""
    df = frames.frame_summary_to_dataframe(mock_features_per_frame)
    expected_cols = {
        "frame_index",
        "n_features",
        "total_area",
        "mean_area",
        "mean_intensity",
    }
    assert isinstance(df, pd.DataFrame)
    assert expected_cols.issubset(df.columns)
    assert len(df) == 3


def test_plot_frame_histogram_returns_axes():
    """plot_frame_histogram returns a matplotlib Axes object."""
    df = frames.frame_summary_to_dataframe(mock_features_per_frame)
    ax = frames.plot_frame_histogram(df, column="n_features")
    assert isinstance(ax, plt.Axes)


def test_plot_frame_histogram_saves_file():
    """plot_frame_histogram saves a file if save_to is provided."""
    df = frames.frame_summary_to_dataframe(mock_features_per_frame)
    with TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "hist.png"
        frames.plot_frame_histogram(df, column="n_features", save_to=out_path)
        assert out_path.exists()
