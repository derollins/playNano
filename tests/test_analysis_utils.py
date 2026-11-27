"""Tests for analysis utils."""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from playnano.afm_stack import AFMImageStack
from playnano.analysis.pipeline import AnalysisPipeline
from playnano.analysis.utils import common, frames, particles
from playnano.analysis.utils.common import (
    NumpyEncoder,
    load_analysis_from_hdf5,
    safe_json_dumps,
)

matplotlib.use("Agg")  # Use a non-interactive backend suitable for testing

# --- Common Utils ---


def create_hdf5_file(structure, dataset_name="analysis_record"):
    """Make a hdf5 file for testing."""
    temp_file = NamedTemporaryFile(delete=False, suffix=".h5")
    with h5py.File(temp_file.name, "w") as h5file:
        group = h5file.create_group(dataset_name)

        def recurse_write(g, obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    recurse_write(g.create_group(k), v)
            elif isinstance(obj, list):
                if len(obj) == 0:
                    g.create_group("empty")
                else:
                    for i, item in enumerate(obj):
                        recurse_write(g.create_group(f"item_{i}"), item)
            elif isinstance(obj, np.ndarray):
                g.create_dataset("values", data=obj)
            elif isinstance(obj, (int, float, str)):
                g.attrs["value"] = obj
            else:
                g.attrs["value"] = str(obj)

        recurse_write(group, structure)
    return temp_file.name


def test_load_valid_structure():
    """Test loading a nested structure with arrays, lists, dicts and strings."""
    data = {
        "a": np.array([1.0, 2.0]),
        "b": [1.0, 2.0],
        "c": {"d": 3.0},
        "e": [],
        "f": "text",
    }
    file_path = create_hdf5_file(data)
    result = load_analysis_from_hdf5(file_path)
    assert result["a"].tolist() == [1, 2]
    assert result["b"] == [1, 2]
    assert result["c"]["d"] == 3
    assert result["e"] == []
    assert result["f"] == "text"


def test_missing_dataset():
    """Test that a KeyError is raised when the specified dataset is missing."""
    file_path = create_hdf5_file({}, dataset_name="other_record")
    with pytest.raises(KeyError, match="Dataset 'analysis_record' not found"):
        load_analysis_from_hdf5(file_path, dataset_name="analysis_record")


def test_scalar_float_conversion():
    """Test that scalar NumPy float is converted to int if it's integer-valued."""
    data = {"x": np.array(5.0)}
    file_path = create_hdf5_file(data)
    result = load_analysis_from_hdf5(file_path)
    assert result["x"] == 5


def test_scalar_array_conversion():
    """Test that scalar array with a single float value is converted to int."""
    file_path = create_hdf5_file({"scalar": np.array(5.0)})
    result = load_analysis_from_hdf5(file_path)
    assert result["scalar"] == 5


def test_full_array_conversion():
    """Test that a full NumPy float array with int values is converted to int array."""
    file_path = create_hdf5_file({"array": np.array([1.0, 2.0, 3.0])})
    result = load_analysis_from_hdf5(file_path)
    assert isinstance(result["array"], np.ndarray)
    assert result["array"].tolist() == [1, 2, 3]


def test_string_array_conversion():
    """Test that a NumPy byte string array is converted to a list of Python strings."""
    file_path = create_hdf5_file({"strings": np.array([b"foo", b"bar"])})
    result = load_analysis_from_hdf5(file_path)
    assert result["strings"].tolist() == ["foo", "bar"]


def test_empty_list_handling():
    """Test that an empty list is correctly reconstructed from the HDF5 group."""
    file_path = create_hdf5_file({"empty_list": []})
    result = load_analysis_from_hdf5(file_path)
    assert result["empty_list"] == []


def test_value_attribute_handling():
    """Test that primitive values stored in attributes are loaded and converted."""
    file_path = create_hdf5_file({"value": 42.0})
    result = load_analysis_from_hdf5(file_path)
    assert result["value"] == 42


def test_list_structure_handling():
    """Test that a list-like group with item_* keys reconstructs as a Python list."""
    file_path = create_hdf5_file({"mylist": [1.0, 2.0, 3.0]})
    result = load_analysis_from_hdf5(file_path)
    assert result["mylist"] == [1, 2, 3]


def test_dict_structure_handling():
    """Test that a dict-like group is reconstructed as a Python dictionary."""
    file_path = create_hdf5_file({"mydict": {"a": 1.0, "b": 2.0}})
    result = load_analysis_from_hdf5(file_path)
    assert result["mydict"] == {"a": 1, "b": 2}


def test_numpy_encoder_serializes_ndarray():
    """Test that numpy encoder serializes a numpy array."""
    data = {"arr": np.array([1, 2, 3])}
    json_str = json.dumps(data, cls=common.NumpyEncoder)
    assert json_str == '{"arr": [1, 2, 3]}'


def test_numpy_encoder_raises_for_unserializable():
    """Test that numpy encoder raises error for unserializable."""

    class Dummy:
        pass

    data = {"obj": Dummy()}
    with pytest.raises(TypeError):
        json.dumps(data, cls=common.NumpyEncoder)


def test_numpy_encoder_callable_serialization():
    """Test that callables are serialized to a string with their function name."""

    def dummy_function():
        pass

    data = {"func": dummy_function}
    encoded = json.dumps(data, cls=common.NumpyEncoder)

    assert '"<function dummy_function>"' in encoded


# Sample nested record
sample_record = {
    "metadata": {"experiment": "test", "version": 1.0},
    "results": {"values": [1, 2, 3], "array": np.array([4.5, 5.5])},
}


def test_export_to_hdf5_creates_file():
    """Test export_to_hdf5 creates an HDF5 file on disk."""
    with TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test.h5"
        common.export_to_hdf5(sample_record, out_path)
        assert out_path.exists()


def test_export_to_hdf5_structure_and_values():
    """Test export_to_hdf5 writes correct structure and values."""
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

            values = [
                json.loads(v) if isinstance(v, (str, bytes)) else v
                for v in values_ds[:]
            ]
            assert values == sample_record["results"]["values"]


def test_safe_json_dumps_serializable():
    """Test that safe_json_dumps serializes a simple object."""
    obj = {"value": np.float32(3.14), "array": np.array([1, 2, 3])}
    result = safe_json_dumps(obj)
    parsed = json.loads(result)  # Should succeed without error
    assert parsed["value"] == pytest.approx(3.14)
    assert parsed["array"] == [1, 2, 3]


def test_safe_json_dumps_fallback(monkeypatch):
    """Test that safe_json_dumps falls back to str() for unserializable objects."""

    # Force the encoder to fail and test fallback to str()
    def failing_default(self, obj):
        raise TypeError("mock failure")

    monkeypatch.setattr(NumpyEncoder, "default", failing_default)
    obj = {"value": 123}
    result = safe_json_dumps(obj)
    assert isinstance(result, str)
    assert "value" in result


def test_safe_json_dumps_non_serializable():
    """Test that safe_json_dumps falls back to str() for non-serializable objects."""
    obj = {"callback": lambda x: x}
    result = safe_json_dumps(obj)
    # Falls back to str()
    assert isinstance(result, str)
    assert "function" in result or "lambda" in result


def test_safe_json_dumps_fallback_on_array():
    """Test that safe_json_dumps falls back to str() for numpy arrays."""
    obj = {"array": np.array([1, 2, 3])}
    result = safe_json_dumps(obj)
    # Not valid JSON; fallback used
    assert isinstance(result, str)
    assert "array" in result


# --- Frame Utils ---

# Mock input data
tracking_outputs = {
    "tracks": [
        {
            "id": 1,
            "frames": [0, 1],
            "point_indices": [0, 0],
            "centroids": [(5, 5), (6, 6)],
            "labels": [10, 11],
        }
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


def test_flatten_particle_features_autodetect_track_id():
    """Autodetects track ID when 'tracks' key is present."""
    grouping = {"tracks": [{"id": 1, "frames": [0], "point_indices": [0]}]}
    detection = {"features_per_frame": [[{"centroid": (1, 1)}]]}
    df = particles.flatten_particle_features(grouping, detection)
    assert "track_id" in df.columns
    assert df.loc[0, "track_id"] == 1


def test_flatten_particle_features_autodetect_cluster_id():
    """Autodetects cluster ID when 'clusters' key is present."""
    grouping = {"clusters": [{"id": 7, "frames": [0], "point_indices": [0]}]}
    detection = {"features_per_frame": [[{"centroid": (1, 1)}]]}
    df = particles.flatten_particle_features(grouping, detection)
    assert "cluster_id" in df.columns
    assert df.loc[0, "cluster_id"] == 7


def test_flatten_particle_features_raises_on_unknown_key():
    """Raises ValueError if object key is not auto-detectable."""
    grouping = {"nonsense": [{"id": 1, "frames": [0], "point_indices": [0]}]}
    detection = {"features_per_frame": [[]]}
    with pytest.raises(ValueError, match="Unable to autodetect object_key"):
        particles.flatten_particle_features(grouping, detection)


def test_flatten_particle_features_raises_on_missing_keys():
    """Raises KeyError if 'frames' or 'point_indices' keys are missing."""
    grouping = {"tracks": [{"id": 1, "frames": [0]}]}  # Missing point_indices
    detection = {"features_per_frame": [[]]}
    with pytest.raises(KeyError, match="point_indices"):
        particles.flatten_particle_features(grouping, detection)


def test_flatten_particle_features_skips_out_of_bounds_frame():
    """Skips features if frame index is out of bounds."""
    grouping = {"tracks": [{"id": 1, "frames": [10], "point_indices": [0]}]}
    detection = {"features_per_frame": [[]]}  # Only 1 frame
    df = particles.flatten_particle_features(grouping, detection)
    assert df.empty


def test_flatten_particle_features_skips_out_of_bounds_point():
    """Skips features if point index is out of bounds."""
    grouping = {"tracks": [{"id": 1, "frames": [0], "point_indices": [99]}]}
    detection = {"features_per_frame": [[{"centroid": (1, 1)}]]}
    df = particles.flatten_particle_features(grouping, detection)
    assert df.empty


def test_flatten_tracks_returns_dataframe():
    """Test flatten_tracks returns a DataFrame with expected columns."""
    df = particles.flatten_particle_features(
        tracking_outputs,
        detection_outputs,
        object_key="tracks",
        object_id_field="track_id",
    )
    expected_cols = {
        "track_id",  # if using object_id_field="track_id"
        "frame",
        "timestamp",
        "label",  # still included from `feat.get("label", idx)`
        "centroid_x",
        "centroid_y",
        "area",
        "mean",
        "min",
        "max",
    }
    assert isinstance(df, pd.DataFrame)
    assert expected_cols.issubset(df.columns)


def _make_detection_output(features_per_frame):
    """
    Create a detection_output dict like FeatureDetectionModule.run() would return.

    features_per_frame: list of list[dict] per frame
    """
    return {"features_per_frame": features_per_frame}


def test_centroid_mapping_is_yx_to_xy():
    """
    The detection module stores centroid as (row, col) = (y, x).

    The flattener must map centroid_x <- centroid[1], centroid_y <- centroid[0].
    """
    features_per_frame = [
        [
            {  # frame 0, feature 0
                "frame_timestamp": 0.0,
                "label": 1,
                "centroid": (10.5, 20.25),  # (y, x)
                "area": 100,
                "mean": 1.2,
                "min": 0.9,
                "max": 1.8,
            },
            {  # frame 0, feature 1
                "frame_timestamp": 0.0,
                "label": 2,
                "centroid": (30.0, 40.0),  # (y, x)
                "area": 50,
                "mean": 0.6,
                "min": 0.4,
                "max": 0.9,
            },
        ]
    ]
    detection_output = _make_detection_output(features_per_frame)

    # grouping: one cluster that references frame 0, feature indices 0 and 1
    grouping_output = {
        "clusters": [
            {
                "id": 7,
                "frames": [0, 0],
                "point_indices": [0, 1],
            }
        ]
    }

    df = particles.flatten_particle_features(grouping_output, detection_output)

    # Expected: two rows
    assert len(df) == 2

    # Check mapping for row 0
    r0 = df.iloc[0]
    assert r0["frame"] == 0
    assert r0["label"] == 1
    assert r0["centroid_y"] == pytest.approx(10.5)  # from centroid[0]
    assert r0["centroid_x"] == pytest.approx(20.25)  # from centroid[1]
    assert r0["area"] == 100
    assert r0["mean"] == pytest.approx(1.2)
    assert r0["min"] == pytest.approx(0.9)
    assert r0["max"] == pytest.approx(1.8)

    # And for row 1
    r1 = df.iloc[1]
    assert r1["centroid_y"] == pytest.approx(30.0)
    assert r1["centroid_x"] == pytest.approx(40.0)


def test_autodetect_tracks_vs_clusters_and_id_field_tracks():
    """
    Test if grouping_output is correctly read and object_id_field set for tracks.

    If grouping_output contains 'tracks', use object_key='tracks'
    and object_id_field='track_id'.
    """
    features_per_frame = [
        [
            {
                "frame_timestamp": 0.0,
                "label": 1,
                "centroid": (5.0, 15.0),
                "area": 12,
                "mean": 0.5,
                "min": 0.2,
                "max": 0.9,
            }
        ],
        [],
    ]
    detection_output = _make_detection_output(features_per_frame)

    grouping_output = {
        "tracks": [
            {
                "id": 3,  # should map to 'track_id'
                "frames": [0],
                "point_indices": [0],
            }
        ]
    }

    df = particles.flatten_particle_features(grouping_output, detection_output)

    assert "track_id" in df.columns
    assert "cluster_id" not in df.columns
    assert df.iloc[0]["track_id"] == 3
    assert df.iloc[0]["centroid_x"] == pytest.approx(15.0)
    assert df.iloc[0]["centroid_y"] == pytest.approx(5.0)


def test_autodetect_clusters_and_id_field_clusters():
    """
    Test if grouping_output is correctly read and object_id_field set.

    If grouping_output contains 'clusters', use object_key='clusters'
    and object_id_field='cluster_id'.
    """
    detection_output = _make_detection_output(
        [
            [
                {
                    "frame_timestamp": 0.0,
                    "label": 1,
                    "centroid": (1.0, 2.0),
                    "area": 5,
                    "mean": 0.3,
                    "min": 0.1,
                    "max": 0.5,
                }
            ]
        ]
    )
    grouping_output = {"clusters": [{"id": 99, "frames": [0], "point_indices": [0]}]}

    df = particles.flatten_particle_features(grouping_output, detection_output)
    assert "cluster_id" in df.columns
    assert df.iloc[0]["cluster_id"] == 99
    assert df.iloc[0]["centroid_x"] == pytest.approx(2.0)
    assert df.iloc[0]["centroid_y"] == pytest.approx(1.0)


def test_skips_out_of_range_indices_cleanly():
    """If (frame, index) points outside available features, row is skipped."""
    features_per_frame = [
        [],  # frame 0 has no features
        [
            {
                "frame_timestamp": 1.0,
                "label": 1,
                "centroid": (3.0, 4.0),
                "area": 10,
                "mean": 0.5,
                "min": 0.4,
                "max": 0.7,
            }
        ],
    ]
    detection_output = _make_detection_output(features_per_frame)

    grouping_output = {
        "clusters": [
            {
                "id": 1,
                "frames": [0, 1, 2],  # frame 2 is out-of-range
                "point_indices": [0, 0, 0],  # idx 0 at frame 0 is invalid (no features)
            }
        ]
    }

    df = particles.flatten_particle_features(grouping_output, detection_output)
    # Only the valid (frame=1, idx=0) should be present
    assert len(df) == 1
    r = df.iloc[0]
    assert r["frame"] == 1
    assert r["centroid_x"] == pytest.approx(4.0)
    assert r["centroid_y"] == pytest.approx(3.0)


def test_required_columns_present_and_types_reasonable():
    """Test that the required column are present in output dataframe with types."""
    features_per_frame = [
        [
            {
                "frame_timestamp": 0.0,
                "label": 7,
                "centroid": (12.3, 45.6),
                "area": 111.0,
                "mean": 2.3,
                "min": 1.9,
                "max": 3.1,
            }
        ]
    ]
    detection_output = _make_detection_output(features_per_frame)
    grouping_output = {"clusters": [{"id": 1, "frames": [0], "point_indices": [0]}]}

    df = particles.flatten_particle_features(grouping_output, detection_output)

    expected_cols = {
        "cluster_id",
        "frame",
        "timestamp",
        "label",
        "centroid_x",
        "centroid_y",
        "area",
        "mean",
        "min",
        "max",
    }

    assert expected_cols.issubset(df.columns)

    r = df.iloc[0]
    assert isinstance(
        r["cluster_id"], (int, np.integer, float)
    )  # there is some weird upcastign to loats if no values found in pandas.
    assert isinstance(
        r["frame"], (int, np.integer, float)
    )  # there is some weird upcastign to loats if no values found in pandas.
    assert isinstance(r["timestamp"], float)
    assert isinstance(
        r["label"], (int, np.integer, float)
    )  # there is some weird upcastign to loats if no values found in pandas.
    assert isinstance(r["centroid_x"], float)
    assert isinstance(r["centroid_y"], float)
    assert isinstance(r["area"], (float))
    assert isinstance(r["mean"], float)
    assert isinstance(r["min"], float)
    assert isinstance(r["max"], float)


def test_explicit_object_key_and_custom_id_field():
    """Test override auto-detection & column name with object_key + object_id_field."""
    features_per_frame = [
        [
            {
                "frame_timestamp": 0.0,
                "label": 1,
                "centroid": (9.0, 8.0),
                "area": 12,
                "mean": 0.1,
                "min": 0.0,
                "max": 0.2,
            }
        ]
    ]
    detection_output = _make_detection_output(features_per_frame)
    grouping_output = {"clusters": [{"id": 5, "frames": [0], "point_indices": [0]}]}

    df = particles.flatten_particle_features(
        grouping_output,
        detection_output,
        object_key="clusters",
        object_id_field="object_id",
        frame_key="frames",
        index_key="point_indices",
    )

    assert "object_id" in df.columns and "cluster_id" not in df.columns
    assert df.iloc[0]["object_id"] == 5
    assert df.iloc[0]["centroid_x"] == pytest.approx(8.0)
    assert df.iloc[0]["centroid_y"] == pytest.approx(9.0)


def test_raises_when_cannot_autodetect_object_key():
    """Test that ValueError is raised when object key not detected."""
    detection_output = _make_detection_output([])
    grouping_output = {"unknown": []}  # neither tracks nor clusters

    with pytest.raises(ValueError):
        particles.flatten_particle_features(grouping_output, detection_output)


def test_plot_tracks_3d_returns_axes():
    """Test plot_tracks_3d returns a matplotlib Axes object."""
    df = particles.flatten_particle_features(
        tracking_outputs,
        detection_outputs,
        object_key="tracks",
        object_id_field="track_id",
    )
    ax = particles.plot_particle_labels_3d(df)
    assert hasattr(ax, "plot")


def test_plot_tracks_3d_saves_file():
    """Test plot_tracks_3d saves a file if save_to is provided."""
    df = particles.flatten_particle_features(
        tracking_outputs,
        detection_outputs,
        object_key="tracks",
        object_id_field="track_id",
    )
    with TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "plot.png"
        particles.plot_particle_labels_3d(df, save_to=out_path)
        assert out_path.exists()


def test_export_particle_csv_creates_file():
    """Test export_particle_csv writes a CSV file to disk."""
    df = particles.flatten_particle_features(
        tracking_outputs,
        detection_outputs,
        object_key="tracks",
        object_id_field="track_id",
    )
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
    """Test frame_summary_to_dataframe returns expected DataFrame structure."""
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
    """Test plot_frame_histogram returns a matplotlib Axes object."""
    df = frames.frame_summary_to_dataframe(mock_features_per_frame)
    ax = frames.plot_frame_histogram(df, column="n_features")
    assert isinstance(ax, plt.Axes)


def test_plot_frame_histogram_saves_file():
    """Test plot_frame_histogram saves a file if save_to is provided."""
    df = frames.frame_summary_to_dataframe(mock_features_per_frame)
    with TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "hist.png"
        frames.plot_frame_histogram(df, column="n_features", save_to=out_path)
        assert out_path.exists()


def _is_plain(value, kind):
    """Check for a built-in int or float, not a NumPy scalar or a bool."""
    return isinstance(value, kind) and not isinstance(value, (bool, np.generic))


def test_hdf5_round_trip_is_exact(tmp_path):
    """Values come back from export_to_hdf5 as they were saved."""
    record = {
        "none": None,
        "empty_dict": {},
        "empty_list": [],
        "floats": [0.0, 1.0],
        "ints": [1, 2],
        "track_with_gap": [0, None, 2],
        "strings": ["a", "b"],
        "nested": {"values": 1, "value": "x", "empty": [3.5]},
        "float_array": np.array([[1.5, 2.0], [3.0, 4.0]]),
        "int_array": np.arange(3),
        "text": "hi",
        "flag": True,
        "count": 3,
        "whole_float": 2.0,
        "pair": (1, 2),
    }
    out = tmp_path / "record.h5"
    common.export_to_hdf5(record, out)
    loaded = load_analysis_from_hdf5(out)

    assert set(loaded) == set(record)
    assert loaded["none"] is None
    assert loaded["empty_dict"] == {}
    assert loaded["empty_list"] == []
    assert loaded["floats"] == [0.0, 1.0]
    assert all(_is_plain(v, float) for v in loaded["floats"])
    assert loaded["ints"] == [1, 2]
    assert all(_is_plain(v, int) for v in loaded["ints"])
    assert loaded["track_with_gap"] == [0, None, 2]
    assert loaded["strings"] == ["a", "b"]
    assert loaded["nested"] == {"values": 1, "value": "x", "empty": [3.5]}
    np.testing.assert_array_equal(loaded["float_array"], record["float_array"])
    assert loaded["float_array"].dtype == record["float_array"].dtype
    np.testing.assert_array_equal(loaded["int_array"], record["int_array"])
    assert loaded["text"] == "hi"
    assert loaded["flag"] is True
    assert loaded["count"] == 3 and _is_plain(loaded["count"], int)
    assert loaded["whole_float"] == 2.0
    assert _is_plain(loaded["whole_float"], float)
    assert loaded["pair"] == [1, 2]


def test_hdf5_round_trip_track_with_gap_flattens(tmp_path):
    """A reloaded particle track with a missing frame still flattens."""
    from playnano.analysis.utils.particles import flatten_particle_features

    record = {
        "tracks": {
            "tracks": [{"id": 0, "frames": [0, 1, 2], "point_indices": [0, None, 0]}]
        },
        "detection": {
            "features_per_frame": [
                [{"area": 1.0, "centroid": (1.0, 1.0)}],
                [],
                [{"area": 2.0, "centroid": (2.0, 2.0)}],
            ]
        },
    }
    out = tmp_path / "record.h5"
    common.export_to_hdf5(record, out)
    loaded = load_analysis_from_hdf5(out)

    df = flatten_particle_features(
        loaded["tracks"], loaded["detection"], object_key="tracks"
    )
    assert df["frame"].tolist() == [0, 2]


def test_sanitizers_convert_numpy_dict_keys():
    """Convert NumPy dict keys so the result can be written to JSON."""
    from playnano.utils.io_utils import make_json_safe as io_make_json_safe

    data = {np.int64(3): 1, np.float64(0.5): 2, (1, 2): 3}
    for cleaned in (
        common.sanitize_analysis_for_logging(data),
        io_make_json_safe(data),
    ):
        assert set(cleaned) == {3, 0.5, "(1, 2)"}
        json.dumps(cleaned)


def test_sanitizer_keeps_deeply_nested_values(caplog):
    """Numbers and strings nested deeper than six levels are not replaced."""
    deep = [1.5, "x"]
    for i in range(10):
        deep = {f"level_{i}": deep}
    cleaned = common.sanitize_analysis_for_logging(deep)

    text = json.dumps(cleaned)
    assert "truncated" not in text
    assert '[1.5, "x"]' in text
    assert "nested more than" not in caplog.text


def test_sanitizer_warns_when_limit_is_hit(caplog):
    """Containers beyond the depth limit are replaced, with a warning."""
    cleaned = common.sanitize_analysis_for_logging(
        {"a": {"b": [1, 2]}, "n": 3}, _max_depth=1
    )
    assert cleaned["a"]["b"].startswith("<... truncated depth 2")
    assert cleaned["n"] == 3
    assert "nested more than 1 levels deep" in caplog.text


def test_analysis_json_has_no_truncation_markers():
    """A real detection, tracking and region extraction run is written in full."""
    sample = Path(__file__).parent / "resources" / "sample_0.h5-jpk"
    stack = AFMImageStack.load_data(sample)
    pipeline = AnalysisPipeline()
    pipeline.add("feature_detection", mask_fn="mask_mean_offset", min_size=5)
    pipeline.add("particle_tracking", max_distance=10.0)
    pipeline.add("particle_region_extraction")
    record = pipeline.run(stack)

    text = json.dumps(common.make_json_safe(record))
    assert "truncated" not in text
    assert "bbox_tight" in text


def test_numpy_encoder_default_converts_numpy_bools():
    """NumpyEncoder.default turns NumPy bools into Python bools."""
    encoder = NumpyEncoder()
    assert encoder.default(np.bool_(True)) is True
    assert encoder.default(np.bool_(False)) is False


def test_export_analysis_to_json_writes_numpy_bools(tmp_path):
    """An analysis record holding NumPy bools exports as JSON true/false."""
    from playnano.analysis.export import export_analysis_to_json

    # Without NumpyEncoder, json cannot write NumPy bools at all.
    with pytest.raises(TypeError):
        json.dumps(np.bool_(True))

    out = tmp_path / "record.json"
    export_analysis_to_json(
        str(out), {"flags": {"a": np.bool_(True), "b": np.bool_(False)}}
    )
    assert json.loads(out.read_text()) == {"flags": {"a": True, "b": False}}


@pytest.mark.parametrize(
    "arr",
    [np.array(["a", "b"]), np.array([b"x"]), np.array([{"k": 1}], dtype=object)],
)
def test_sanitize_summarises_non_numeric_arrays_without_summary(arr):
    """Describe text and object arrays by shape and dtype, with no summary."""
    out = common.sanitize_analysis_for_logging({"v": arr})["v"]
    assert out["_array_shape"] == arr.shape
    assert out["_array_dtype"] == str(arr.dtype)
    assert out["_summary"] is None
    json.dumps(out)  # JSON-safe


def test_sanitize_still_summarises_numeric_and_bool_arrays():
    """Numeric and bool arrays keep their min, max and mean."""
    num = common.sanitize_analysis_for_logging(np.array([1.0, 3.0]))
    assert num["_summary"] == {"min": 1.0, "max": 3.0, "mean": 2.0}
    flags = common.sanitize_analysis_for_logging(np.array([True, False]))
    assert flags["_summary"] == {"min": 0.0, "max": 1.0, "mean": 0.5}
