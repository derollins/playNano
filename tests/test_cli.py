"""Tests for the playNano CLI."""

import argparse
import builtins
import inspect
import json
import logging
import tempfile
from argparse import Namespace
from collections import UserDict
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Union
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest
import yaml

import playNano.cli.actions as actions
from playNano.afm_stack import AFMImageStack
from playNano.cli import utils as cli_utils
from playNano.cli.actions import IO, Wizard, analyze_pipeline_mode
from playNano.cli.entrypoint import setup_logging
from playNano.cli.handlers import handle_analyze, handle_play, handle_wizard
from playNano.cli.utils import (
    FILTER_MAP,
    MASK_MAP,
    _ask_with_spec,
    _get_analysis_class,
    _get_processing_callable,
    _normalize_loaded,
    _process_pending_entries,
    _resolve_condition,
    _sanitize_for_dump,
    ask_for_processing_params,
    get_processing_step_type,
    is_valid_step,
    parse_analysis_file,
    parse_analysis_string,
    parse_processing_file,
    parse_processing_string,
)
from playNano.errors import LoadError
from playNano.processing.filters import register_filters
from playNano.processing.mask_generators import register_masking
from playNano.processing.masked_filters import register_mask_filters

register_filters()
register_masking()


@patch("playNano.cli.actions.process_stack", side_effect=Exception("boom"))
def test_process_pipeline_mode_load_error_logs_and_returns(mock_process, caplog):
    """Test that a processing failure logs an error and exits."""
    caplog.set_level(logging.ERROR)
    with pytest.raises(SystemExit) as exc:
        actions.process_pipeline_mode(
            input_file="in.jpk",
            channel="chan",
            processing_str="",
            processing_file=None,
            export=None,
            make_gif=False,
            output_folder=None,
            output_name=None,
            scale_bar_nm=None,
        )
    assert exc.value.code == 1
    assert any("Failed to process AFM stack" in rec.message for rec in caplog.records)


@patch(
    "playNano.cli.actions.parse_processing_string",
    return_value=[("f1", {}), ("f2", {"a": 1})],
)
@patch("playNano.cli.actions.process_stack")
@patch("playNano.cli.actions.export_bundles")
@patch("playNano.cli.actions.export_gif")
def test_process_pipeline_mode_flow(mock_gif, mock_bundles, mock_proc, mock_parse):
    """Test the full flow of process_pipeline_mode with processing string."""
    pipe = MagicMock()
    mock_proc.return_value = pipe

    actions.process_pipeline_mode(
        "in.jpk", "ch", "f1;f2:a=1", None, "npz,h5", True, "od", "nm", 10
    )
    mock_parse.assert_called_once()
    mock_proc.assert_called_once_with(
        Path("in.jpk"), "ch", [("f1", {}), ("f2", {"a": 1})]
    )
    mock_bundles.assert_called_once_with(pipe, "od", "nm", ["npz", "h5"])


@pytest.fixture
def mock_pipeline(monkeypatch):
    """
    Fixture to mock the AnalysisPipeline class and its run method.

    Returns a MagicMock pipeline instance.
    """
    pipeline = MagicMock()
    pipeline.run.return_value = {"analysis": "result"}
    monkeypatch.setattr("playNano.cli.actions.AnalysisPipeline", lambda: pipeline)
    return pipeline


def test_analyze_pipeline_basic_flow(tmp_path, monkeypatch, mock_pipeline):
    """
    Test that analyze_pipeline_mode performs the full pipeline flow correctly.

    Tests on an inline analysis string (no file).

    Checks:
    - AFMImageStack.load_data called with input and channel.
    - warn_if_unprocessed called on loaded stack.
    - parse_analysis_string called with provided analysis string.
    - Pipeline steps added and run called properly.
    - JSON file opened and written.
    - HDF5 export called with expected arguments.
    """
    input_file = "input.afm"
    channel = "height_trace"
    analysis_str = "step1:param=1"
    analysis_file = None
    output_folder = str(tmp_path)
    output_name = None

    # Mock dependencies
    mock_load_data = MagicMock(return_value="stack")
    monkeypatch.setattr("playNano.cli.actions.AFMImageStack.load_data", mock_load_data)

    mock_warn = MagicMock()
    monkeypatch.setattr("playNano.cli.actions.warn_if_unprocessed", mock_warn)

    mock_parse_analysis_string = MagicMock(return_value=[("step1", {"param": 1})])
    monkeypatch.setattr(
        "playNano.cli.actions.parse_analysis_string", mock_parse_analysis_string
    )

    mock_make_json_safe = MagicMock(side_effect=lambda x: x)
    monkeypatch.setattr("playNano.cli.actions.make_json_safe", mock_make_json_safe)

    mock_export = MagicMock()
    monkeypatch.setattr("playNano.cli.actions.export_to_hdf5", mock_export)

    # Mock builtins.open for JSON writing
    m_open = mock_open()
    monkeypatch.setattr(Path, "open", m_open)

    analyze_pipeline_mode(
        input_file, channel, analysis_str, analysis_file, output_folder, output_name
    )

    # Now you can check that Path.open was called with "w"
    m_open.assert_called_with("w")
    # Assertions
    mock_load_data.assert_called_once_with(input_file, channel=channel)
    mock_warn.assert_called_once_with("stack")
    mock_parse_analysis_string.assert_called_once_with(analysis_str)
    mock_pipeline.add.assert_called_with("step1", param=1)
    mock_pipeline.run.assert_called_once_with("stack", log_to=None)

    handle = m_open()
    handle.write.assert_called()

    expected_h5_path = Path(output_folder) / (Path(input_file).stem + ".h5")
    mock_export.assert_called_once_with(
        mock_pipeline.run.return_value, out_path=expected_h5_path
    )


def test_analyze_pipeline_prefers_file_over_str(tmp_path, monkeypatch, mock_pipeline):
    """
    Test that analyze_pipeline_mode prefers parsing analysis steps from a file.

    Checks that if a file is providedif provided, it ignores the analysis string.

    Checks:
    - parse_analysis_file called with given file.
    - parse_analysis_string not called.
    - Pipeline steps added and HDF5 export called.
    - AFMImageStack.load_data called.
    """
    input_file = "input.afm"
    channel = "height_trace"
    analysis_str = "step1:param=1"
    analysis_file = "analysis.yaml"
    output_folder = str(tmp_path)
    output_name = "customname"

    mock_load_data = MagicMock(return_value="stack")
    monkeypatch.setattr("playNano.cli.actions.AFMImageStack.load_data", mock_load_data)

    monkeypatch.setattr("playNano.cli.actions.warn_if_unprocessed", MagicMock())

    mock_parse_file = MagicMock(return_value=[("stepfile", {})])
    monkeypatch.setattr("playNano.cli.actions.parse_analysis_file", mock_parse_file)

    mock_parse_str = MagicMock()
    monkeypatch.setattr("playNano.cli.actions.parse_analysis_string", mock_parse_str)

    mock_make_json_safe = MagicMock(side_effect=lambda x: x)
    monkeypatch.setattr("playNano.cli.actions.make_json_safe", mock_make_json_safe)

    mock_export = MagicMock()
    monkeypatch.setattr("playNano.cli.actions.export_to_hdf5", mock_export)

    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: MagicMock())

    analyze_pipeline_mode(
        input_file, channel, analysis_str, analysis_file, output_folder, output_name
    )

    mock_parse_file.assert_called_once_with(analysis_file)
    mock_parse_str.assert_not_called()
    mock_pipeline.add.assert_called_with("stepfile")
    mock_export.assert_called()
    mock_load_data.assert_called_once_with(input_file, channel=channel)


def test_analyze_pipeline_creates_output_folder(monkeypatch, tmp_path, mock_pipeline):
    """
    Test that analyze_pipeline_mode creates the output folder if it does not exist.

    Checks:
    - Output folder directory is created.
    - AFMImageStack.load_data called.
    """
    input_file = "input.afm"
    channel = "chan"
    analysis_str = "step1"
    analysis_file = None
    output_folder = str(tmp_path / "newfolder")  # folder does not exist yet
    output_name = None

    mock_load_data = MagicMock(return_value="stack")
    monkeypatch.setattr("playNano.cli.actions.AFMImageStack.load_data", mock_load_data)

    monkeypatch.setattr("playNano.cli.actions.warn_if_unprocessed", MagicMock())

    mock_parse_analysis_string = MagicMock(return_value=[("step1", {})])
    monkeypatch.setattr(
        "playNano.cli.actions.parse_analysis_string", mock_parse_analysis_string
    )

    mock_make_json_safe = MagicMock(side_effect=lambda x: x)
    monkeypatch.setattr("playNano.cli.actions.make_json_safe", mock_make_json_safe)

    mock_export = MagicMock()
    monkeypatch.setattr("playNano.cli.actions.export_to_hdf5", mock_export)

    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: MagicMock())

    analyze_pipeline_mode(
        input_file, channel, analysis_str, analysis_file, output_folder, output_name
    )

    assert Path(output_folder).exists()
    mock_load_data.assert_called_once_with(input_file, channel=channel)


@patch("playNano.cli.actions.AFMImageStack.load_data", side_effect=Exception("err"))
def test_play_pipeline_mode_load_error_exits(mock_load, caplog):
    """Test that play_pipeline_mode raises LoadError on loading failure."""
    caplog.set_level(logging.ERROR)
    with pytest.raises(LoadError) as exc:
        actions.play_pipeline_mode(
            "in.jpk",
            "ch",
            None,
            None,
            None,
            False,
            None,
        )
    assert "Failed to load in.jpk" in str(exc.value)


@patch("playNano.cli.actions.gui_entry")
@patch("playNano.cli.actions.AFMImageStack.load_data")
def test_play_pipeline_mode_with_valid_zmin_zmax(
    mock_load_data, mock_gui_entry, tmp_path
):
    """Test that play_pipeline_mode correctly handles valid zmin and zmax."""
    mock_stack = MagicMock()
    mock_stack.frame_metadata = [{"line_rate": 512}]
    mock_stack.image_shape = (512, 512)
    mock_load_data.return_value = mock_stack

    actions.play_pipeline_mode(
        input_file="dummy.afm",
        channel="height_trace",
        processing_str=None,
        processing_file=None,
        output_folder=str(tmp_path),
        output_name="test_output",
        scale_bar_nm=100,
        zmin="0.0",
        zmax="1.0",
    )

    args, kwargs = mock_gui_entry.call_args
    assert kwargs["zmin"] == 0.0
    assert kwargs["zmax"] == 1.0


@patch("playNano.cli.actions.gui_entry")
@patch("playNano.cli.actions.AFMImageStack.load_data")
def test_play_pipeline_mode_with_invalid_zmin_logs_error(
    mock_load_data, mock_gui_entry, caplog, tmp_path
):
    """Test that play_pipeline_mode logs an error for invalid zmin."""
    mock_stack = MagicMock()
    mock_stack.frame_metadata = [{"line_rate": 256}]
    mock_stack.image_shape = (256, 256)
    mock_load_data.return_value = mock_stack

    with caplog.at_level("ERROR"):
        actions.play_pipeline_mode(
            input_file="dummy.afm",
            channel="height_trace",
            processing_str=None,
            processing_file=None,
            output_folder=str(tmp_path),
            output_name="test_output",
            scale_bar_nm=100,
            zmin="not_a_number",
            zmax="auto",
        )

    assert "zmin must be either a number or the string 'auto'" in caplog.text


@patch("playNano.cli.actions.gui_entry")
@patch("playNano.cli.actions.AFMImageStack.load_data")
def test_play_pipeline_mode_with_invalid_zmax_logs_error(
    mock_load_data, mock_gui_entry, caplog, tmp_path
):
    """Test that play_pipeline_mode logs an error for invalid zmax."""
    mock_stack = MagicMock()
    mock_stack.frame_metadata = [{"line_rate": 512}]
    mock_stack.image_shape = (512, 512)
    mock_load_data.return_value = mock_stack

    with caplog.at_level("ERROR"):
        actions.play_pipeline_mode(
            input_file="dummy.afm",
            channel="height_trace",
            processing_str=None,
            processing_file=None,
            output_folder=str(tmp_path),
            output_name="test_output",
            scale_bar_nm=100,
            zmin="auto",
            zmax="not_a_number",
        )

    assert "zmax must be either a number or the string 'auto'" in caplog.text


@patch("playNano.cli.actions.parse_processing_file")
@patch("playNano.cli.actions.gui_entry")
@patch("playNano.cli.actions.AFMImageStack.load_data")
def test_play_pipeline_mode_uses_processing_file(
    mock_load_data, mock_gui_entry, mock_parse_file
):
    """Test that play_pipeline_mode uses processing file correctly."""
    mock_stack = MagicMock()
    mock_stack.frame_metadata = [{"line_rate": 512}]
    mock_stack.image_shape = (512, 512)
    mock_load_data.return_value = mock_stack
    mock_parse_file.return_value = [("filter_name", {"param": 1})]

    actions.play_pipeline_mode(
        input_file="dummy.afm",
        channel="height_trace",
        processing_str=None,
        processing_file="filters.yaml",
        output_folder=None,
        output_name=None,
        scale_bar_nm=100,
        zmin="auto",
        zmax="auto",
    )

    mock_parse_file.assert_called_once_with("filters.yaml")


@patch("playNano.cli.actions.parse_processing_string")
@patch("playNano.cli.actions.gui_entry")
@patch("playNano.cli.actions.AFMImageStack.load_data")
def test_play_pipeline_mode_uses_processing_str(
    mock_load_data, mock_gui_entry, mock_parse_str
):
    """Test that a processed string is used in play mode."""
    mock_stack = MagicMock()
    mock_stack.frame_metadata = [{"line_rate": 512}]
    mock_stack.image_shape = (512, 512)
    mock_load_data.return_value = mock_stack
    mock_parse_str.return_value = [("filter_name", {"param": 1})]

    actions.play_pipeline_mode(
        input_file="dummy.afm",
        channel="height_trace",
        processing_str="gaussian_filter:sigma=2",
        processing_file=None,
        output_folder=None,
        output_name=None,
        scale_bar_nm=100,
        zmin="auto",
        zmax="auto",
    )

    mock_parse_str.assert_called_once_with("gaussian_filter:sigma=2")


def test_wizard_mode_file_not_found(monkeypatch, caplog):
    """Test that wizard mode raises FileNotFoundError for missing file."""
    caplog.set_level(logging.ERROR)
    monkeypatch.setattr(Path, "exists", lambda self: False)
    with pytest.raises(FileNotFoundError) as exc:
        actions.Wizard("nofile.jpk", "ch", None, None, None)
    assert str(exc.value) == "File not found: nofile.jpk"


# Fixture to prepare wizard environment
@pytest.fixture(autouse=True)
def setup_wizard_env(monkeypatch):
    """Set up the environment for wizard mode tests."""
    # Prevent side effects
    monkeypatch.setattr(actions, "export_bundles", lambda *a, **k: None)
    monkeypatch.setattr(actions, "export_gif", lambda *a, **k: None)
    # Always treat file as existing and load dummy stack
    monkeypatch.setattr(Path, "exists", lambda self: True)
    fake = SimpleNamespace(n_frames=3, image_shape=(4, 4))
    monkeypatch.setattr(AFMImageStack, "load_data", lambda p, channel: fake)


@patch("builtins.input", side_effect=EOFError)
def test_wizard_eof_exit(mock_input):
    """EOFError from input should exit cleanly with code 0."""  # noqa
    wiz = Wizard("in.jpk", "chan", None, None, None)
    with pytest.raises(SystemExit) as exc:
        wiz.run()
    assert exc.value.code == 0


# --- Help and listing ---


def test_wizard_help_prints_commands(capsys):
    """Help command should print available commands."""
    inputs = iter(["help", "quit"])
    monkey = pytest.MonkeyPatch()
    monkey.setattr(builtins, "input", lambda prompt="": next(inputs))

    wiz = actions.Wizard("in.jpk", "chan", None, None, None)
    with pytest.raises(SystemExit):  # quit will trigger sys.exit()
        wiz.run()

    out = capsys.readouterr().out
    assert "Commands:" in out
    assert "add <filter_name>" in out

    monkey.undo()


# --- Add command behaviors ---


def test_wizard_add_invalid_name(capsys):
    """Adding unknown step should print error and not add."""
    inputs = iter(["add foo", "quit"])
    monkey = pytest.MonkeyPatch()
    monkey.setattr(actions, "is_valid_step", lambda n: False)
    monkey.setattr(builtins, "input", lambda prompt="": next(inputs))
    wiz = actions.Wizard("in.jpk", "chan", None, None, None)
    with pytest.raises(SystemExit):
        wiz.run()
    out = capsys.readouterr().out
    assert "Unknown processing step: 'foo'" in out
    monkey.undo()


# --- Remove and move valid indexes ---


def test_wizard_remove_and_move_valid(capsys):
    """Test remove then move on populated steps."""
    # Preload two steps
    inputs = iter(
        [
            "add mask_threshold",
            "",  # default threshold
            "add polynomial_flatten",
            "2",  # order=2
            "remove 1",  # remove first
            "list",  # should show only polynomial
            "add mask_mean_offset",
            "1.2",  # add new
            "move 2 1",  # swap positions
            "list",
            "exit",
        ]
    )
    monkey = pytest.MonkeyPatch()
    monkey.setattr(
        actions,
        "is_valid_step",
        lambda n: n in ["mask_threshold", "polynomial_flatten", "mask_mean_offset"],
    )  # noqa: E501
    monkey.setattr(builtins, "input", lambda prompt="": next(inputs))
    wiz = actions.Wizard("in.jpk", "chan", None, None, None)
    with pytest.raises(SystemExit):
        wiz.run()

    out = capsys.readouterr().out
    # After removal, only polynomial_flatten
    assert "1) polynomial_flatten (order=2)" in out
    # After move, mask_mean_offset should be first
    assert "1) mask_mean_offset (factor=1.2)" in out
    monkey.undo()


def test_wizard_remove_and_move_valid_analysis(capsys):
    """Test remove then move on populated analysis steps."""
    # Preload two steps
    inputs_analysis = iter(
        [
            "aadd feature_detection",
            "mask_threshold",  # mask_fn
            "",  # mask_key
            "",  # min_size default
            "",  # remove_edge default
            "",  # fill_holes default
            "0.3",  # hole_area 0.3
            "aadd x_means_clustering",
            "",  # detection_module default (feature_detection)
            "",  # coord_key default
            "",  # coord_collumns default
            "",  # use_time default
            "",  # min_k default
            "",  # max_k default
            "",  # normalise default
            "0.5",  # time_weight
            "",  # replicated default
            "",  # max_iter
            "0.2",  # bic_threshold
            "aremove 1",  # remove first
            "alist",  # should show only x_means_clustering
            "aadd log_blob_detection",
            "",  # min_sigma default
            "",  # max_sigma default
            "",  # num_sigma default
            "",  # threshold default
            "",  # overlap default
            "",  # include radius defualt
            "amove 2 1",  # swap positions
            "alist",
            "exit",
        ]
    )
    monkey = pytest.MonkeyPatch()
    monkey.setattr(
        actions,
        "is_valid_step",
        lambda n: n
        in ["feature_detection", "x_means_clustering", "log_blob_detection"],
    )
    monkey.setattr(builtins, "input", lambda prompt="": next(inputs_analysis))
    wiz = actions.Wizard("in.jpk", "chan", None, None, None)
    with pytest.raises(SystemExit):
        wiz.run()

    out_analysis = capsys.readouterr().out
    # After removal, only x_means_clustering
    assert (
        "x_means_clustering (detection_module=feature_detection, "
        "coord_key=features_per_frame, coord_columns=('centroid_x', 'centroid_y'), "
        "use_time=True, min_k=1, max_k=10, normalise=True, time_weight=0.5, "
        "replicates=3, max_iter=300, bic_threshold=0.2)" in out_analysis
    )
    # After move, log_blob_detection should be first
    assert (
        "1) log_blob_detection (min_sigma=1.0, max_sigma=5.0, num_sigma=10, "
        "threshold=0.1, overlap=0.5, include_radius=True)" in out_analysis
    )
    print("analysis output:")
    print(f"Out analysis: {out_analysis}")
    monkey.undo()


# --- Save workflow ---


def test_wizard_save_generates_yaml(tmp_path):
    """Save should serialize current steps to YAML file."""
    yaml_file = tmp_path / "cfg.yaml"
    inputs = iter(["add mask_threshold", "1.2", f"save {yaml_file}", "quit"])  # default
    monkey = pytest.MonkeyPatch()
    monkey.setattr(actions, "is_valid_step", lambda n: n == "mask_threshold")
    monkey.setattr(builtins, "input", lambda prompt="": next(inputs))
    wiz = actions.Wizard("in.jpk", "chan", None, None, None)
    with pytest.raises(SystemExit):
        wiz.run()

    data = yaml.safe_load(yaml_file.read_text())
    assert data == {"filters": [{"name": "mask_threshold", "threshold": 1.2}]}
    monkey.undo()


def test_wizard_asave_generates_yaml(tmp_path):
    """Test that asave serializes current asteps to YAML file."""
    yaml_file = tmp_path / "acfg.yaml"
    inputs = iter(
        [
            "aadd feature_detection",
            "mask_threshold",  # mask_fn
            "",  # mask_key
            "",  # min_size default
            "",  # remove_edge default
            "",  # fill_holes default
            "3",  # hole_area 3])  # default
            f"asave {yaml_file}",
            "quit",
        ]
    )
    monkey = pytest.MonkeyPatch()
    monkey.setattr(actions, "is_valid_step", lambda n: n == "feature_detection")
    monkey.setattr(builtins, "input", lambda prompt="": next(inputs))
    wiz = actions.Wizard("in.jpk", "chan", None, None, None)
    with pytest.raises(SystemExit):
        wiz.run()

    data = yaml.safe_load(yaml_file.read_text())
    assert data == {
        "analysis": [
            {
                "name": "feature_detection",
                "mask_fn": "mask_threshold",
                "min_size": 10,
                "remove_edge": True,
                "fill_holes": False,
            }
        ]
    }
    monkey.undo()


# Tests for utils

register_masking()
register_filters()
register_mask_filters()


@pytest.fixture
def mock_filters(monkeypatch):
    """Mock creating the mask and filters maps."""
    monkeypatch.setitem(MASK_MAP, "mock_mask", lambda: None)
    monkeypatch.setitem(FILTER_MAP, "mock_filter", lambda: None)


def test_parse_processing_string_with_mock(mock_filters):
    """Test the parseing of the processing steps string input."""
    from playNano.cli.utils import parse_processing_string

    s = "mock_mask:param1=1; mock_filter:param2=2"
    steps = parse_processing_string(s)
    assert steps[0][0] == "mock_mask"
    assert steps[1][0] == "mock_filter"


@pytest.mark.parametrize("name", ["invalid_step", "blur", "xyz123"])
def test_is_valid_step_false(name):
    """Test that invalid steps are identified."""
    assert is_valid_step(name) is False


def test_parse_processing_string_basic(mock_filters):
    """Test the parsing of a processing string give correct steps and params."""
    s = "remove_plane; gaussian_filter:sigma=2.0; mask_threshold:threshold=2"
    FILTER_MAP["gaussian_filter"] = lambda: None
    MASK_MAP["mask_threshold"] = lambda: None
    steps = parse_processing_string(s)
    assert steps == [
        ("remove_plane", {}),
        ("gaussian_filter", {"sigma": 2.0}),
        ("mask_threshold", {"threshold": 2}),
    ]


def test_parse_processing_string_with_bools_and_ints(mock_filters):
    """Test the parsing of bools and intergers from rpocessing strings."""
    MASK_MAP["some_mask"] = lambda: None
    s = "remove_plane; some_mask:enabled=true,threshold=5"
    steps = parse_processing_string(s)
    assert steps == [
        ("remove_plane", {}),
        ("some_mask", {"enabled": True, "threshold": 5}),
    ]


def test_parse_processing_string_invalid_name():
    """Test the parsing of a processing string with an unknown step."""
    with pytest.raises(ValueError, match="Unknown processing step: 'bad_step'"):
        parse_processing_string("bad_step")


def test_parse_processing_string_invalid_param_format():
    """Test parsing a string with an invalid parameter format."""
    s = "gaussian_filter:sigma2.0"
    with pytest.raises(ValueError, match="Invalid parameter expression"):
        parse_processing_string(s)


def test_parse_processing_file_yaml(tmp_path):
    """Test the parsing of a yaml processing file."""
    yaml_data = {
        "filters": [
            {"name": "remove_plane"},
            {"name": "gaussian_filter", "sigma": 2.0},
            {"name": "mask_threshold", "threshold": 3},
        ]
    }
    yaml_path = tmp_path / "filters.yaml"
    yaml_path.write_text(yaml.dump(yaml_data))
    steps = parse_processing_file(str(yaml_path))
    assert steps == [
        ("remove_plane", {}),
        ("gaussian_filter", {"sigma": 2.0}),
        ("mask_threshold", {"threshold": 3}),
    ]


def test_parse_processing_file_json(tmp_path):
    """Test the parsing of a json file."""
    json_data = {
        "filters": [
            {"name": "remove_plane"},
            {"name": "mask_threshold", "threshold": 1},
        ]
    }
    json_path = tmp_path / "filters.json"
    json_path.write_text(json.dumps(json_data))

    steps = parse_processing_file(str(json_path))
    assert steps == [
        ("remove_plane", {}),
        ("mask_threshold", {"threshold": 1}),
    ]


def test_parse_processing_file_invalid_file():
    """Test the identification of an invalid processing file."""
    with pytest.raises(FileNotFoundError):
        parse_processing_file("non_existent.yaml")


def test_parse_processing_file_invalid_schema(tmp_path):
    """Test the processing of a yaml file without the correct schema."""
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("not_a_dict: [1, 2, 3]")
    with pytest.raises(ValueError, match="processing file must contain top-level key"):
        parse_processing_file(str(bad_yaml))


def test_parse_processing_file_invalid_filter_entry(tmp_path):
    """Test the parsing of a processing file with an invlaid step."""
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(yaml.dump({"filters": [{"sigma": 1.0}]}))
    with pytest.raises(ValueError, match="must be a dict containing 'name'"):
        parse_processing_file(str(bad_yaml))


def test_parse_analysis_string_basic():
    """Parses single step with multiple numeric parameters."""
    s = "log_blob_detection:min_sigma=1.0,max_sigma=3.0"
    result = parse_analysis_string(s)
    assert result == [("log_blob_detection", {"min_sigma": 1.0, "max_sigma": 3.0})]


def test_parse_analysis_string_multiple_steps(monkeypatch):
    """Parses multiple steps with numeric parameters."""
    monkeypatch.setattr("playNano.cli.utils.is_valid_analysis_step", lambda name: True)
    s = "step1:param1=1;step2:param2=2,param3=3"
    result = parse_analysis_string(s)
    assert result == [
        ("step1", {"param1": 1}),
        ("step2", {"param2": 2, "param3": 3}),
    ]


def test_parse_analysis_string_with_booleans_and_strings(monkeypatch):
    """Parses booleans and strings in parameters."""
    monkeypatch.setattr("playNano.cli.utils.is_valid_analysis_step", lambda name: True)
    s = "foo:flag=true,label=sample"
    result = parse_analysis_string(s)
    assert result == [("foo", {"flag": True, "label": "sample"})]


def test_parse_analysis_string_no_params(monkeypatch):
    """Parses step with no parameters."""
    monkeypatch.setattr("playNano.cli.utils.is_valid_analysis_step", lambda name: True)
    s = "bar"
    result = parse_analysis_string(s)
    assert result == [("bar", {})]


def test_parse_analysis_string_invalid_param_syntax(monkeypatch):
    """Raises on invalid param syntax with no '='."""
    monkeypatch.setattr("playNano.cli.utils.is_valid_analysis_step", lambda name: True)
    with pytest.raises(ValueError, match="Invalid parameter expression"):
        parse_analysis_string("step:invalidparam")


def test_parse_analysis_string_unknown_step():
    """Raises if an analysis step name is not recognized."""
    with pytest.raises(ValueError, match="Unknown analysis step: 'does_not_exist'"):
        parse_analysis_string("does_not_exist:param=1")


def test_parse_analysis_string_empty_segment():
    """Test that empty segments are skipped."""
    result = parse_analysis_string(" ; ;log_blob_detection:min_sigma=1.0")
    assert result == [("log_blob_detection", {"min_sigma": 1.0})]


def test_parse_analysis_string_invalid_param_expression():
    """Test that malformed param expression raises ValueError."""
    with pytest.raises(
        ValueError,
        match="Invalid parameter expression 'badparam' in analysis step 'log_blob_detection'",  # noqa
    ):
        parse_analysis_string("log_blob_detection:badparam")


def test_parse_analysis_string_skips_empty_segments():
    """Test that empty segments are skipped."""

    result = parse_analysis_string(" ; ;log_blob_detection:min_sigma=1.0")
    assert result == [("log_blob_detection", {"min_sigma": 1.0})]


def test_parse_analysis_string_skips_empty_param_pairs():
    """Test that empty parameter pairs are skipped."""

    result = parse_analysis_string("log_blob_detection:min_sigma=1.0,,max_sigma=5.0")
    assert result == [("log_blob_detection", {"min_sigma": 1.0, "max_sigma": 5.0})]


def test_parse_analysis_string_invalid_step_name():
    """Test that unknown analysis step raises ValueError."""
    with patch("playNano.cli.utils.is_valid_analysis_step", return_value=False):
        from playNano.cli.utils import parse_analysis_string

        with pytest.raises(ValueError, match="Unknown analysis step: 'bad_step'"):
            parse_analysis_string("bad_step:param=1")


def make_temp_analysis_file(data: dict, suffix=".yaml") -> str:
    """Create a temporary YAML or JSON analysis config file."""
    with tempfile.NamedTemporaryFile("w+", suffix=suffix, delete=False) as f:
        if suffix == ".json":
            json.dump(data, f)
        else:
            yaml.safe_dump(data, f)
        return f.name


def test_parse_analysis_file_yaml(monkeypatch):
    """Parses YAML file into step/param tuples."""
    monkeypatch.setattr("playNano.cli.utils.is_valid_analysis_step", lambda name: True)
    data = {"analysis": [{"name": "foo", "param": 1}, {"name": "bar"}]}
    path = make_temp_analysis_file(data, ".yaml")
    result = parse_analysis_file(path)
    assert result == [("foo", {"param": 1}), ("bar", {})]


def test_parse_analysis_file_json(monkeypatch):
    """Parses JSON file into step/param tuples."""
    monkeypatch.setattr("playNano.cli.utils.is_valid_analysis_step", lambda name: True)
    data = {"analysis": [{"name": "step1", "thresh": 0.5}]}
    path = make_temp_analysis_file(data, ".json")
    result = parse_analysis_file(path)
    assert result == [("step1", {"thresh": 0.5})]


def test_parse_analysis_file_missing(monkeypatch):
    """Raises FileNotFoundError if path does not exist."""
    path = "nonexistent_file.yaml"
    with pytest.raises(
        FileNotFoundError, match="No such file or directory: 'nonexistent_file.yaml'"
    ):
        parse_analysis_file(path)


def test_parse_analysis_file_invalid_schema(monkeypatch):
    """Raises if top-level key 'analysis' is missing."""
    path = make_temp_analysis_file({"invalid": []}, ".yaml")
    with pytest.raises(
        ValueError,
        match="Invalid analysis file: expected top-level 'analysis' or list of steps",
    ):
        parse_analysis_file(path)


def test_parse_analysis_file_invalid_entries(monkeypatch):
    """Raises if any entry in 'analysis' is not a dict with 'name'."""
    monkeypatch.setattr("playNano.cli.utils.is_valid_analysis_step", lambda name: True)
    path = make_temp_analysis_file({"analysis": [123]}, ".yaml")
    with pytest.raises(
        ValueError, match="each step must be a mapping with a 'name' key"
    ):
        parse_analysis_file(path)


def test_parse_analysis_file_unknown_step(monkeypatch):
    """Raises if a step name in the file is not recognized."""
    monkeypatch.setattr("playNano.cli.utils.is_valid_analysis_step", lambda name: False)
    path = make_temp_analysis_file({"analysis": [{"name": "does_not_exist"}]}, ".yaml")
    with pytest.raises(ValueError, match="Unknown analysis step"):
        parse_analysis_file(path)


def test_parse_analysis_file_fallback_to_json(monkeypatch):
    """Falls back to JSON parsing if YAML parse fails."""

    # Valid JSON, but invalid YAML (YAML would interpret this as a string)
    json_text = '{"analysis": [{"name": "step1", "param": 1}]}'

    # Create a fake file containing this content
    with tempfile.NamedTemporaryFile("w+", suffix=".yaml", delete=False) as f:
        f.write(json_text)
        f.flush()
        path = f.name

    # Monkeypatch to accept any step as valid
    monkeypatch.setattr("playNano.cli.utils.is_valid_analysis_step", lambda name: True)

    # Should parse via fallback JSON logic
    result = parse_analysis_file(path)
    assert result == [("step1", {"param": 1})]


def test_parse_analysis_file_yaml_fails_json_succeeds(monkeypatch):
    """Forces YAML parse to fail and confirms JSON fallback succeeds."""
    data = {"analysis": [{"name": "step1", "value": 42}]}
    path = make_temp_analysis_file(data, suffix=".json")

    # Force yaml.safe_load to raise an exception
    monkeypatch.setattr(
        yaml, "safe_load", lambda _: (_ for _ in ()).throw(Exception("mock YAML fail"))
    )

    # Ensure is_valid_analysis_step returns True
    monkeypatch.setattr(cli_utils, "is_valid_analysis_step", lambda name: True)

    result = parse_analysis_file(path)
    assert result == [("step1", {"value": 42})]


def test_parse_analysis_file_invalid_yaml_and_json(monkeypatch):
    """Raises ValueError if both YAML and JSON parsing fail."""
    # Write garbage content that is neither valid YAML nor JSON
    with tempfile.NamedTemporaryFile("w+", suffix=".yaml", delete=False) as f:
        f.write("this is not: [valid json or yaml")  # deliberately malformed
        f.flush()
        path = f.name

    with pytest.raises(
        ValueError, match="Unable to parse analysis file as YAML or JSON"
    ):
        parse_analysis_file(path)


def test__process_pending_entries(monkeypatch):
    """Test processing of pending entries with conditions."""

    # Mock _resolve_condition to return True for all conditions
    monkeypatch.setattr(
        "playNano.cli.utils._resolve_condition", lambda cond, kwargs: True
    )

    # Mock _prompt_and_cast to return a fixed value
    monkeypatch.setattr(
        "playNano.cli.utils._prompt_and_cast",
        lambda name, typ, default: f"mocked_{name}",
    )

    pending = [
        {"name": "param1", "type": str, "default": "default1", "condition": "always"},
        {"name": "param2", "type": int, "default": 42, "condition": "always"},
    ]
    kwargs = {}

    progressed, to_retry = _process_pending_entries(pending, kwargs)

    assert progressed is True
    assert to_retry == []
    assert kwargs["param1"] == "mocked_param1"
    assert kwargs["param2"] == "mocked_param2"


def test__normalize_loaded():
    """Test normalization of nested structures with tuples, lists, and dicts."""

    input_data = {"a": (1, 2, {"b": (3, 4)}), "c": [5, (6, 7)], "d": "unchanged"}

    expected_output = {"a": [1, 2, {"b": [3, 4]}], "c": [5, [6, 7]], "d": "unchanged"}

    result = _normalize_loaded(input_data)
    assert result == expected_output


def test__get_analysis_class_value_error():
    """Test ValueError raised when module isn't found in built-ins or entry points."""
    with (
        patch("playNano.cli.utils.BUILTIN_ANALYSIS_MODULES", {}),
        patch("playNano.cli.utils.metadata.entry_points") as mock_entry_points,
    ):
        mock_entry_points.return_value.select.return_value = []
        with pytest.raises(
            ValueError, match="Analysis module 'unknown_module' not found"
        ):
            _get_analysis_class("unknown_module")


def test__get_analysis_class_generic_exception(caplog):
    """Test generic Exception is logged and re-raised during entry point loading."""
    with (
        patch("playNano.cli.utils.BUILTIN_ANALYSIS_MODULES", {}),
        patch("playNano.cli.utils.metadata.entry_points") as mock_entry_points,
    ):
        mock_entry_points.return_value.select.side_effect = RuntimeError(
            "entry point failure"
        )
        with pytest.raises(RuntimeError, match="entry point failure"):
            _get_analysis_class("some_module")
        assert "Failed to load analysis module 'some_module'" in caplog.text


# --- Tests for the handlers


def test_handle_play_accepts_path_object():
    """Test handle_play accepts a Path object as input_file."""
    # input_file is a Path object, not a string
    input_path = Path("some/fake/path")

    args = Namespace(
        input_file=input_path,
        channel="height",
        processing=None,
        processing_file=None,
        output_folder=None,
        output_name=None,
        scale_bar_nm=100,
        zmin="auto",
        zmax="auto",
    )

    with patch("playNano.cli.handlers.play_pipeline_mode") as mock_play:
        handle_play(args)

        # Check that the Path object was passed directly
        mock_play.assert_called_once()
        called_path = mock_play.call_args.kwargs["input_file"]
        assert isinstance(called_path, Path)
        assert called_path == input_path


def test_handle_play_invalid_path_with_cli_flags():
    """Test handle_play provides infomative value error if cli flags in input_file."""
    bad_path = "C:\\Users\\test\\AFMdata\\ --channel Height"
    args = Namespace(
        input_file=bad_path,
        channel="height_trace",
        processing=None,
        processing_file=None,
        output_folder=None,
        output_name=None,
        scale_bar_nm=100,
    )

    with pytest.raises(ValueError) as excinfo:
        handle_play(args)

    assert "includes CLI flags" in str(excinfo.value)
    assert "--channel" in str(excinfo.value)
    assert "💡 FIX" in str(excinfo.value)


def make_args(**kwargs) -> argparse.Namespace:
    """Build a dummy argparse.Namespace."""
    defaults = {
        "input_file": "test_data/test.jpk",
        "channel": "Height",
        "output_folder": None,
        "output_name": None,
        "scale_bar_nm": 100,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


@patch("playNano.cli.handlers.Wizard")
def test_handle_processing_wizard_success(mock_wizard):
    """Test the processing wizard handler with valid arguments."""
    args = make_args()
    handle_wizard(args)
    mock_wizard.assert_called_once_with(
        input_file="test_data/test.jpk",
        channel="Height",
        output_folder=None,
        output_name=None,
        scale_bar_nm=100,
    )


@patch("playNano.cli.handlers.Wizard", side_effect=RuntimeError("Test error"))
def test_handle_processing_wizard_raises(mock_wizard, caplog):
    """Test that an error is raised if wizard mode fails."""
    args = make_args()

    with caplog.at_level("ERROR"), pytest.raises(SystemExit) as exc_info:
        handle_wizard(args)

    # Check that sys.exit was called with 1
    assert exc_info.value.code == 1

    # Check that an error was logged
    assert "Test error" in caplog.text

    # Optional: verify Wizard was actually called before the failure
    mock_wizard.assert_called_once()


@patch("playNano.cli.handlers.logging.basicConfig")
def test_setup_logging_defaults(mock_basic_config):
    """Test that setup_logging uses default logging configuration."""
    setup_logging()  # uses default level=logging.INFO
    mock_basic_config.assert_called_once_with(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,  # add this
    )


@patch("playNano.cli.handlers.logging.basicConfig")
def test_setup_logging_debug(mock_basic_config):
    """Test that setup_logging sets DEBUG level when specified."""
    setup_logging(logging.DEBUG)
    mock_basic_config.assert_called_once_with(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,  # add this
    )


@patch("playNano.cli.actions.export_gif")
@patch("playNano.cli.actions.export_bundles")
@patch("playNano.cli.actions.process_stack")
@patch("playNano.cli.actions.AFMImageStack.load_data")
@patch("builtins.input")
def test_wizard_mode_zscale_input(
    mock_input, mock_load_data, mock_process_stack, mock_export_bundles, mock_export_gif
):
    """Test that wizard mode correctly accepts zmin and zmax."""
    # Mock AFM stack
    mock_stack = MagicMock()
    mock_stack.n_frames = 2
    mock_stack.image_shape = (512, 512)
    mock_stack.frame_metadata = [{"timestamp": 0}, {"timestamp": 1}]
    mock_load_data.return_value = mock_stack
    mock_process_stack.return_value = mock_stack

    # Simulate user input sequence
    mock_input.side_effect = [
        "add gaussian_filter",  # add a filter
        "",  # accept default sigma
        "run",  # run processing
        "y",  # export results
        "tif",  # export formats
        "y",  # create GIF
        "0.0",  # zmin
        "1.0",  # zmax
        "quit",  # exit - now must be done explicitly
    ]
    wiz = Wizard(
        input_file="dummy.afm",
        channel="height_trace",
        output_folder="output",
        output_name="test",
        scale_bar_nm=100,
    )
    with pytest.raises(SystemExit) as exit_info:  # noqa
        wiz.run()

    mock_export_gif.assert_called_once()
    _, kwargs = mock_export_gif.call_args
    assert kwargs["zmin"] == 0.0
    assert kwargs["zmax"] == 1.0


# More wizard tests
# --- Helpers / fakes -------------------------------------------------------


class DummyStack:
    """Minimal AFM-like object used for tests."""

    def __init__(self):
        """Initialize a dummy stack with some fake data."""
        self.n_frames = 10
        self.image_shape = (64, 64)
        # metadata used by play pipeline if required
        self.frame_metadata = [{"line_rate": 1.0} for _ in range(self.n_frames)]
        # simulated processed dictionary (if any)
        self.processed = {"raw": True}


class FakeIO:
    """Fake IO that answers prompts from a queue and collects outputs."""

    def __init__(self, answers=None):
        """Initialize with a list of answers to return for prompts."""
        self.answers = iter(answers or [])
        self.outputs = []

    def ask(self, prompt: str) -> str:
        """Fake a prompt and return the next answer."""
        self.outputs.append(prompt)  # record prompt for inspection
        try:
            return next(self.answers)
        except StopIteration:
            # default: empty string (simulate pressing Enter)
            return ""

    def say(self, msg: str) -> None:
        """Fake output."""
        self.outputs.append(msg)


@pytest.fixture
def wizard(tmp_path, monkeypatch):
    """Fake wizard for testing."""
    # Patch AFMImageStack.load_data to return a dummy stack
    monkeypatch.setattr(
        actions,
        "AFMImageStack",
        type("AFMImageStack", (), {"load_data": lambda *a, **k: DummyStack()}),
    )
    w = actions.Wizard("in.jpk", "chan", str(tmp_path), "out", 100)
    return w


@pytest.fixture
def wizard_mock_io(tmp_path):
    """Fake wizard to test with the mocked IO."""
    # Setup Wizard with mocked IO so we control input/output
    io = IO()
    wizard = Wizard(
        input_file="dummy.afm",
        channel="height",
        output_folder=str(tmp_path),
        output_name="test_output",
        scale_bar_nm=100,
        io=io,
    )
    # patch load_data and process_stack inside Wizard init
    wizard.afm_stack = MagicMock(n_frames=2, image_shape=(512, 512))
    return wizard


# --- Tests ---------------------------------------------------------------


def test_handle_add_and_handle_run_processing(monkeypatch, tmp_path):
    """
    Test adding a processing step (gaussian_filter) and running processing.

    - monkeypatch ask_for_processing_params to avoid interactive prompt
    - monkeypatch AFMImageStack.load_data and process_stack
    """
    # 1) patch load_data to return dummy AFM stack (so Wizard.__init__ succeeds)
    monkeypatch.setattr(
        actions.AFMImageStack,
        "load_data",
        staticmethod(lambda p, channel=None: DummyStack()),
    )

    # 2) patch ask_for_processing_params to return deterministic kwargs
    monkeypatch.setattr(
        actions, "ask_for_processing_params", lambda name: {"sigma": 1.5}
    )

    # 3) patch process_stack to return DummyStack when run
    monkeypatch.setattr(
        actions, "process_stack", lambda p, channel, steps: DummyStack()
    )

    # Create a tiny fake AFM file path
    fake_file = tmp_path / "fake.afm"
    fake_file.write_text("dummy")

    # Fake IO: export=no, gif=no (answers used in handle_run)
    io = FakeIO(answers=["n", "n"])

    # Instantiate wizard with fake IO
    w = Wizard(str(fake_file), "height", None, None, None, io=io)

    # Add gaussian_filter
    w.handle_add(["add", "gaussian_filter"])
    assert len(w.process_steps) == 1
    assert w.process_steps[0][0] == "gaussian_filter"
    assert w.process_steps[0][1]["sigma"] == 1.5

    # Run processing; should set processed_stack
    w.handle_run(["run"])
    assert w.processed_stack is not None
    # check that prompts/outputs were recorded
    assert any("Export results?" in out or "Create a GIF?" in out for out in io.outputs)


def test_handle_aadd_and_handle_arun_analysis(monkeypatch, tmp_path):
    """
    Test adding an inline analysis step and running analysis (no processing).

    - monkeypatch parse_analysis_string and analyze_pipeline_mode.
    """
    # patch load_data for Wizard init
    monkeypatch.setattr(
        actions.AFMImageStack,
        "load_data",
        staticmethod(lambda p, channel=None: DummyStack()),
    )

    # patch parse_analysis_string to return a known analysis spec
    monkeypatch.setattr(
        actions, "parse_analysis_string", lambda s: [("test_analysis", {"k": 2})]
    )

    # Patch AnalysisPipeline.run to return dummy data
    monkeypatch.setattr(
        actions.AnalysisPipeline,
        "run",
        lambda self, stack, log_to=None: {
            "environment": {"python": "3.10"},
            "analysis": {"result": "ok", "k": 2},
            "provenance": {"source": "test"},
        },
    )

    fake_file = tmp_path / "fake.afm"
    fake_file.write_text("dummy")

    io = FakeIO(answers=[])
    w = Wizard(str(fake_file), "height", str(tmp_path), "outname", None, io=io)

    # add inline analysis
    w.handle_aadd(["aadd", "test_analysis:k=2"])
    assert len(w.analysis_steps) == 1
    assert w.analysis_steps[0][0] == "test_analysis"
    assert w.analysis_steps[0][1]["k"] == 2

    # Run analysis
    w.handle_arun(["arun"])

    # Check output files
    json_path = tmp_path / "outname.json"
    h5_path = tmp_path / "outname.h5"
    assert json_path.exists()
    assert h5_path.exists()

    # Check JSON contents
    with open(json_path) as f:
        data = json.load(f)
    assert data["analysis"]["result"] == "ok"
    assert data["analysis"]["k"] == 2
    assert data["environment"]["python"] == "3.10"
    assert data["provenance"]["source"] == "test"


@patch("playNano.cli.actions.ask_for_analysis_params")
def test_handle_aadd_inline_spec(mock_ask_params, wizard_mock_io):
    """Test adding a analysis step."""
    wiz = wizard_mock_io
    # Provide a spec with colon so parse_analysis_string is triggered
    with patch("playNano.cli.actions.parse_analysis_string") as mock_parse_str:
        mock_parse_str.return_value = [("step1", {"param": 1})]
        wiz.io = MagicMock()
        wiz.handle_aadd(["aadd", "step1:param=1"])
        assert ("step1", {"param": 1}) in wiz.analysis_steps


@patch("playNano.cli.actions.ask_for_analysis_params")
def test_handle_aadd_interactive(mock_ask_params, wizard_mock_io):
    """Test adding ana nalysis step interactivly."""
    wiz = wizard_mock_io
    mock_ask_params.return_value = {"param": 2}
    wiz.io = MagicMock()
    wiz.handle_aadd(["aadd", "stepname"])
    assert any(name == "stepname" for name, _ in wiz.analysis_steps)


def test_handle_aremove(wizard_mock_io):
    """Test removign and analysis step."""
    wiz = wizard_mock_io
    wiz.analysis_steps = [("step1", {}), ("step2", {})]
    wiz.io = MagicMock()
    wiz.handle_aremove(["aremove", "1"])
    assert len(wiz.analysis_steps) == 1
    assert wiz.analysis_steps[0][0] == "step2"


def test_handle_amove(wizard_mock_io):
    """Test moving an analysis step."""
    wiz = wizard_mock_io
    wiz.analysis_steps = [("step1", {}), ("step2", {}), ("step3", {})]
    wiz.io = MagicMock()
    wiz.handle_amove(["amove", "1", "3"])
    # step1 should now be last
    assert wiz.analysis_steps[-1][0] == "step1"


def test_handle_aload(monkeypatch, tmp_path):
    """Test loading an analysis steps file."""
    # Create a dummy analysis file path
    dummy_file = tmp_path / "dummy.yaml"
    dummy_file.write_text("analysis:\n  - name: stepX")

    # Monkeypatch parse_analysis_file to return a known value
    monkeypatch.setattr(
        "playNano.cli.utils.parse_analysis_file", lambda path: [("stepX", {})]
    )

    # Create Wizard instance
    io = FakeIO(answers=[])
    wiz = Wizard(str(dummy_file), "height", str(tmp_path), "outname", None, io=io)

    # Run aload
    wiz.handle_aload(["aload", str(dummy_file)])

    # Assert that the analysis step was loaded
    assert any(name == "stepX" for name, _ in wiz.analysis_steps)


def test_handle_asave(tmp_path, wizard_mock_io):
    """Test the saving of an analysis Yaml config file."""
    wiz = wizard_mock_io
    wiz.analysis_steps = [("step1", {"a": 1})]
    save_path = tmp_path / "analysis.json"
    wiz.io = MagicMock()
    wiz.handle_asave(["asave", str(save_path)])
    assert save_path.exists()


@patch("playNano.cli.actions.AnalysisPipeline")
def test_handle_arun_runs_processing_and_analysis(mock_pipeline_cls, wizard_mock_io):
    """Test that handle_arun runs the processing and analysis pipelines."""
    wiz = wizard_mock_io
    wiz.process_steps = [("filter", {})]
    wiz.processed_stack = MagicMock()
    wiz.processed_steps_snapshot = wiz.process_steps.copy()
    wiz.analysis_steps = [("analysis1", {})]
    wiz.io = MagicMock()

    pipeline_mock = MagicMock()
    pipeline_mock.run.return_value = {
        "environment": {"python_version": "3.11"},
        "analysis": {"some_metric": 42},
        "provenance": {"processing_steps": []},
    }
    mock_pipeline_cls.return_value = pipeline_mock

    wiz.handle_arun(["arun"])

    pipeline_mock.run.assert_called_once()
    wiz.io.say.assert_any_call("Analysis complete (ran on processed stack).")


def test_handle_arun_no_analysis_steps(wizard_mock_io):
    """Test that handle_arun warns if no analysis steps are defined."""
    wiz = wizard_mock_io
    wiz.analysis_steps = []
    wiz.io = MagicMock()
    wiz.handle_arun(["arun"])
    wiz.io.say.assert_any_call("No analysis steps. Use aadd first.")


def make_fake_stack():
    """Create a minimal valid mock AFMImageStack with data and provenance."""
    stack = SimpleNamespace()
    stack.data = np.zeros((3, 5, 5))  # (n_frames, height, width)
    stack.provenance = {"analysis": {}}
    stack.analysis = {}
    return stack


def test_handle_analyze_success(monkeypatch):
    """Test handle_analyze calls analyze_pipeline_mode with a valid analysis step."""
    args = SimpleNamespace(
        input_file="input.afm",
        channel="height",
        analysis_steps="log_blob_detection:min_sigma=1.0",
        analysis_file=None,
        output_folder="/tmp",
        output_name=None,
    )

    # Patch AFMImageStack.load_data to return a real-looking mock stack
    monkeypatch.setattr(
        actions.AFMImageStack, "load_data", lambda *a, **k: make_fake_stack()
    )
    monkeypatch.setattr(actions, "warn_if_unprocessed", lambda stack: None)
    monkeypatch.setattr(actions, "make_json_safe", lambda record: record)
    monkeypatch.setattr(actions, "export_to_hdf5", lambda record, out_path: None)

    # Use real parse_analysis_string to allow valid step
    from playNano.cli.utils import parse_analysis_string

    monkeypatch.setattr(actions, "parse_analysis_string", parse_analysis_string)

    # Patch AnalysisPipeline
    class MockPipeline:
        def __init__(self):
            self.added = []

        def add(self, name, **kwargs):
            self.added.append((name, kwargs))

        def run(self, stack, log_to=None):
            return {"dummy_result": 123}

    monkeypatch.setattr(actions, "AnalysisPipeline", MockPipeline)

    handle_analyze(args)  # Should complete without exception


def test_handle_analyze_exception(monkeypatch, caplog):
    """Test handle_analyze logs error and exits with code 1 if an exception occurs."""
    args = SimpleNamespace(
        input_file="input.afm",
        channel="height",
        analysis_steps="log_blob_detection:min_sigma=1.0",
        analysis_file=None,
        output_folder="/tmp",
        output_name=None,
    )

    def raise_exc(*a, **k):
        raise RuntimeError("fail")

    monkeypatch.setattr("playNano.cli.actions.analyze_pipeline_mode", raise_exc)

    with patch("sys.exit") as mock_exit:
        with caplog.at_level(logging.ERROR):
            handle_analyze(args)
        mock_exit.assert_called_once_with(1)

    # Confirm the error message was logged
    assert any("fail" in record.message for record in caplog.records)


def test_warn_if_unprocessed_warns(caplog):
    """Test that warn_if_unprocessed logs a warning if data is unprocessed."""
    stack = DummyStack()
    stack.processed = None  # invalid
    caplog.set_level("WARNING")
    actions.warn_if_unprocessed(stack)
    assert "not been run through" in caplog.text


def test_process_pipeline_mode_loaderror(monkeypatch, tmp_path, caplog):
    """Test that process_pipeline_mode handles LoadError."""
    caplog.set_level("ERROR")

    def boom(*a, **k):
        raise LoadError("bad")

    monkeypatch.setattr(actions, "process_stack", boom)
    with pytest.raises(SystemExit):
        actions.process_pipeline_mode(
            "f", "c", None, None, None, False, None, None, 100
        )
    assert "bad" in caplog.text


def test_play_pipeline_mode_loaderror(monkeypatch):
    """Test that play_pipeline_mode handles LoadError."""

    def boom(*a, **k):
        raise Exception("fail")

    monkeypatch.setattr(
        actions,
        "AFMImageStack",
        type("AFMImageStack", (), {"load_data": staticmethod(boom)}),
    )
    with pytest.raises(LoadError):
        actions.play_pipeline_mode("f", "c", None, None, None, None, 100)


def test_play_pipeline_mode_invalid_zmin_zmax(monkeypatch):
    """Test that play_pipeline_mode handles invalid zmin/zmax."""
    monkeypatch.setattr(
        actions,
        "AFMImageStack",
        type("AFMImageStack", (), {"load_data": lambda *a, **k: DummyStack()}),
    )
    called = {}

    def fake_gui(*a, **k):
        called["yes"] = True

    monkeypatch.setattr(actions, "gui_entry", fake_gui)
    actions.play_pipeline_mode(
        "f", "c", None, None, None, None, 100, zmin="bad", zmax="bad"
    )
    assert called["yes"]


def test_wizard_unknown_and_help(wizard, capsys):
    """Test that the wizard handles unknown commands and help."""
    inputs = iter(["foo", "help", "quit"])
    wizard.io.ask = lambda prompt="": next(inputs)
    with pytest.raises(SystemExit):
        wizard.run()
    out = capsys.readouterr().out
    assert "Unknown command" in out
    assert "Commands:" in out


def test_wizard_run_with_no_steps(wizard, capsys):
    """Test that the wizard reports no steps if run without added steps."""
    wizard.handle_run([])
    assert "No steps to run" in capsys.readouterr().out


def test_handle_add_invalid(wizard, capsys):
    """Test that invalid processing steps are flagged."""
    wizard.handle_add(["add"])  # no name
    wizard.handle_add(["add", "not_a_step"])  # invalid name
    out = capsys.readouterr().out
    assert "Usage: add" in out
    assert "Unknown processing step" in out


def test_handle_move_and_remove_out_of_range(wizard, capsys):
    """Test that moving and removing steps out of range is handled."""
    wizard.handle_remove(["remove"])  # bad args
    wizard.handle_move(["move", "1", "2"])  # no steps
    out = capsys.readouterr().out
    assert "Usage: remove" in out or "Indices out of range" in out


def test_analysis_no_steps(wizard, capsys):
    """Test that handle_arun reports no analysis steps."""
    wizard.handle_arun([])
    assert "No analysis steps" in capsys.readouterr().out


def test_analysis_with_processing_reruns(monkeypatch, wizard, tmp_path):
    """Test that handle_arun runs analysis with processing steps."""
    wizard.process_steps = [("step", {})]
    wizard.analysis_steps = [("analysis", {})]
    dummy_stack = DummyStack()
    monkeypatch.setattr(actions, "process_stack", lambda *a, **k: dummy_stack)

    class DummyPipeline:
        """Dummy pipeline to simulate analysis."""

        def add(self, *a, **k):
            """Add method in DummyPipeline."""
            pass

        def run(self, *a, **k):
            """Eun method that returns a value in the DummyPipeline."""
            return {"result": 42}

    monkeypatch.setattr(actions, "AnalysisPipeline", lambda: DummyPipeline())
    monkeypatch.setattr(actions, "make_json_safe", lambda x: x)
    monkeypatch.setattr(actions, "export_to_hdf5", lambda *a, **k: None)
    wizard.handle_arun([])  # should run analysis without error


def test_handle_asave_and_aload(tmp_path, wizard):
    """Test that analysis steps are saved and loaded."""
    wizard.analysis_steps = [("a", {"p": 1})]
    yaml_path = tmp_path / "f.yaml"
    json_path = tmp_path / "f.json"
    wizard.handle_asave([None, str(yaml_path)])
    assert yaml_path.exists()
    wizard.handle_asave([None, str(json_path)])
    assert json_path.exists()

    # Now load from JSON
    with json_path.open("w") as f:
        json.dump({"analysis": [{"name": "x"}]}, f)
    wizard.handle_aload(["aload", str(json_path)])
    assert wizard.analysis_steps


def test_asave_aload_roundtrip(tmp_path):
    """Test that analysis steps can be saved and loaded correctly."""
    io = CaptureIO([""])  # stub IO if needed
    wiz = make_wizard(tmp_path, io)
    wiz.analysis_steps = [
        (
            "feature_detection",
            {"mask_fn": "mask_threshold", "coord_columns": ("x", "y")},
        )
    ]
    outp = tmp_path / "test.yaml"
    wiz.handle_asave(["asave", str(outp)])
    # now load via parse_analysis_file
    loaded = parse_analysis_file(str(outp))
    assert loaded[0][0] == "feature_detection"
    assert loaded[0][1]["coord_columns"] == ["x", "y"]  # tuple turned into list


def test_print_env_info(monkeypatch, capsys):
    """Help command should print available commands."""
    # Patch gather_environment_info in the namespace where print_env_info will import it
    monkeypatch.setattr(
        "playNano.utils.system_info.gather_environment_info", lambda: {"k": "v"}
    )

    actions.print_env_info()
    out = capsys.readouterr().out
    assert '"k": "v"' in out


def test_wizard_run_with_export_and_gif(monkeypatch, tmp_path):
    """Test that the wizard runs with data export and GIF creation."""
    # Patch AFMImageStack load
    monkeypatch.setattr(
        actions,
        "AFMImageStack",
        SimpleNamespace(load_data=lambda *a, **k: DummyStack()),
    )
    # Patch process_stack to return dummy stack
    monkeypatch.setattr(actions, "process_stack", lambda *a, **k: DummyStack())

    # Patch export functions to record calls
    called = {}
    monkeypatch.setattr(
        actions, "export_bundles", lambda *a, **k: called.setdefault("bundles", True)
    )
    monkeypatch.setattr(
        actions, "export_gif", lambda *a, **k: called.setdefault("gif", True)
    )

    # Patch is_valid_step/get_processing_step_type to accept our fake filter
    monkeypatch.setattr(actions, "is_valid_step", lambda name: True)
    monkeypatch.setattr(actions, "get_processing_step_type", lambda name: "filter")
    monkeypatch.setattr(actions, "ask_for_processing_params", lambda name: {})

    # IO stub to feed commands and parameters
    inputs = iter(
        [
            "add fake_filter",  # add a processing step
            "run",  # run processing
            "y",  # export? yes
            "tif,npz",  # formats
            "y",  # gif? yes
            "auto",  # zmin
            "auto",  # zmax
            "quit",  # exit
        ]
    )

    class StubIO(actions.IO):
        def ask(self, prompt=""):
            return next(inputs)

        def say(self, msg):
            print(msg)

    wiz = actions.Wizard("in.jpk", "chan", tmp_path, "outname", 100, io=StubIO())
    with pytest.raises(SystemExit):
        wiz.run()

    assert called["bundles"]
    assert called["gif"]


def test_ask_for_processing_params_no_conditions(monkeypatch):
    """Test asking for parameters with no conditions."""
    from playNano.cli.utils import ask_for_processing_params

    def dummy_func(data, param1: int = 5, param2: str = "default"):
        pass

    monkeypatch.setattr(
        "playNano.cli.utils._get_processing_callable", lambda name: dummy_func
    )
    monkeypatch.setattr(
        "builtins.input", lambda prompt: "42" if "param1" in prompt else "hello"
    )

    result = ask_for_processing_params("dummy_step")
    assert result == {"param1": 42, "param2": "hello"}


def test_ask_for_processing_params_condition_false(monkeypatch):
    """Test that parameters with False conditions are skipped."""

    def dummy_func(data, param1: int = 5, param2: str = "default"):
        pass

    dummy_func._param_conditions = {"param1": lambda kwargs: False}

    monkeypatch.setattr(
        "playNano.cli.utils._get_processing_callable", lambda name: dummy_func
    )
    monkeypatch.setattr("builtins.input", lambda prompt: "hello")

    result = ask_for_processing_params("dummy_step")
    assert result == {"param2": "hello"}  # param1 skipped


def test_ask_for_processing_params_condition_keyerror(monkeypatch):
    """Test that KeyError in condition postpones the parameter."""

    def dummy_func(data, param1: int = 5, param2: str = "default"):
        pass

    dummy_func._param_conditions = {"param1": lambda kwargs: kwargs["missing"]}

    monkeypatch.setattr(
        "playNano.cli.utils._get_processing_callable", lambda name: dummy_func
    )
    monkeypatch.setattr("builtins.input", lambda prompt: "42")

    result = ask_for_processing_params("dummy_step")
    assert result == {"param1": 42, "param2": "42"}  # param2 is str


def test_parse_processing_file_not_found():
    """Test that missing file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        parse_processing_file("nonexistent.yaml")


def test_parse_processing_file_invalid_yaml_and_json(tmp_path):
    """Test that invalid YAML and JSON raises ValueError."""
    path = tmp_path / "bad.yaml"
    path.write_text("not: valid: yaml: or json")

    with pytest.raises(
        ValueError, match="Unable to parse processing file as YAML or JSON"
    ):
        parse_processing_file(str(path))


def test_parse_processing_file_missing_filters_key(tmp_path):
    """Test that missing 'filters' key raises ValueError."""
    path = tmp_path / "nofilters.yaml"
    path.write_text("not_filters: []")

    with pytest.raises(ValueError, match="must contain top-level key 'filters'"):
        parse_processing_file(str(path))


def test_parse_processing_file_filters_not_list(tmp_path):
    """Test that non-list 'filters' raises ValueError."""
    path = tmp_path / "badfilters.yaml"
    path.write_text("filters: not_a_list")

    with pytest.raises(ValueError, match="'filters' must be a list"):
        parse_processing_file(str(path))


def test_parse_processing_file_entry_missing_name(tmp_path):
    """Test that entry without 'name' raises ValueError."""
    path = tmp_path / "missingname.yaml"
    path.write_text("filters:\n  - threshold: 2")

    with pytest.raises(ValueError, match="must be a dict containing 'name'"):
        parse_processing_file(str(path))


def test_parse_processing_file_invalid_step_name(tmp_path):
    """Test that unknown step name raises ValueError."""
    path = tmp_path / "badstep.yaml"
    path.write_text("filters:\n  - name: not_a_step")

    with patch("playNano.cli.utils.is_valid_step", return_value=False):
        with pytest.raises(ValueError, match="Unknown processing step"):
            parse_processing_file(str(path))


def test_parse_processing_file_valid_yaml(tmp_path):
    """Test that valid YAML returns parsed steps."""
    path = tmp_path / "valid.yaml"
    path.write_text(
        """
filters:
  - name: gaussian_filter
    sigma: 2.0
  - name: threshold_mask
    threshold: 5
"""
    )
    with patch("playNano.cli.utils.is_valid_step", return_value=True):
        result = parse_processing_file(str(path))
        assert result == [
            ("gaussian_filter", {"sigma": 2.0}),
            ("threshold_mask", {"threshold": 5}),
        ]


def test_prompt_remaining_skips_on_false_condition():
    """Test that _prompt_remaining skips entries with False condition."""
    entry = {
        "name": "param1",
        "type": int,
        "default": 0,
        "condition": lambda kwargs: False,
    }
    kwargs = {}

    with patch("playNano.cli.utils._prompt_and_cast") as mock_cast:
        from playNano.cli.utils import _prompt_remaining

        _prompt_remaining([entry], kwargs)
        mock_cast.assert_not_called()
        assert kwargs == {}


def test_prompt_remaining_adds_value():
    """Test that _prompt_remaining adds value when condition passes."""
    entry = {"name": "param1", "type": int, "default": 0, "condition": None}
    kwargs = {}

    with patch("playNano.cli.utils._prompt_and_cast", return_value=42):
        from playNano.cli.utils import _prompt_remaining

        _prompt_remaining([entry], kwargs)
        assert kwargs == {"param1": 42}


def test_prompt_signature_remaining_skips_on_false_condition():
    """Test that _prompt_signature_remaining skips parameters with False condition."""
    param = inspect.Parameter(
        "param1", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=0, annotation=int
    )
    conds = {"param1": lambda kwargs: False}
    kwargs = {}

    with patch("playNano.cli.utils._prompt_and_cast") as mock_cast:
        from playNano.cli.utils import _prompt_signature_remaining

        _prompt_signature_remaining([("param1", param)], kwargs, conds)
        mock_cast.assert_not_called()
        assert kwargs == {}


def test_prompt_signature_remaining_adds_value():
    """Test that _prompt_signature_remaining adds value when condition passes."""
    param = inspect.Parameter(
        "param1", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=0, annotation=int
    )
    conds = {}
    kwargs = {}

    with patch("playNano.cli.utils._prompt_and_cast", return_value=42):
        from playNano.cli.utils import _prompt_signature_remaining

        _prompt_signature_remaining([("param1", param)], kwargs, conds)
        assert kwargs == {"param1": 42}


def test_resolve_condition_keyerror():
    """Test that KeyError in condition returns None."""

    def condition(kwargs):
        return kwargs["missing"]

    assert _resolve_condition(condition, {}) is None


def test_resolve_condition_exception():
    """Test that general exceptions return True."""

    def condition(kwargs):
        raise RuntimeError("unexpected")

    assert _resolve_condition(condition, {}) is True


def test_resolve_condition_other_exception():
    """Test that other exceptions return True."""

    def cond():
        return lambda kwargs: 1 / 0

    assert _resolve_condition(cond, {}) is True


def test_resolve_condition_false():
    """Test that False condition returns False."""

    def condition(kwargs):
        return False

    assert _resolve_condition(condition, {}) is False


def test_resolve_condition_none():
    """Test that None condition returns True."""
    assert _resolve_condition(None, {}) is True


@pytest.mark.parametrize(
    "input_obj,expected",
    [
        (Path("/some/path"), str(Path("/some/path"))),
        (np.int32(42), 42),
        (np.float64(3.14), 3.14),
        (np.array([1, 2, 3]), [1, 2, 3]),
        (42, 42),
        (3.14, 3.14),
        ("hello", "hello"),
        (True, True),
        (None, None),
        ((1, 2, 3), [1, 2, 3]),
        ([1, (2, 3)], [1, [2, 3]]),
        ({"a": (1, 2), "b": np.array([3, 4])}, {"a": [1, 2], "b": [3, 4]}),
    ],
)
def test_sanitize_for_dump_basic(input_obj, expected):
    """Test basic sanitization of various Python and NumPy types for safe dumping."""
    result = _sanitize_for_dump(input_obj)
    assert result == expected


def test_sanitize_for_dump_nested_dict():
    """Test recursive sanitization of a nested dictionary with mixed types."""
    input_obj = {
        "path": Path("/tmp/file"),
        "data": {
            "array": np.array([1.0, 2.0]),
            "tuple": (np.int32(1), np.float64(2.0)),
        },
    }
    expected = {
        "path": str(Path("/tmp/file")),
        "data": {
            "array": [1.0, 2.0],
            "tuple": [1, 2.0],
        },
    }
    result = _sanitize_for_dump(input_obj)
    assert result == expected


def test_sanitize_for_dump_userdict():
    """Test sanitization of a UserDict containing NumPy and Path types."""
    user_dict = UserDict({"x": np.int32(5), "y": Path("/home")})
    expected = {"x": 5, "y": str(Path("/home"))}
    result = _sanitize_for_dump(user_dict)
    assert result == expected


def test_sanitize_for_dump_fallback():
    """Test fallback string conversion for unsupported custom objects."""

    class CustomObject:
        """Custom object with string representation for fallback sanitization."""

        def __str__(self):
            return "custom"

    obj = CustomObject()
    result = _sanitize_for_dump(obj)
    assert result == "custom"


class CaptureIO(actions.IO):
    """IO adapter that takes an iterator of inputs and records outputs."""

    def __init__(self, inputs):
        """Initialize with a list of inputs to return for prompts."""
        self._inputs = iter(inputs)
        self.outputs = []

    def ask(self, prompt=""):
        """Return the next input."""
        try:
            return next(self._inputs)
        except StopIteration:
            raise EOFError() from None

    def say(self, msg):
        """Add output messages to a list."""
        # keep trimming trailing new-lines for easy assertions
        self.outputs.append(str(msg).strip())


@pytest.fixture(autouse=True)
def patch_stack_and_processing(monkeypatch):
    """Patch AFMImageStack.load_data & process_stack to return DummyStack."""
    monkeypatch.setattr(
        actions,
        "AFMImageStack",
        SimpleNamespace(load_data=lambda *a, **k: DummyStack()),
    )
    monkeypatch.setattr(actions, "process_stack", lambda *a, **k: DummyStack())
    yield


def make_wizard(tmp_path, io):
    """Create a Wizard instance pointing to a small temporary input file."""
    in_file = tmp_path / "in.jpk"
    in_file.write_text("dummy")  # file must exist for Wizard.__init__
    return actions.Wizard(str(in_file), "chan", str(tmp_path), "out", 100, io=io)


@pytest.mark.parametrize(
    "inputs, expected_substrings",
    [
        (["help", "quit"], ["Commands:", "add <filter_name>"]),
        (["notacommand", "quit"], ["Unknown command"]),
    ],
)
def test_help_and_unknown_command(tmp_path, inputs, expected_substrings):
    """Test that help and unknown commands are handled correctly."""
    io = CaptureIO(inputs)
    wiz = make_wizard(tmp_path, io)
    with pytest.raises(SystemExit):
        wiz.run()
    out = "\n".join(io.outputs)
    for substr in expected_substrings:
        assert substr in out


def test_run_with_export_and_gif(tmp_path, monkeypatch):
    """Test that wizard can data export and make gif after processing with run."""
    called = {}

    # patch exports to mark calls
    monkeypatch.setattr(
        actions, "export_bundles", lambda *a, **k: called.setdefault("bundles", True)
    )
    monkeypatch.setattr(
        actions, "export_gif", lambda *a, **k: called.setdefault("gif", True)
    )

    # accept any filter name and return no params
    monkeypatch.setattr(actions, "is_valid_step", lambda name: True)
    monkeypatch.setattr(actions, "get_processing_step_type", lambda name: "filter")
    monkeypatch.setattr(actions, "ask_for_processing_params", lambda name: {})

    inputs = [
        "add fake_filter",  # add
        "run",  # run processing
        "y",  # export?
        "tif,npz",  # formats
        "y",  # create gif?
        "auto",  # zmin
        "auto",  # zmax
        "quit",
    ]
    io = CaptureIO(inputs)
    wiz = make_wizard(tmp_path, io)
    with pytest.raises(SystemExit):
        wiz.run()

    assert called.get("bundles", False) is True
    assert called.get("gif", False) is True
    assert ("fake_filter", {}) in wiz.process_steps


@pytest.mark.parametrize(
    "map_name,step_name",
    [
        ("FILTER_MAP", "filter_step"),
        ("MASK_MAP", "mask_step"),
        ("MASK_FILTERS_MAP", "mask_filter_step"),
        ("VIDEO_FILTER_MAP", "video_filter_step"),
        ("STACK_EDIT_MAP", "stack_edit_step"),
    ],
)
def test_get_processing_callable_maps(map_name, step_name):
    """Test that _get_processings_callable maps to correct function."""
    module_path = "playNano.cli.utils"
    with patch(f"{module_path}.{map_name}", {step_name: lambda x: x}):
        result = _get_processing_callable(step_name)
        assert callable(result)


def test_get_processing_callable_filter():
    """Test that a known filter step returns the correct callable."""

    def dummy_func():
        return lambda x: x

    with patch("playNano.cli.utils.FILTER_MAP", {"dummy_filter": dummy_func}):
        result = _get_processing_callable("dummy_filter")
        assert result is dummy_func


def test_get_processing_callable_mask():
    """Test that a known filter step returns the correct callable."""

    def dummy_mask():
        return lambda x: x

    with patch("playNano.cli.utils.MASK_MAP", {"dummy_mask": dummy_mask}):
        result = _get_processing_callable("dummy_mask")
        assert result is dummy_mask


def test_get_processing_callable_plugin():
    """Test that a plugin entry point is loaded correctly."""
    mock_entry_point = MagicMock()
    mock_entry_point.load.return_value = "plugin_callable"
    with patch(
        "playNano.cli.utils._PLUGIN_ENTRYPOINTS", {"plugin_step": mock_entry_point}
    ):
        result = _get_processing_callable("plugin_step")
        assert result == "plugin_callable"


def test_get_processing_callable_not_found():
    """Test that an unknown step raises ValueError."""
    with (
        patch("playNano.cli.utils.FILTER_MAP", {}),
        patch("playNano.cli.utils.MASK_MAP", {}),
        patch("playNano.cli.utils.MASK_FILTERS_MAP", {}),
        patch("playNano.cli.utils.VIDEO_FILTER_MAP", {}),
        patch("playNano.cli.utils.STACK_EDIT_MAP", {}),
        patch("playNano.cli.utils._PLUGIN_ENTRYPOINTS", {}),
    ):
        with pytest.raises(
            ValueError, match="Processing step 'unknown_step' not found"
        ):
            _get_processing_callable("unknown_step")


def test_get_processing_step_type_filter():
    """Test that get_processing_step_type correctly identifies filters."""
    with patch("playNano.cli.utils.FILTER_MAP", {"dummy_filter": lambda x: x}):
        assert get_processing_step_type("dummy_filter") == "filter"


def test_get_processing_step_type_plugin():
    """Test that _get_processing_step_type identifies plugins."""
    with patch("playNano.cli.utils._PLUGIN_ENTRYPOINTS", {"plugin_step": MagicMock()}):
        assert get_processing_step_type("plugin_step") == "plugin filter"


@pytest.mark.parametrize(
    "map_name,step_name,expected_type",
    [
        ("FILTER_MAP", "filter_step", "filter"),
        ("MASK_MAP", "mask_step", "mask generator"),
        ("MASK_FILTERS_MAP", "mask_filter_step", "mask filter"),
        ("_PLUGIN_ENTRYPOINTS", "plugin_step", "plugin filter"),
        ("VIDEO_FILTER_MAP", "video_filter_step", "video filter"),
        ("STACK_EDIT_MAP", "stack_edit_step", "stack edit"),
    ],
)
def test_get_processing_step_type_known(map_name, step_name, expected_type):
    """Test that get_processing step_type gives the expected type for known steps."""
    module_path = "playNano.cli.utils"
    with patch(f"{module_path}.{map_name}", {step_name: None}):
        result = get_processing_step_type(step_name)
        assert result == expected_type


def test_get_processing_step_type_unknown():
    """Test that get_processing_step_type handles unknown steps."""
    with (
        patch("playNano.cli.utils.FILTER_MAP", {}),
        patch("playNano.cli.utils.MASK_MAP", {}),
        patch("playNano.cli.utils.MASK_FILTERS_MAP", {}),
        patch("playNano.cli.utils._PLUGIN_ENTRYPOINTS", {}),
        patch("playNano.cli.utils.VIDEO_FILTER_MAP", {}),
        patch("playNano.cli.utils.STACK_EDIT_MAP", {}),
    ):
        result = get_processing_step_type("nonexistent_step")
        assert result == "unknown"


@pytest.mark.parametrize(
    "handler,args,expected",
    [
        (
            "handle_remove",
            ["remove", "1"],
            ("Index out of range", "No processing steps to remove."),
        ),
        ("handle_move", ["move", "1", "2"], ("Usage: move", "Indices out of range")),
    ],
)
def test_handlers_invalid_indices(tmp_path, handler, args, expected):
    """Tests that handlers can gracefully handle invlid indices."""
    io = CaptureIO([])
    wiz = make_wizard(tmp_path, io)
    method = getattr(wiz, handler)
    method(args)
    joined = "\n".join(io.outputs)
    if isinstance(expected, tuple):
        assert any(e in joined for e in expected)
    else:
        assert expected in joined


def test_prompt_for_processing_params_retry(tmp_path):
    """Test that prompt_for_processing_params retries on invalid input."""
    # polynomial_flatten expects integer 'order'; give bad input then good
    io = CaptureIO(["bad", "3"])
    wiz = make_wizard(tmp_path, io)
    params = wiz.prompt_for_processing_params("polynomial_flatten")
    assert params.get("order") == 3
    assert any("Invalid" in s for s in io.outputs)


def test_ask_with_spec_progresses():
    """Test _ask_with_spec returns kwargs when progress is made."""
    spec = [{"name": "param1", "type": str, "default": "default"}]

    with patch("playNano.cli.utils._process_pending_entries") as mock_process:
        mock_process.return_value = (True, [])  # progress made, no retry
        result = _ask_with_spec(spec)
        assert isinstance(result, dict)


def test_ask_with_spec_falls_back_to_prompt_remaining():
    """Test _ask_with_spec calls _prompt_remaining when no progress is made."""
    spec = [{"name": "param1", "type": str, "default": "default"}]

    with (
        patch("playNano.cli.utils._process_pending_entries") as mock_process,
        patch("playNano.cli.utils._prompt_remaining") as mock_prompt,
    ):
        mock_process.side_effect = [(False, spec)]
        result = _ask_with_spec(spec)
        mock_prompt.assert_called_once_with(spec, result)
        assert isinstance(result, dict)


@pytest.mark.parametrize(
    "monkeypatch_target,patch_val,expect_substr",
    [
        (
            "parse_analysis_string",
            (lambda s: (_ for _ in ()).throw(ValueError("bad spec"))),
            "Invalid analysis spec",
        ),
        (
            "ask_for_analysis_params",
            (lambda name: (_ for _ in ()).throw(RuntimeError("oops"))),
            "Unable to introspect module",
        ),
    ],
)
def test_aadd_error_branches(
    tmp_path, monkeypatch, monkeypatch_target, patch_val, expect_substr
):
    """Test that aadd handles errors gracefully."""
    io = CaptureIO([])
    wiz = make_wizard(tmp_path, io)
    monkeypatch.setattr(actions, monkeypatch_target, patch_val)
    # If parse_analysis_string is the target, call with colon spec;
    # otherwise use simple module name
    arg = (
        ["aadd", "bad:spec"]
        if monkeypatch_target == "parse_analysis_string"
        else ["aadd", "some_module"]
    )
    wiz.handle_aadd(arg)
    assert any(expect_substr in s for s in io.outputs)


def test_aload_and_asave_and_arun_branches(tmp_path, monkeypatch):
    """Test the branches of aload, asave, and arun handlers."""
    io = CaptureIO([])
    wiz = Wizard("dummy.afm", "height", str(tmp_path), "outname", None, io=io)

    # --- Test aload failure ---
    monkeypatch.setattr(
        actions,
        "parse_analysis_file",
        lambda p: (_ for _ in ()).throw(RuntimeError("nope")),
    )
    wiz.handle_aload(["aload", "fakepath"])
    assert any("Error loading analysis file" in s for s in io.outputs)

    # --- Test asave to JSON path ---
    wiz.analysis_steps.append(("mod", {"a": 1}))
    json_path = tmp_path / "a.json"
    wiz.handle_asave(["asave", str(json_path)])
    assert json_path.exists()

    # --- Patch AnalysisPipeline.run to use analysis step data ---
    def mock_run(self, stack, log_to=None):
        return {
            "environment": {"python": "3.10"},
            "analysis": {**wiz.analysis_steps[-1][1], "result": "ok", "k": 2},
            "provenance": {"source": "test"},
        }

    monkeypatch.setattr(actions.AnalysisPipeline, "run", mock_run)

    # --- Ensure afm_stack is set and no processing is triggered ---
    wiz.afm_stack = DummyStack()
    wiz.process_steps = []  # Ensure it runs on afm_stack, not processed_stack

    wiz.handle_arun([])

    # --- Check output files ---
    json_out = tmp_path / "outname.json"
    h5_out = tmp_path / "outname.h5"
    assert json_out.exists()
    assert h5_out.exists()

    # --- Check JSON contents ---
    with open(json_out) as f:
        data = json.load(f)
    assert data["analysis"]["result"] == "ok"
    assert data["analysis"]["a"] == 1


def test_run_eof_exit(tmp_path):
    """Test that EOFError in IO exits gracefully."""

    class EOFIO(actions.IO):
        """IO that raises EOFError on ask."""

        def ask(self, prompt=""):
            """Simulate EOF by raising EOFError."""
            raise EOFError()

        def say(self, msg):
            """Do nothing on say."""
            pass

    io = EOFIO()
    wiz = make_wizard(tmp_path, io)
    with pytest.raises(SystemExit):
        wiz.run()


# --- Test wizard IO ---


def test_io_ask(monkeypatch):
    """Test that IO.ask returns user input."""
    inputs = iter(["hello"])

    def mock_input(prompt):
        """Mock the promt input."""
        return next(inputs)

    monkeypatch.setattr(builtins, "input", mock_input)
    io_adapter = IO()
    result = io_adapter.ask("Enter: ")
    assert result == "hello"


def test_io_say(capsys):
    """Test that IO.say outputs a message."""
    io_adapter = IO()
    io_adapter.say("test message")
    captured = capsys.readouterr()
    assert "test message" in captured.out


@pytest.mark.parametrize(
    "s, expected_type, default, expected",
    [
        ("4", int, None, 4),
        ("4.5", float, None, 4.5),
        ("", float, 1.23, 1.23),
        ("True", bool, False, True),
        ("no", bool, True, False),
        ("1,2,3", list, None, ["1", "2", "3"]),
        ("a,b", tuple, None, ("a", "b")),
        ("4", Optional[int], None, 4),
        ("", Optional[int], 9, 9),
        ("5", Union[int, None], None, 5),
        ("", Union[int, None], 0, 0),
        ("abc", str, None, "abc"),
    ],
)
def test_cast_input(s, expected_type, default, expected):
    """Test the _cast_input utility function with various inputs and types."""
    assert cli_utils._cast_input(s, expected_type, default) == expected
