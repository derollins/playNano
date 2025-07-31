"""Tests for the playNano CLI."""

import argparse
import builtins
import json
import logging
import tempfile
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest
import yaml

import playNano.cli.actions as actions
from playNano.afm_stack import AFMImageStack
from playNano.cli import utils as cli_utils
from playNano.cli.actions import analyze_pipeline_mode, wizard_mode
from playNano.cli.entrypoint import setup_logging
from playNano.cli.handlers import handle_analyze, handle_play, handle_processing_wizard
from playNano.cli.utils import (
    FILTER_MAP,
    MASK_MAP,
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


@patch("playNano.cli.actions.AFMImageStack.load_data", side_effect=Exception("boom"))
def test_process_pipeline_mode_load_error_logs_and_returns(mock_load, caplog):
    """Test that loading AFM data failure logs an error and returns None."""
    caplog.set_level(logging.ERROR)
    with pytest.raises(SystemExit) as exc:
        actions.process_pipeline_mode(
            "in.jpk", "ch", None, None, None, False, None, None, None
        )
    assert exc.value.code == 1


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
        actions.wizard_mode("nofile.jpk", "ch", None, None, None)
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
    with pytest.raises(SystemExit) as exc:
        actions.wizard_mode("in.jpk", "chan", None, None, None)
    assert exc.value.code == 0


# --- Help and listing ---


def test_wizard_help_prints_commands(capsys):
    """Help command should print available commands."""
    inputs = iter(["help", "quit"])
    monkey = pytest.MonkeyPatch()
    monkey.setattr(builtins, "input", lambda prompt="": next(inputs))
    with pytest.raises(SystemExit):
        actions.wizard_mode("in.jpk", "chan", None, None, None)
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
    with pytest.raises(SystemExit):
        actions.wizard_mode("in.jpk", "chan", None, None, None)
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

    with pytest.raises(SystemExit):
        actions.wizard_mode("in.jpk", "chan", None, None, None)

    out = capsys.readouterr().out
    # After removal, only polynomial_flatten
    assert "1) polynomial_flatten (order=2)" in out
    # After move, mask_mean_offset should be first
    assert "1) mask_mean_offset (factor=1.2)" in out
    monkey.undo()


# --- Save workflow ---


def test_wizard_save_generates_yaml(tmp_path):
    """Save should serialize current steps to YAML file."""
    yaml_file = tmp_path / "cfg.yaml"
    inputs = iter(["add mask_threshold", "", f"save {yaml_file}", "quit"])  # default
    monkey = pytest.MonkeyPatch()
    monkey.setattr(actions, "is_valid_step", lambda n: n == "mask_threshold")
    monkey.setattr(builtins, "input", lambda prompt="": next(inputs))

    with pytest.raises(SystemExit):
        actions.wizard_mode("in.jpk", "chan", None, None, None)

    data = yaml.safe_load(yaml_file.read_text())
    assert data == {"filters": [{"name": "mask_threshold", "threshold": 1.0}]}
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
    with pytest.raises(ValueError, match="must contain top-level key 'filters'"):
        parse_analysis_file(path)


def test_parse_analysis_file_invalid_entries(monkeypatch):
    """Raises if any entry in 'analysis' is not a dict with 'name'."""
    monkeypatch.setattr("playNano.cli.utils.is_valid_analysis_step", lambda name: True)
    path = make_temp_analysis_file({"analysis": [123]}, ".yaml")
    with pytest.raises(ValueError, match="Each entry under 'analysis' must be a dict"):
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
        ValueError, match="Unable to parse processing file as YAML or JSON"
    ):
        parse_analysis_file(path)


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


@patch("playNano.cli.handlers.wizard_mode")
def test_handle_processing_wizard_success(mock_wizard):
    """Test the processing wizard handler with valid arguments."""
    args = make_args()
    handle_processing_wizard(args)
    mock_wizard.assert_called_once_with(
        input_file="test_data/test.jpk",
        channel="Height",
        output_folder=None,
        output_name=None,
        scale_bar_nm=100,
    )


@patch("playNano.cli.handlers.wizard_mode", side_effect=RuntimeError("Test error"))
def test_handle_processing_wizard_raises(mock_wizard, caplog):
    """Test that an error is raised if wizard mode fails."""
    args = make_args()

    with caplog.at_level("ERROR"), pytest.raises(SystemExit) as exc_info:
        handle_processing_wizard(args)

    # Check that sys.exit was called with 1
    assert exc_info.value.code == 1

    # Check that an error was logged
    assert "Test error" in caplog.text

    # Optional: verify wizard_mode was actually called before the failure
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
    ]

    with pytest.raises(SystemExit) as exit_info:  # noqa
        wizard_mode(
            input_file="dummy.afm",
            channel="height_trace",
            output_folder="output",
            output_name="test",
            scale_bar_nm=100,
        )

    mock_export_gif.assert_called_once()
    _, kwargs = mock_export_gif.call_args
    assert kwargs["zmin"] == "0.0"
    assert kwargs["zmax"] == "1.0"


def make_fake_stack():
    """Create a minimal valid mock AFMImageStack with data and provenance."""
    stack = SimpleNamespace()
    stack.data = np.zeros((3, 5, 5))  # (n_frames, height, width)
    stack.provenance = {"analysis": {}}
    stack.analysis = {}
    return stack


def test_handle_analyze_success(monkeypatch):
    """
    Test that handle_analyze successfully calls analyze_pipeline_mode
    with a valid analysis step and stack.
    """
    args = SimpleNamespace(
        input_file="input.afm",
        channel="height",
        analysis_steps="log_blob_detection:min_sigma=1.0",
        analysis_file=None,
        output_folder="/tmp",
        output_name=None,
    )

    # Patch AFMImageStack.load_data to return a real-looking mock stack
    from playNano.cli import actions

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
    """
    Test that handle_analyze logs error and exits with code 1 if an exception occurs.
    """
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
