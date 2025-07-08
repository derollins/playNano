"""Tests for the playNano CLI."""

import argparse
import builtins
import json
import logging
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

import playNano.cli.actions as actions
from playNano.afm_stack import AFMImageStack
from playNano.cli.actions import wizard_mode
from playNano.cli.handlers import handle_play, handle_processing_wizard, setup_logging
from playNano.cli.utils import (
    FILTER_MAP,
    MASK_MAP,
    is_valid_step,
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
def test_run_pipeline_mode_load_error_logs_and_returns(mock_load, caplog):
    """Test that loading AFM data failure logs an error and returns None."""
    caplog.set_level(logging.ERROR)
    with pytest.raises(SystemExit) as exc:
        actions.run_pipeline_mode(
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
def test_run_pipeline_mode_flow(mock_gif, mock_bundles, mock_proc, mock_parse):
    """Test the full flow of run_pipeline_mode with processing string."""
    pipe = MagicMock()
    mock_proc.return_value = pipe

    actions.run_pipeline_mode(
        "in.jpk", "ch", "f1;f2:a=1", None, "npz,h5", True, "od", "nm", 10
    )
    mock_parse.assert_called_once()
    mock_proc.assert_called_once_with(
        Path("in.jpk"), "ch", [("f1", {}), ("f2", {"a": 1})]
    )
    mock_bundles.assert_called_once_with(pipe, "od", "nm", ["npz", "h5"])


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


@patch("playNano.cli.actions.play_stack_cv")
@patch("playNano.cli.actions.AFMImageStack.load_data")
def test_play_pipeline_mode_with_valid_zmin_zmax(
    mock_load_data, mock_play_stack_cv, tmp_path
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

    args, kwargs = mock_play_stack_cv.call_args
    assert kwargs["zmin"] == 0.0
    assert kwargs["zmax"] == 1.0


@patch("playNano.cli.actions.play_stack_cv")
@patch("playNano.cli.actions.AFMImageStack.load_data")
def test_play_pipeline_mode_with_invalid_zmin_logs_error(
    mock_load_data, mock_play_stack_cv, caplog, tmp_path
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


@patch("playNano.cli.actions.play_stack_cv")
@patch("playNano.cli.actions.AFMImageStack.load_data")
def test_play_pipeline_mode_with_invalid_zmax_logs_error(
    mock_load_data, mock_play_stack_cv, caplog, tmp_path
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


@patch("playNano.cli.actions.play_stack_cv")
@patch("playNano.cli.actions.AFMImageStack.load_data")
def test_play_pipeline_mode_defaults_fps_when_line_rate_missing(
    mock_load_data, mock_play_stack_cv, caplog
):
    """Test that play_pipeline_mode defaults to 1 fps when line_rate is missing."""
    mock_stack = MagicMock()
    mock_stack.frame_metadata = [{}]  # no line_rate
    mock_stack.image_shape = (512, 512)
    mock_load_data.return_value = mock_stack

    with caplog.at_level("WARNING"):
        actions.play_pipeline_mode(
            input_file="dummy.afm",
            channel="height_trace",
            processing_str=None,
            processing_file=None,
            output_folder=None,
            output_name=None,
            scale_bar_nm=100,
            zmin="auto",
            zmax="auto",
        )

    assert "defaulting to 1 fps" in caplog.text
    args, kwargs = mock_play_stack_cv.call_args
    assert kwargs["fps"] == 1.0


@patch("playNano.cli.actions.play_stack_cv")
@patch("playNano.cli.actions.AFMImageStack.load_data")
def test_play_pipeline_mode_computes_fps_from_line_rate(
    mock_load_data, mock_play_stack_cv
):
    """Test that the fps is calculated from the line rate and image shape."""
    mock_stack = MagicMock()
    mock_stack.frame_metadata = [{"line_rate": 2048}]
    mock_stack.image_shape = (512, 512)
    mock_load_data.return_value = mock_stack

    actions.play_pipeline_mode(
        input_file="dummy.afm",
        channel="height_trace",
        processing_str=None,
        processing_file=None,
        output_folder=None,
        output_name=None,
        scale_bar_nm=100,
        zmin="auto",
        zmax="auto",
    )

    args, kwargs = mock_play_stack_cv.call_args
    assert kwargs["fps"] == 4.0  # 2048 / 512


@patch("playNano.cli.actions.parse_processing_file")
@patch("playNano.cli.actions.play_stack_cv")
@patch("playNano.cli.actions.AFMImageStack.load_data")
def test_play_pipeline_mode_uses_processing_file(
    mock_load_data, mock_play_stack_cv, mock_parse_file
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
@patch("playNano.cli.actions.play_stack_cv")
@patch("playNano.cli.actions.AFMImageStack.load_data")
def test_play_pipeline_mode_uses_processing_str(
    mock_load_data, mock_play_stack_cv, mock_parse_str
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
            "add threshold_mask",
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
        lambda n: n in ["threshold_mask", "polynomial_flatten", "mask_mean_offset"],
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
    s = "remove_plane; gaussian_filter:sigma=2.0; threshold_mask:threshold=2"
    FILTER_MAP["gaussian_filter"] = lambda: None
    MASK_MAP["threshold_mask"] = lambda: None
    steps = parse_processing_string(s)
    assert steps == [
        ("remove_plane", {}),
        ("gaussian_filter", {"sigma": 2.0}),
        ("threshold_mask", {"threshold": 2}),
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
            {"name": "threshold_mask", "threshold": 3},
        ]
    }
    yaml_path = tmp_path / "filters.yaml"
    yaml_path.write_text(yaml.dump(yaml_data))

    steps = parse_processing_file(str(yaml_path))
    assert steps == [
        ("remove_plane", {}),
        ("gaussian_filter", {"sigma": 2.0}),
        ("threshold_mask", {"threshold": 3}),
    ]


def test_parse_processing_file_json(tmp_path):
    """Test the parsing of a json file."""
    json_data = {
        "filters": [
            {"name": "remove_plane"},
            {"name": "threshold_mask", "threshold": 1},
        ]
    }
    json_path = tmp_path / "filters.json"
    json_path.write_text(json.dumps(json_data))

    steps = parse_processing_file(str(json_path))
    assert steps == [
        ("remove_plane", {}),
        ("threshold_mask", {"threshold": 1}),
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
    )


@patch("playNano.cli.handlers.logging.basicConfig")
def test_setup_logging_debug(mock_basic_config):
    """Test that setup_logging sets DEBUG level when specified."""
    setup_logging(logging.DEBUG)
    mock_basic_config.assert_called_once_with(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
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
