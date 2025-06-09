"""Tests for the playNano CLI."""

import builtins
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

import playNano.cli.actions as actions
from playNano.errors import LoadError
from playNano.stack.afm_stack import AFMImageStack


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
    mock_bundles.assert_called_once_with(pipe, ["npz", "h5"], "od", "nm")


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
    """EOFError from input should exit cleanly with code 0."""
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
    inputs = iter(["add threshold_mask", "", f"save {yaml_file}", "quit"])  # default
    monkey = pytest.MonkeyPatch()
    monkey.setattr(actions, "is_valid_step", lambda n: n == "threshold_mask")
    monkey.setattr(builtins, "input", lambda prompt="": next(inputs))

    with pytest.raises(SystemExit):
        actions.wizard_mode("in.jpk", "chan", None, None, None)

    data = yaml.safe_load(yaml_file.read_text())
    assert data == {"filters": [{"name": "threshold_mask", "threshold": 1.0}]}
    monkey.undo()
