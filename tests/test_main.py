"""Tests for playNano's main.py CLI script and its functions."""

import logging
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from playNano.afm_stack import AFMImageStack
from playNano.cli.entrypoint import main, setup_logging
from playNano.utils.io_utils import prepare_output_directory, sanitize_output_name


def test_setup_logging_sets_correct_level():
    """Test that setup_logging sets the root logger to the specified level."""
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    logger = logging.getLogger("test_logger")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    setup_logging(logging.DEBUG)  # sets root config
    logger.debug("Debug log")

    handler.flush()
    contents = stream.getvalue()
    assert "Debug log" in contents

    logger.removeHandler(handler)


def test_parse_args_defaults(monkeypatch):
    """Test that main() runs without errors with default arguments."""
    monkeypatch.setattr(sys, "argv", ["prog", "play", "sample_path.jpk"])
    monkeypatch.setattr(Path, "exists", lambda self: True)

    monkeypatch.setattr(
        AFMImageStack,
        "load_data",
        lambda path, channel=None: AFMImageStack(
            data=np.zeros((1, 10, 10)),
            pixel_size_nm=1.0,
            channel="height_trace",
            file_path=path,
            frame_metadata=[{"line_rate": 1.0}],
        ),
    )

    # Patch out the GUI entry function so QApplication isn't called
    monkeypatch.setattr("playNano.cli.actions.gui_entry", lambda *args, **kwargs: None)

    # Should not raise
    result = main()
    assert result is None


def test_load_jpk_file(resource_path):
    """
    Test loading a .jpk folder returns a valid AFMImageStack.

    Alternativly this is skipped on NumPy 2.x.
    """
    folder = resource_path / "jpk_folder_0"

    try:
        stack = AFMImageStack.load_data(folder)
    except AttributeError as e:
        # NumPy 2.0 removed ndarray.newbyteorder(); skip if that occurs
        msg = str(e)
        if "newbyteorder" in msg:
            pytest.skip(
                "Skipping test_load_jpk_file: NumPy 2.x incompatibility in AFMReader"
            )
        else:
            raise

    assert isinstance(stack, AFMImageStack)
    assert stack.data.ndim == 3
    assert stack.data.shape[0] > 0
    assert stack.data.shape[1] > 0 and stack.data.shape[2] > 0


def test_sanitize_valid_name():
    """Test that valid output names are stripped of whitespace."""
    assert sanitize_output_name("  result.gif  ", "default") == "result"


def test_sanitize_invalid_characters():
    """Test that invalid characters in output names raise a ValueError."""
    with pytest.raises(ValueError):
        sanitize_output_name("inva|id", "default")


def test_sanitize_empty_name():
    """Test that an empty output name defaults to the fallback name."""
    assert sanitize_output_name("", "default") == "default"


def test_clear_directory_whitespace(tmp_path, monkeypatch):
    """Test that directory paths are stripped of surrounding whitespace."""
    called = {"mk": False}

    def fake_mkdir(self, parents=False, exist_ok=False):
        called["mk"] = True

    monkeypatch.setattr(Path, "mkdir", fake_mkdir)
    result = prepare_output_directory(" ./test/directory/for/test   ", "default")
    expected = Path("./test/directory/for/test").resolve()
    assert result.resolve() == expected
    assert called["mk"] is True


def test_sanitize_invalid_characters_dir():
    """Test that invalid characters in directory paths raise a ValueError."""
    with pytest.raises(ValueError):
        prepare_output_directory("./inva|id", "default")


def test_sanitize_empty_output_dir():
    """Test that an empty directory path defaults to the fallback path."""
    result = prepare_output_directory("", "default")
    assert result.resolve().name == "default"


def test_setup_logging_warn_level():
    """Test that setup_logging sets the root logger to WARNING level."""
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    logger = logging.getLogger("test_logger")
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)

    setup_logging(logging.WARNING)
    logger.warning("A warning")

    handler.flush()
    contents = stream.getvalue()
    assert "A warning" in contents

    logger.removeHandler(handler)


def test_sanitize_name_with_extension():
    """Test sanitize_output_name strips extension properly."""
    assert sanitize_output_name("figure.gif", "fallback") == "figure"


def test_sanitize_name_with_weird_spacing():
    """Test sanitize_output_name strips weird spacing."""
    assert sanitize_output_name("   weird   name.gif  ", "fallback") == "weird   name"


def test_prepare_output_dir_with_valid_path(tmp_path):
    """Test prepare_output_directory creates valid directory."""
    test_path = tmp_path / "subdir"
    result = prepare_output_directory(str(test_path), "fallback")
    assert result.exists()
    assert result == test_path


def test_prepare_output_dir_default_used(tmp_path, monkeypatch):
    """Test fallback directory path is used when input is empty."""
    monkeypatch.chdir(tmp_path)
    result = prepare_output_directory("", "fallback_dir")
    expected = tmp_path / "fallback_dir"
    assert result.resolve() == expected.resolve()
    assert result.exists()


def test_main_invalid_command(monkeypatch, capfd):
    """Test main exits on unknown command."""
    monkeypatch.setattr(sys, "argv", ["prog", "nonsense"])
    with pytest.raises(SystemExit):
        main()
    out, err = capfd.readouterr()
    assert "invalid choice" in err.lower()


def test_main_missing_args(monkeypatch, capfd):
    """Test main exits if required args are missing."""
    monkeypatch.setattr(sys, "argv", ["prog"])
    with pytest.raises(SystemExit):
        main()
    out, err = capfd.readouterr()
    assert "usage:" in err.lower()


def test_prepare_output_directory_nested(tmp_path):
    """Test prepare_output_directory creates nested directories."""
    path = tmp_path / "nested1/nested2"
    result = prepare_output_directory(str(path), "fallback")
    assert result.exists()
    assert result == path


def test_sanitize_output_name_trims_and_defaults():
    """Test sanitize_output_name trims and uses default if needed."""
    assert sanitize_output_name("   ", "fallback") == "fallback"


def test_handle_play_file_not_found(tmp_path, caplog):
    """Test handle_play raises SystemExit if input file does not exist."""
    from argparse import Namespace

    from playNano.cli.handlers import handle_play

    args = Namespace(
        input_file="nonexistent.jpk",
        channel="height_trace",
        processing=None,
        processing_file=None,
        output_folder=None,
        output_name=None,
        scale_bar_nm=100,  # optional, but matches run handler
        zmin="auto",
        zmax="auto",
    )

    with pytest.raises(SystemExit):
        handle_play(args)
    assert "Failed to load nonexistent.jpk" in caplog.text


def test_handle_play_load_error(monkeypatch, tmp_path, caplog):
    """Test handle_play raises SystemExit if loading AFMImageStack fails."""
    from argparse import Namespace

    from playNano.cli.handlers import handle_play

    mock_load = MagicMock(side_effect=Exception("load failed"))
    monkeypatch.setattr("playNano.cli.actions.AFMImageStack.load_data", mock_load)

    file = tmp_path / "file.jpk"
    file.write_text("data")

    args = Namespace(
        input_file=str(file),
        channel="height_trace",
        processing=None,
        processing_file=None,
        output_folder=None,
        output_name=None,
        scale_bar_nm=None,
        zmin="auto",
        zmax="auto",
    )

    with pytest.raises(SystemExit):
        handle_play(args)
    assert "Failed to load" in caplog.text


def test_handle_process_bad_output_folder(monkeypatch, tmp_path, caplog):
    """Test handle_process raises SystemExit for invalid output folder path."""
    from argparse import Namespace

    from playNano.cli.handlers import handle_process

    good_stack = MagicMock()
    good_stack.frame_metadata = [{"timestamp": 1}]
    monkeypatch.setattr(
        "playNano.cli.actions.AFMImageStack.load_data", lambda *a, **k: good_stack
    )

    args = Namespace(
        input_file=str(tmp_path / "test.jpk"),
        channel="height_trace",
        processing=None,
        processing_file=None,
        scale_bar_nm=None,
        export="tif",
        make_gif=False,
        output_folder="bad|name",
        output_name=None,
        zmin="auto",
        zmax="auto",
    )
    (tmp_path / "test.jpk").write_text("data")

    with pytest.raises(SystemExit):
        handle_process(args)
    assert "Invalid characters in output folder" in caplog.text


def test_handle_process_make_gif(monkeypatch, tmp_path):
    """Test handle_process creates a GIF when make_gif is True."""
    from argparse import Namespace

    from playNano.cli.handlers import handle_process

    fake_data = np.zeros((10, 10, 10))
    fake_stack = MagicMock()
    fake_stack.data = fake_data
    fake_stack.pixel_size_nm = 1.0
    fake_stack.frame_metadata = [{"timestamp": i} for i in range(10)]
    fake_stack.channel = "height_trace"
    fake_stack.image_shape = (10, 10)
    fake_stack.file_path = tmp_path / "sample.jpk"
    fake_stack.processed = {}

    # Patch load_data to return our fake stack
    monkeypatch.setattr(
        "playNano.cli.actions.AFMImageStack.load_data", lambda *a, **k: fake_stack
    )

    # Patch the actual gif creation so no file is written
    monkeypatch.setattr("playNano.cli.actions.export_gif", lambda *a, **k: None)

    args = Namespace(
        input_file=str(tmp_path / "sample.jpk"),
        channel="height_trace",
        processing=None,
        processing_file=None,
        export=None,
        make_gif=True,
        output_folder=str(tmp_path),
        output_name="outputname",
        scale_bar_nm=100,
        zmin="auto",
        zmax="auto",
    )
    (tmp_path / "sample.jpk").write_text("x")

    # Run the function
    handle_process(args)


def test_main_no_command(monkeypatch, capsys):
    """Test main() exits with usage message when no command is provided."""
    import sys

    from playNano.cli.entrypoint import main

    monkeypatch.setattr(sys, "argv", ["prog"])
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 0
    captured = capsys.readouterr()
    assert "usage:" in captured.err
