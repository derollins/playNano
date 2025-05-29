"""Tests for playNano's main.py CLI script and its functions."""

import logging
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from playNano.cli import (
    load_afm_stack,
    main,
    parse_args,
    prepare_output_directory,
    sanitize_output_name,
    setup_logging,
)


def test_setup_logging_sets_correct_level(caplog):
    """Check setup_logging sets the specified logging level correctly."""
    with caplog.at_level(logging.DEBUG):
        setup_logging(logging.DEBUG)
        logger = logging.getLogger("test_logger")
        logger.debug("Debug log")
    assert "Debug log" in caplog.text


def test_parse_args_defaults(monkeypatch):
    """Verify parse_args parses CLI args with correct defaults."""
    test_args = ["prog", "sample_path.jpk"]
    monkeypatch.setattr(sys, "argv", test_args)
    args = parse_args()
    assert args.input_file == "sample_path.jpk"
    assert args.channel == "height_trace"
    assert not args.save_raw
    assert args.log_level == "INFO"
    assert args.output_folder is None
    assert not args.make_gif


@patch("playNano.cli.load_afm_stack", return_value=False)
def test_main_file_not_found(mock_loader, monkeypatch, caplog):
    """Ensure main() exits with SystemExit if input file is missing."""
    # Patch sys.argv to simulate CLI input
    test_args = ["prog", "nonexistent.jpk"]
    monkeypatch.setattr(sys, "argv", test_args)

    # Patch Path.exists to simulate a missing file
    with patch("pathlib.Path.exists", return_value=False):
        caplog.set_level(logging.ERROR)
        with pytest.raises(SystemExit):  # main() calls sys.exit on error
            main()

    assert "File not found" in caplog.text


def test_load_jpk_file(resource_path):
    """Test loading a .jpk folder returns a valid AFM image stack."""
    # Locate folder containing the .jpk file
    folder = resource_path / "jpk_folder_0"

    # Attempt to load the AFM stack
    stack = load_afm_stack(folder)

    # Check that the output is an AFMImageStack (or similar)
    assert stack is not None
    assert stack.image_stack.ndim == 3  # Ensure it's a stack of 2D images
    assert stack.image_stack.shape[0] > 0  # At least one frame
    assert stack.image_stack.shape[1] > 0 and stack.image_stack.shape[2] > 0


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


def test_clear_directory_whitespace():
    """Test that directory paths are stripped of surrounding whitespace."""
    with patch.object(Path, "mkdir") as mock_mkdir:
        result = prepare_output_directory(" ./test/directory/for/test   ", "default")
        assert result == Path("./test/directory/for/test")
        mock_mkdir.assert_called_once()


def test_sanitize_invalid_characters_dir():
    """Test that invalid characters in directory paths raise a ValueError."""
    with pytest.raises(ValueError):
        prepare_output_directory("./inva|id", "default")


def test_sanitize_empty_output_dir():
    """Test that an empty directory path defaults to the fallback path."""
    assert prepare_output_directory("", "default") == Path("default")
