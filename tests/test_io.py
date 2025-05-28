"""Tests for the functions within io."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image, ImageSequence

from playNano.io.gif_export import (
    create_gif_with_scale_and_timestamp,
    normalize_to_uint8,
)
from playNano.io.loader import (
    get_loader_for_file,
    get_loader_for_folder,
    load_afm_stack,
)
from playNano.loaders.read_asd_folder import load_asd_folder
from playNano.loaders.read_h5jpk import load_h5jpk
from playNano.loaders.read_jpk_folder import load_jpk_folder
from playNano.loaders.read_spm_folder import load_spm_folder
from playNano.stack.image_stack import AFMImageStack


@pytest.mark.parametrize(
    "filename, expected_ext",
    [
        ("example.JPK", ".jpk"),
        ("file1.JpK", ".jpk"),
        ("file2.AsD", ".asd"),
        ("file.spm", ".spm"),
    ],
)
def test_get_loader_for_folder_detects_extensions(tmp_path, filename, expected_ext):
    """Test that get_loader_for_folder detects file extension and returns loader."""
    (tmp_path / filename).touch()
    (tmp_path / "subfolder").mkdir()

    folder_loaders = {
        ".jpk": load_jpk_folder,
        ".asd": load_asd_folder,
        ".spm": load_spm_folder,
    }

    ext, loader = get_loader_for_folder(tmp_path, folder_loaders)
    assert ext.lower() == expected_ext
    assert callable(loader)


def test_load_afm_stack_raises_on_unsupported_folder(tmp_path):
    """Test load_afm_stack raises error when folder contains only unsupported files."""
    (tmp_path / "data.txt").touch()

    with pytest.raises(
        FileNotFoundError, match="No supported AFM files found in the folder."
    ):
        load_afm_stack(tmp_path)


def test_get_loader_for_folder_raises_file_not_found():
    """Test get_loader_for_folder raises FileNotFoundError with no files present."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdir = Path(tmpdirname)

        folder_loaders = {
            ".jpk": load_jpk_folder,
            ".asd": load_asd_folder,
            ".spm": load_spm_folder,
        }

        with pytest.raises(FileNotFoundError):
            get_loader_for_folder(tmpdir, folder_loaders)


def test_get_loader_for_folder_returns_callable():
    """Test that a callable is returned when a supported file extension is detected."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdir = Path(tmpdirname)
        (tmpdir / "example.JPK").touch()

        folder_loaders = {
            ".jpk": load_jpk_folder,
            ".asd": load_asd_folder,
            ".spm": load_spm_folder,
        }

        ext, loader = get_loader_for_folder(tmpdir, folder_loaders)
        assert ext.lower() == ".jpk"
        assert callable(loader)


def test_get_loader_for_folder_ignores_directories():
    """Test that get_loader_for_folder ignores subdirectories when finding loaders."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdir = Path(tmpdirname)
        (tmpdir / "file.spm").touch()
        (tmpdir / "subfolder").mkdir()

        folder_loaders = {
            ".jpk": load_jpk_folder,
            ".asd": load_asd_folder,
            ".spm": load_spm_folder,
        }

        ext, loader = get_loader_for_folder(tmpdir, folder_loaders)
        assert ext == ".spm"
        assert callable(loader)


def test_get_loader_for_folder_case_insensitive():
    """Test get_loader_for_folder detects file extensions in a case-insensitive way."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdir = Path(tmpdirname)
        (tmpdir / "file1.JpK").touch()
        (tmpdir / "file2.AsD").touch()

        folder_loaders = {
            ".jpk": load_jpk_folder,
            ".asd": load_asd_folder,
            ".spm": load_spm_folder,
        }

        ext, loader = get_loader_for_folder(tmpdir, folder_loaders)
        assert ext.lower() == ".jpk"
        assert callable(loader)


def test_get_loader_for_file_known_extension():
    """Test get_loader_for_file returns correct loader for supported file extension."""
    fake_path = Path("/fake/path/sample.h5-jpk")

    file_loaders = {
        ".h5-jpk": load_h5jpk,
    }
    folder_loaders = {
        ".jpk": load_jpk_folder,
        ".asd": load_asd_folder,
        ".spm": load_spm_folder,
    }

    loader = get_loader_for_file(fake_path, file_loaders, folder_loaders)
    assert callable(loader)
    assert loader == load_h5jpk


def test_get_loader_for_file_folder_extension_raises():
    """Test get_loader_for_file raises ValueError when given folder-like extensions."""
    fake_path = Path("/fake/path/sample.jpk")

    file_loaders = {
        ".h5-jpk": load_h5jpk,
    }
    folder_loaders = {
        ".jpk": load_jpk_folder,
    }

    with pytest.raises(ValueError, match="typically a single-frame export"):
        get_loader_for_file(fake_path, file_loaders, folder_loaders)


def test_get_loader_for_file_unknown_extension_raises():
    """Test get_loader_for_file raises ValueError for unsupported file extensions."""
    fake_path = Path("/fake/path/sample.unknown")

    file_loaders = {
        ".h5-jpk": load_h5jpk,
    }
    folder_loaders = {
        ".jpk": load_jpk_folder,
    }

    with pytest.raises(ValueError, match="Unsupported file type"):
        get_loader_for_file(fake_path, file_loaders, folder_loaders)


def test_load_afm_stack_folder_calls_correct_loader(tmp_path):
    """Test load_afm_stack calls folder loader and returns valid AFMImageStack."""
    dummy_file = tmp_path / "frame1.jpk"
    dummy_file.touch()

    dummy_image_stack = np.zeros((1, 10, 10))
    mock_stack = AFMImageStack(
        image_stack=dummy_image_stack,
        pixel_size_nm=1.0,
        img_shape=(10, 10),
        line_rate=1.0,
        channel="height_trace",
        file_path=str(tmp_path),
        frame_metadata=[{}],
    )

    with patch(
        "playNano.io.loader.load_jpk_folder", return_value=mock_stack
    ) as mock_loader:
        result = load_afm_stack(tmp_path)

        mock_loader.assert_called_once_with(tmp_path, channel="height_trace")
        assert isinstance(result, AFMImageStack)
        assert result.image_stack.shape == (1, 10, 10)


def test_normalize_to_uint8_handles_nan_and_constant_range():
    """
    Test normalize_to_uint8 handles NaNs, infinities, and constant images correctly.

    This checks that:
    - NaNs and infinite values are replaced with 0.
    - Constant images return zero-valued arrays of dtype uint8.
    - Values are scaled correctly between 0 and 255.
    """
    # NaN and constant image
    image_nan = np.full((5, 5), np.nan)
    image_inf = np.full((5, 5), np.inf)
    image_const = np.ones((5, 5)) * 42
    image_varied = np.array([[0.0, 1.0], [2.0, 3.0]])

    assert np.all(normalize_to_uint8(image_nan) == 0)
    assert np.all(normalize_to_uint8(image_inf) == 0)
    assert np.all(normalize_to_uint8(image_const) == 0)
    result = normalize_to_uint8(image_varied)
    assert result.dtype == np.uint8
    assert result.min() == 0
    assert result.max() == 255


def test_create_gif_with_scale_and_timestamp_outputs_gif(tmp_path):
    """
    Test create_gif_with_scale_and_timestamp creates a valid animated GIF file.

    This verifies that:
    - A GIF file is created at the given path.
    - The number of frames in the GIF matches the number of input frames.
    - The image content is RGB and has expected size.
    """
    # Create a 3-frame dummy image stack
    stack = np.random.rand(3, 10, 10)
    timestamps = [0.0, 1.0, 2.0]
    output_path = tmp_path / "test_output.gif"

    create_gif_with_scale_and_timestamp(
        image_stack=stack,
        pixel_size_nm=1.0,
        timestamps=timestamps,
        scale_bar_length_nm=5,
        output_path=output_path,
        duration=0.2,
        cmap_name="afmhot",
    )

    assert output_path.exists()

    # Check frame count and image size
    with Image.open(output_path) as img:
        frames = [frame.copy() for frame in ImageSequence.Iterator(img)]
        assert len(frames) == 3
        assert all(f.size == (10, 10) for f in frames)
        assert all(f.mode in ("P", "RGB", "RGBA") for f in frames)  # Flexible for GIFs
