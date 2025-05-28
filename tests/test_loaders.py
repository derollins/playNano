"""Test for loading various file types."""

from unittest.mock import patch

import numpy as np
import pytest
from playNano.io.loader import get_loader_for_folder
from playNano.main import load_afm_stack
from playNano.stack.image_stack import AFMImageStack


def test_load_afm_stack_file_calls_correct_loader(tmp_path):
    """
    Test that `load_afm_stack()` calls h5-jpk loader when a .h5-jpk file is provided.

    Ensures:
    - The appropriate loader function is called.
    - The returned object is an instance of AFMImageStack.
    - The image stack has the expected shape.
    """
    test_file = tmp_path / "sample.h5-jpk"
    test_file.touch()

    dummy_stack = AFMImageStack(
        image_stack=np.zeros((1, 5, 5)),
        pixel_size_nm=1.0,
        img_shape=(5, 5),
        line_rate=1.0,
        channel="height_trace",
        file_path=str(test_file),
        frame_metadata=[{}],
    )

    with patch(
        "playNano.io.loader.load_h5jpk",
        return_value=dummy_stack,
    ) as mock_loader:
        result = load_afm_stack(test_file)

        mock_loader.assert_called_once_with(test_file, channel="height_trace")
        assert isinstance(result, AFMImageStack)
        assert result.image_stack.shape == (1, 5, 5)


@pytest.mark.parametrize(
    "filename, expected_ext, loader_func_name",
    [
        ("example.JPK", ".jpk", "load_jpk_folder"),
        ("file1.JpK", ".jpk", "load_jpk_folder"),
        ("file2.AsD", ".asd", "load_asd_folder"),
        ("file.spm", ".spm", "load_spm_folder"),
    ],
)
def test_load_afm_stack_file_calls_correct_folder_loader(
    tmp_path, filename, expected_ext, loader_func_name
):
    """
    Parametrized test that `load_afm_stack()` identifies the appropriate loader.

    Ensures:
    - Folders containing supported AFM file types with various capitalizations load.
    - The correct loader is called based on the file extension.
    - The returned object is an instance of AFMImageStack.
    - Extension detection in `get_loader_for_folder()` is case-insensitive.
    """
    (tmp_path / filename).touch()
    (tmp_path / "subfolder").mkdir()  # extra content to ensure robustness

    dummy_stack = AFMImageStack(
        image_stack=np.zeros((1, 5, 5)),
        pixel_size_nm=1.0,
        img_shape=(5, 5),
        line_rate=1.0,
        channel="height_trace",
        file_path=str(tmp_path),
        frame_metadata=[{}],
    )

    patch_path = f"playNano.io.loader.{loader_func_name}"

    with patch(patch_path, return_value=dummy_stack) as mock_loader:
        result = load_afm_stack(tmp_path)
        mock_loader.assert_called_once_with(tmp_path, channel="height_trace")
        assert isinstance(result, AFMImageStack)

    folder_loaders = {
        ".jpk": lambda p: None,
        ".asd": lambda p: None,
        ".spm": lambda p: None,
    }
    detected_ext, _ = get_loader_for_folder(tmp_path, folder_loaders)
    assert detected_ext.lower() == expected_ext


def test_load_afm_stack_with_multiple_files(tmp_path):
    """
    Test that `load_afm_stack()` loads supported files if mixed extensions are present.

    Ensures:
    - The loader is selected correctly even with unrelated files in the folder.
    """
    (tmp_path / "data1.txt").touch()
    (tmp_path / "data2.JPK").touch()
    (tmp_path / "readme.md").touch()

    dummy_stack = AFMImageStack(
        image_stack=np.zeros((1, 5, 5)),
        pixel_size_nm=1.0,
        img_shape=(5, 5),
        line_rate=1.0,
        channel="height_trace",
        file_path=str(tmp_path),
        frame_metadata=[{}],
    )

    with patch(
        "playNano.io.loader.load_jpk_folder", return_value=dummy_stack
    ) as mock_loader:
        result = load_afm_stack(tmp_path)
        mock_loader.assert_called_once_with(tmp_path, channel="height_trace")
        assert isinstance(result, AFMImageStack)


def test_load_afm_stack_raises_with_unknown_extension(tmp_path):
    """
    Test that `load_afm_stack()` raises FileNotFoundError.

    Ensures:
    - An appropriate exception is raised for unsupported folder contents.
    """
    (tmp_path / "file.unknown").touch()

    with pytest.raises(
        FileNotFoundError, match="No supported AFM files found in the folder."
    ):
        load_afm_stack(tmp_path)


def test_load_afm_stack_raises_with_unknown_extension_file(tmp_path):
    """
    Test that `load_afm_stack()` raises ValueError when an unsupported file is passed.

    Ensures:
    - File-based validation works for bad extensions (e.g. .unknown).
    """
    test_file = tmp_path / "sample.unknown"
    test_file.touch()

    with pytest.raises(ValueError, match="Unsupported file type: .unknown"):
        load_afm_stack(test_file)


def test_get_loader_for_folder_detects_extension(tmp_path):
    """
    Test that `get_loader_for_folder()` correctly detects file extensions in folders.

    Ensures:
    - The first valid extension found is returned.
    - Case-insensitivity in extension matching works as intended.
    """
    (tmp_path / "file1.JPK").touch()
    (tmp_path / "file2.txt").touch()

    folder_loaders = {
        ".jpk": lambda p: None,
        ".asd": lambda p: None,
    }

    ext, loader = get_loader_for_folder(tmp_path, folder_loaders)
    assert ext.lower() == ".jpk"
    assert callable(loader)
