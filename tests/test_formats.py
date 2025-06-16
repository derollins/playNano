"""Test for loading various file types."""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest

from playNano.afm_stack import AFMImageStack
from playNano.io.formats.read_asd import _standardize_units_to_nm, load_asd_file
from playNano.io.formats.read_h5jpk import (
    _get_z_unit_h5,
    _guess_and_standardize_units_to_nm,
    apply_z_unit_conversion,
    load_h5jpk,
)
from playNano.io.formats.read_jpk_folder import load_jpk_folder
from playNano.io.loader import get_loader_for_folder


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
        data=np.zeros((1, 5, 5)),
        pixel_size_nm=1.0,
        channel="height_trace",
        file_path=Path(test_file),
        frame_metadata=[{}],
    )

    with patch(
        "playNano.io.loader.load_h5jpk",
        return_value=dummy_stack,
    ) as mock_loader:
        result = AFMImageStack.load_data(test_file)

        mock_loader.assert_called_once_with(test_file, channel="height_trace")
        assert isinstance(result, AFMImageStack)
        assert result.data.shape == (1, 5, 5)


@pytest.mark.parametrize(
    "filename, expected_ext, loader_func_name",
    [
        ("example.JPK", ".jpk", "load_jpk_folder"),
        ("file1.JpK", ".jpk", "load_jpk_folder"),
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
        data=np.zeros((1, 5, 5)),
        pixel_size_nm=1.0,
        channel="height_trace",
        file_path=str(tmp_path),
        frame_metadata=[{}],
    )

    patch_path = f"playNano.io.loader.{loader_func_name}"

    with patch(patch_path, return_value=dummy_stack) as mock_loader:
        result = AFMImageStack.load_data(tmp_path)
        mock_loader.assert_called_once_with(tmp_path, channel="height_trace")
        assert isinstance(result, AFMImageStack)

    folder_loaders = {
        ".jpk": lambda p: None,
        ".spm": lambda p: None,
    }
    detected_ext, _ = get_loader_for_folder(tmp_path, folder_loaders)
    assert detected_ext.lower() == expected_ext


def test_load_data_with_multiple_files(tmp_path):
    """
    Test `AFMImageStack.load_data()` loads supported files if mixed ext are present.

    Ensures:
    - The loader is selected correctly even with unrelated files in the folder.
    """
    (tmp_path / "data1.txt").touch()
    (tmp_path / "data2.JPK").touch()
    (tmp_path / "readme.md").touch()

    dummy_stack = AFMImageStack(
        data=np.zeros((1, 5, 5)),
        pixel_size_nm=1.0,
        channel="height_trace",
        file_path=str(tmp_path),
        frame_metadata=[{}],
    )

    with patch(
        "playNano.io.loader.load_jpk_folder", return_value=dummy_stack
    ) as mock_loader:
        result = AFMImageStack.load_data(tmp_path)
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
        AFMImageStack.load_data(tmp_path)


def test_load_afm_stack_raises_with_unknown_extension_file(tmp_path):
    """
    Test that `load_afm_stack()` raises ValueError when an unsupported file is passed.

    Ensures:
    - File-based validation works for bad extensions (e.g. .unknown).
    """
    test_file = tmp_path / "sample.unknown"
    test_file.touch()

    with pytest.raises(ValueError, match="Unsupported file type: .unknown"):
        AFMImageStack.load_data(test_file)


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


def test_open_file(resource_path):
    """Test if the file can be read."""
    with h5py.File(resource_path / "sample_0.h5-jpk", "r") as f:
        assert list(f.keys())  # Just trigger reading


def test_h5jpk_file_is_hdf5(resource_path):
    """Check if the file is a valid HDF5 file before opening."""
    file_path = resource_path / "sample_0.h5-jpk"

    assert file_path.exists(), f"File does not exist: {file_path}"
    assert h5py.is_hdf5(file_path), f"File is not a valid HDF5 file: {file_path}"


def test_h5jpk_file_is_valid(resource_path):
    """Safely check if a .h5-jpk file is a valid HDF5 file."""
    file_path = resource_path / "sample_0.h5-jpk"  # Adjust to your test file
    try:
        with h5py.File(file_path, "r") as f:
            assert isinstance(f, h5py.File)
            assert len(f.keys()) > 0  # Ensure it has some content
    except OSError as e:
        pytest.fail(f"Failed to open HDF5 file: {e}")


@pytest.mark.parametrize(
    (
        "file_name",
        "channel",
        "flip_image",
        "pixel_to_nm_scaling",
        "stack_shape",
        "image_dtype",
        "metadata_dtype",
        "stack_sum",
    ),
    [
        pytest.param(
            "sample_0.h5-jpk",
            "height_trace",
            True,
            1.171875,
            (4, 128, 128),
            float,
            dict,
            48525583.047271535,
            id="test image 0",
        )
    ],
)
def test_read_h5jpk_valid_file(
    file_name: str,
    channel: str,
    flip_image: bool,
    pixel_to_nm_scaling: float,
    stack_shape: tuple[int, int, int],
    image_dtype: type[np.floating],
    metadata_dtype: type,
    stack_sum: float,
    resource_path: Path,
) -> None:
    """Test the normal operation of loading a .h5-jpk file."""
    result = load_h5jpk(resource_path / file_name, channel, flip_image)

    assert isinstance(result, AFMImageStack)
    assert result.pixel_size_nm == pixel_to_nm_scaling
    assert isinstance(result.data, np.ndarray)
    assert result.data.shape == stack_shape
    assert result.data.dtype == np.dtype(image_dtype)
    assert isinstance(result.frame_metadata, list)
    assert all(isinstance(frame, metadata_dtype) for frame in result.frame_metadata)
    assert result.data.sum() == stack_sum
    assert len(result.frame_metadata) == result.data.shape[0]


def test_get_loader_for_folder_no_valid_files(tmp_path):
    """Test to raise FileNotFoundError when no supported files are present."""
    (tmp_path / "file.txt").touch()
    folder_loaders = {".jpk": lambda p: None}
    with pytest.raises(FileNotFoundError):
        get_loader_for_folder(tmp_path, folder_loaders)


@pytest.mark.parametrize(
    (
        "file_name",
        "channel",
        "pixel_to_nm_scaling",
        "stack_shape",
        "image_dtype",
        "metadata_dtype",
        "stack_sum",
    ),
    [
        pytest.param(
            "asd_sample_0.asd",
            "TP",
            0.5,
            (32, 200, 200),
            float,
            dict,
            -251179816.91781396,
            id="test image 0",
        )
    ],
)
def test_read_asd_valid_file(
    file_name: str,
    channel: str,
    pixel_to_nm_scaling: float,
    stack_shape: tuple[int, int, int],
    image_dtype: type[np.floating],
    metadata_dtype: type,
    stack_sum: float,
    resource_path: Path,
) -> None:
    """Test the normal operation of loading a .h5-jpk file."""
    result = load_asd_file(resource_path / file_name, channel)

    assert isinstance(result, AFMImageStack)
    assert result.pixel_size_nm == pixel_to_nm_scaling
    assert isinstance(result.data, np.ndarray)
    assert result.data.shape == stack_shape
    assert result.data.dtype == np.dtype(image_dtype)
    assert isinstance(result.frame_metadata, list)
    assert all(isinstance(frame, metadata_dtype) for frame in result.frame_metadata)
    assert result.data.sum() == stack_sum
    assert len(result.frame_metadata) == result.data.shape[0]


class TestStandardizeUnitsToNM(unittest.TestCase):
    """Tests ofr the standardisation of units to nm in the asd reader."""

    def test_pm_conversion(self):
        """Test if input is in pm, range is 100000 → guessed unit 'pm'."""
        data = np.array([[100000, 200000]])  # pm
        expected = np.array([[100.0, 200.0]])  # in nm
        result = _standardize_units_to_nm(data.copy(), "TP")
        np.testing.assert_allclose(result, expected)

    def test_um_conversion(self):
        """Test if input has range 2e-5 → guessed unit 'um'."""
        data = np.array([[0.0, 2e-4]])  # um
        expected = np.array([[0.0, 0.2]])  # in nm
        result = _standardize_units_to_nm(data.copy(), "TP")
        np.testing.assert_allclose(result, expected)

    def test_ignore_non_topography_channel(self):
        """Test that non-topography channel aren't converted."""
        data = np.array([[1.0, 2.0]])
        result = _standardize_units_to_nm(data.copy(), "CP")
        np.testing.assert_array_equal(result, data)

    def test_fallback_to_nm_on_invalid_data(self):
        """Test that if no unit is guessed nm is assumed."""
        data = np.array([[np.nan, np.nan]])
        result = _standardize_units_to_nm(data.copy(), "TP")
        self.assertTrue(np.all(np.isnan(result)))

    def test_returns_same_array(self):
        """Test that when sata has a range of 1 the same array is returned."""
        data = np.array([[1.0, 2.0]])
        result = _standardize_units_to_nm(data, "TP")
        self.assertIs(result, data)


@pytest.mark.parametrize(
    (
        "folder_name",
        "channel",
        "flip_image",
        "pixel_to_nm_scaling",
        "stack_shape",
        "image_dtype",
        "metadata_dtype",
        "stack_sum",
    ),
    [
        pytest.param(
            "jpk_folder_0",
            "height_trace",
            True,
            1.953125,
            (3, 512, 512),
            float,
            dict,
            304613162.9259033,
            id="test image 0",
        )
    ],
)
def test_read_jpk_valid_files(
    folder_name: str,
    channel: str,
    flip_image: bool,
    pixel_to_nm_scaling: float,
    stack_shape: tuple[int, int, int],
    image_dtype: type[np.floating],
    metadata_dtype: type,
    stack_sum: float,
    resource_path: Path,
) -> None:
    """Test the normal operation of loading a .h5-jpk file."""
    result = load_jpk_folder(resource_path / folder_name, channel, flip_image)

    assert isinstance(result, AFMImageStack)
    assert result.pixel_size_nm == pixel_to_nm_scaling
    assert isinstance(result.data, np.ndarray)
    assert result.data.shape == stack_shape
    assert result.data.dtype == np.dtype(image_dtype)
    assert isinstance(result.frame_metadata, list)
    assert all(isinstance(frame, metadata_dtype) for frame in result.frame_metadata)
    assert result.data.sum() == stack_sum
    assert len(result.frame_metadata) == result.data.shape[0]


class TestGetZUnitH5(unittest.TestCase):
    """Tests for the `_get_z_unit_h5` helper function."""

    def test_returns_unit_string(self):
        """Should return the unit string from group attributes."""
        mock_group = MagicMock()
        mock_group.attrs.get.return_value = "nm"
        self.assertEqual(_get_z_unit_h5(mock_group), "nm")

    def test_returns_numeric_unit_as_string(self):
        """Should return numeric unit converted to string."""
        mock_group = MagicMock()
        mock_group.attrs.get.return_value = 1.0
        self.assertEqual(_get_z_unit_h5(mock_group), "1.0")

    def test_returns_none_on_missing_attr(self):
        """Should return None if attribute access fails."""
        mock_group = MagicMock()
        mock_group.attrs.get.side_effect = Exception("broken")
        self.assertIsNone(_get_z_unit_h5(mock_group))


class TestGuessAndStandardizeUnitsToNM(unittest.TestCase):
    """Tests for `_guess_and_standardize_units_to_nm` conversion logic."""

    def test_converts_pm_to_nm(self):
        """Should convert picometer input to nanometers."""
        data = np.array([[100000.0, 200000.0]])  # pm → expect [100.0, 200.0] nm
        expected = np.array([[100.0, 200.0]])
        result = _guess_and_standardize_units_to_nm(data.copy())
        np.testing.assert_allclose(result, expected)

    def test_converts_um_to_nm(self):
        """Should convert micrometer input to nanometers."""
        data = np.array([[0.0, 2e-4]])  # um → expect [1000.0, 2000.0] nm
        expected = np.array([[0.0, 0.2]])
        result = _guess_and_standardize_units_to_nm(data.copy())
        np.testing.assert_allclose(result, expected)

    def test_handles_nan_only_data(self):
        """Should leave NaN-only data unchanged."""
        data = np.array([[np.nan, np.nan]])
        result = _guess_and_standardize_units_to_nm(data.copy())
        self.assertTrue(np.all(np.isnan(result)))

    def test_returns_same_array_instance(self):
        """Should perform in-place modification of the original array."""
        data = np.array([[1.0, 2.0]])
        result = _guess_and_standardize_units_to_nm(data)
        self.assertIs(result, data)


class TestZUnitBlock:
    """Tests for apply_z_unit_conversion."""

    @patch("playNano.io.formats.read_h5jpk._get_z_unit_h5", return_value="um")
    @patch(
        "playNano.io.formats.read_h5jpk.convert_height_units_to_nm",
        side_effect=lambda img, unit: img * 1000,
    )
    def test_known_unit_conversion(self, mock_convert, mock_get_unit):
        """Should convert units like 'um' to nm."""
        images = np.array([[1e-3, 2e-3]])
        channel_group = MagicMock()

        result = apply_z_unit_conversion(images.copy(), channel_group)

        np.testing.assert_allclose(result, [[1.0, 2.0]])

        mock_convert.assert_called_once()
        called_args, _ = mock_convert.call_args
        np.testing.assert_allclose(called_args[0], np.array([[1e-3, 2e-3]]))
        assert called_args[1] == "um"

    @patch("playNano.io.formats.read_h5jpk._get_z_unit_h5", return_value="deg")
    def test_passthrough_for_non_scaled_units(self, mock_get_unit):
        """Should not modify images if unit is in ['V', 'v', 'deg']."""
        images = np.array([[0.3, 0.7]])
        channel_group = MagicMock()

        result = apply_z_unit_conversion(images.copy(), channel_group)

        np.testing.assert_array_equal(result, images)

    @patch("playNano.io.formats.read_h5jpk._get_z_unit_h5", return_value=None)
    @patch(
        "playNano.io.formats.read_h5jpk._guess_and_standardize_units_to_nm",
        side_effect=lambda img: img * 1e9,
    )
    def test_fallback_to_guessing(self, mock_guess, mock_get_unit):
        """Should guess and convert if no unit is present."""
        images = np.array([[1e-9, 2e-9]])
        channel_group = MagicMock()

        result = apply_z_unit_conversion(images.copy(), channel_group)

        np.testing.assert_allclose(result, [[1.0, 2.0]])
        mock_guess.assert_called_once()
