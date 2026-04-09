"""Test for loading various file types."""

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest

from playnano.afm_stack import AFMImageStack
from playnano.io.formats.read_aris import (
    _aris_global_pixel_to_nm_scaling_h5,
    _get_channel_names,
    _get_sorted_frame_keys,
    load_aris,
    load_frames_and_scaling,
)
from playnano.io.formats.read_asd import _standardize_units_to_nm, load_asd_file
from playnano.io.formats.read_h5jpk import (
    _get_z_scaling_h5,
    _get_z_unit_h5,
    _guess_and_standardize_units_to_nm,
    apply_z_unit_conversion,
    load_h5jpk,
)
from playnano.io.formats.read_jpk_folder import load_jpk_folder
from playnano.io.formats.read_nhf_folder import (
    get_image_number,
    get_nhf_time,
    load_nhf_folder,
)
from playnano.io.formats.read_spm_folder import load_spm_folder, parse_spm_header
from playnano.io.loader import get_loader_for_file, get_loader_for_folder


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
        "playnano.io.loader.load_h5jpk",
        return_value=dummy_stack,
    ) as mock_loader:
        mock_loader.__name__ = "load_h5jpk_file"
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

    patch_path = f"playnano.io.loader.{loader_func_name}"

    with patch(patch_path, return_value=dummy_stack) as mock_loader:
        mock_loader.__name__ = loader_func_name
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
        "playnano.io.loader.load_jpk_folder", return_value=dummy_stack
    ) as mock_loader:
        mock_loader.__name__ = "load_jpk_folder"
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
        ".spm": lambda p: None,
    }

    ext, loader = get_loader_for_folder(tmp_path, folder_loaders)
    assert ext.lower() == ".jpk"
    assert callable(loader)


def dummy_file_loader():
    """Make a dummy file loader for testing."""
    pass


def test_returns_correct_loader():
    """Test that get_loader_for_file returns correct file loader."""
    file_loaders = {".txt": dummy_file_loader}
    folder_loaders = {}

    ext, loader = get_loader_for_file(Path("test.txt"), file_loaders, folder_loaders)

    assert ext == ".txt"
    assert loader is dummy_file_loader


def test_file_loader_raises_for_folder_loader_extension():
    """Test that get_loader_for_file rises ValueError if passed folder loader file."""
    file_loaders = {}
    folder_loaders = {".hsa": "placeholder"}

    with pytest.raises(ValueError) as excinfo:
        get_loader_for_file(Path("frame.hsa"), file_loaders, folder_loaders)

    assert "pass the full folder instead" in str(excinfo.value)


def test_get_loader_for_file_raises_for_unsupported_extension():
    """Test that get_loader_for_file raises a ValueError for an invalid extension."""
    file_loaders = {}
    folder_loaders = {}

    with pytest.raises(ValueError) as excinfo:
        get_loader_for_file(Path("data.weird"), file_loaders, folder_loaders)

    assert "Unsupported file type" in str(excinfo.value)


def test_get_loader_for_file_handles_multi_suffix():
    """Test that get_loader_for_file handles multi suffix extentions."""
    file_loaders = {".ome.tif": dummy_file_loader}
    folder_loaders = {}
    file = "test.ome.tif"
    ext, loader = get_loader_for_file(Path(file), file_loaders, folder_loaders)

    assert ext == ".ome.tif"
    assert loader is dummy_file_loader


def test_raises_for_missing_extension():
    """Test that get_loader_for_file raises a ValueError for missing extension."""
    file_loaders = {}
    folder_loaders = {}

    with pytest.raises(ValueError) as excinfo:
        get_loader_for_file(Path("noextension"), file_loaders, folder_loaders)

    assert "has no extension" in str(excinfo.value)


@pytest.mark.parametrize(
    "ext",
    [
        ".h5-jpk",
        ".asd",
        ".aris",
        ".npz",
        ".h5",
        ".tif",
        ".ome.tif",
        ".tiff",
        ".ome.tiff",
    ],
)
def test_get_loader_for_file_valid_all_formats(ext):
    """Test that all file loaders open valid extensions."""

    # Create a different dummy loader per extension
    def dummy_loader():
        """Make a dummy loader for testing."""
        return f"loaded {ext}"

    file_loaders = {ext: dummy_loader}
    folder_loaders = {}

    # Simulate input file in lowercase
    file_path = Path(f"test{ext}")

    returned_ext, returned_loader = get_loader_for_file(
        file_path, file_loaders, folder_loaders
    )

    assert returned_ext == ext
    assert returned_loader is dummy_loader


def test_get_loader_for_file_folder_conflict():
    """Test that get_loader_for_file raises a ValueError when given a file format."""
    file_loaders = {}
    folder_loaders = {".jpk": dummy_file_loader}

    with pytest.raises(ValueError):
        get_loader_for_file(Path("frame.jpk"), file_loaders, folder_loaders)


def test_get_loader_for_file_unsupported():
    """Test that get_loader_for_file raises a ValueError for unknown extension."""
    file_loaders = {}
    folder_loaders = {}

    with pytest.raises(ValueError):
        get_loader_for_file(Path("data.unknown"), file_loaders, folder_loaders)


def test_get_loader_for_file_no_extension():
    """Test that get_loader_for_file raises a ValueError for no extension."""
    with pytest.raises(ValueError):
        get_loader_for_file(Path("noext"), {}, {})


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


@pytest.fixture
def h5_file_missing_scaling(tmp_path):
    """Create a test hdf5 file without scaling attributes for testing."""
    file_path = tmp_path / "test_missing_scaling.h5"
    with h5py.File(file_path, "w") as f:
        f.create_group("channel")
    return h5py.File(file_path, "r")


def test_get_z_scaling_logs_warning(caplog, h5_file_missing_scaling):
    """Test _get_z_scaling logs warnings if scaling attributes are not in h5-jpk."""
    grp = h5_file_missing_scaling["channel"]

    with caplog.at_level("WARNING"):
        multiplier, offset = _get_z_scaling_h5(grp)

    assert multiplier == 1.0
    assert offset == 0.0

    # Check that both warnings were logged
    assert "Missing attribute 'net-encoder.scaling.multiplier'" in caplog.text
    assert "Missing attribute 'net-encoder.scaling.offset'" in caplog.text


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
    """Test the normal operation of loading a .asd file."""
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
    """Test the normal operation of loading a .jpk folder."""
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

    @patch("playnano.io.formats.read_h5jpk._get_z_unit_h5", return_value="um")
    @patch(
        "playnano.io.formats.read_h5jpk.convert_height_units_to_nm",
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

    @patch("playnano.io.formats.read_h5jpk._get_z_unit_h5", return_value="deg")
    def test_passthrough_for_non_scaled_units(self, mock_get_unit):
        """Should not modify images if unit is in ['V', 'v', 'deg']."""
        images = np.array([[0.3, 0.7]])
        channel_group = MagicMock()

        result = apply_z_unit_conversion(images.copy(), channel_group)

        np.testing.assert_array_equal(result, images)

    @patch("playnano.io.formats.read_h5jpk._get_z_unit_h5", return_value=None)
    @patch(
        "playnano.io.formats.read_h5jpk._guess_and_standardize_units_to_nm",
        side_effect=lambda img: img * 1e9,
    )
    def test_fallback_to_guessing(self, mock_guess, mock_get_unit):
        """Should guess and convert if no unit is present."""
        images = np.array([[1e-9, 2e-9]])
        channel_group = MagicMock()

        result = apply_z_unit_conversion(images.copy(), channel_group)

        np.testing.assert_allclose(result, [[1.0, 2.0]])
        mock_guess.assert_called_once()


def test_parse_spm_header_skips_malformed_lines():
    """Test that `parse_spm_header` skips malformed header lines."""
    # Create a temporary file with malformed and valid header lines
    malformed_header = (
        "\\Scan Rate 2.0\n"  # Missing colon (malformed)
        "\\Scan Size: 1.0\n"  # Valid
        "\\AnotherMalformed\n"  # Also malformed
        "\\Valid: entry\n"  # Valid
    )

    with tempfile.NamedTemporaryFile("w+b", delete=False) as temp:
        temp.write(malformed_header.encode("latin1"))
        temp_path = Path(temp.name)

    # Run parser
    header = parse_spm_header(temp_path)

    # Check that only the valid lines were included
    assert header == {
        "Scan Size": "1.0",
        "Valid": "entry",
    }

    # Clean up temp file
    temp_path.unlink()


def test_load_spm_folder_raises_if_not_directory(tmp_path):
    """Test that `load_spm_folder` raises ValueError if the path is not a directory."""
    fake_file = tmp_path / "not_a_dir.txt"
    fake_file.write_text("I'm not a folder")

    with pytest.raises(ValueError, match="is not a directory"):
        load_spm_folder(fake_file, channel="height")


def test_load_spm_folder_raises_if_no_spm_files(tmp_path):
    """Test `load_spm_folder` raises FileNotFoundError if no .spm files are found."""
    # Add unrelated file
    (tmp_path / "something.txt").write_text("Not an spm file")

    with pytest.raises(FileNotFoundError, match="No .spm files found"):
        load_spm_folder(tmp_path, channel="height")


@patch("playnano.io.formats.read_spm_folder.spm.load_spm")
@patch("playnano.io.formats.read_spm_folder.parse_spm_header")
def test_load_spm_folder_missing_line_rate_raises(
    mock_parse_header, mock_load_spm, tmp_path
):
    """Test `load_spm_folder` raises ValueError if 'Scan Rate' is missing in header."""
    dummy_file = tmp_path / "frame1.spm"
    dummy_file.write_text("placeholder")

    # Mock image loading: valid shape and pixel size
    mock_load_spm.return_value = (np.ones((10, 10)), 1.0)

    # Simulate missing "Scan Rate" in header
    mock_parse_header.return_value = {}

    with pytest.raises(ValueError, match="Missing data: line_rate=None"):
        load_spm_folder(tmp_path, channel="height")


@patch("playnano.io.formats.read_spm_folder.spm.load_spm")
@patch("playnano.io.formats.read_spm_folder.parse_spm_header")
def test_load_spm_folder_inconsistent_shape_raises(
    mock_parse_header, mock_load_spm, tmp_path
):
    """Test `load_spm_folder` raises ValueError for inconsistent image shapes."""
    # Create two dummy .spm files
    f1 = tmp_path / "frame1.spm"
    f2 = tmp_path / "frame2.spm"
    f1.write_text("placeholder")
    f2.write_text("placeholder")

    # First image has 10x10, second has 8x8
    mock_load_spm.side_effect = [
        (np.ones((10, 10)), 1.0),
        (np.ones((8, 8)), 1.0),
    ]
    mock_parse_header.return_value = {"Scan Rate": "10"}

    with pytest.raises(ValueError, match="Inconsistent image shape"):
        load_spm_folder(tmp_path, channel="height")


@pytest.mark.parametrize(
    (
        "folder_name",
        "channel",
        "pixel_to_nm_scaling",
        "stack_shape",
        "image_dtype",
        "metadata_dtype",
        "stack_sum",
    ),
    [
        pytest.param(
            "spm_folder_0",
            "Height",
            1.953125,
            (4, 256, 512),
            float,
            dict,
            -78983151.45184162,
            id="test image 0",
        )
    ],
)
def test_read_spm_valid_files(
    folder_name: str,
    channel: str,
    pixel_to_nm_scaling: float,
    stack_shape: tuple[int, int, int],
    image_dtype: type[np.floating],
    metadata_dtype: type,
    stack_sum: float,
    resource_path: Path,
) -> None:
    """Test the normal operation of loading a .spm folder."""
    result = load_spm_folder(resource_path / folder_name, channel)

    assert isinstance(result, AFMImageStack)
    assert result.pixel_size_nm == pixel_to_nm_scaling
    assert isinstance(result.data, np.ndarray)
    assert result.data.shape == stack_shape
    assert result.data.dtype == np.dtype(image_dtype)
    assert isinstance(result.frame_metadata, list)
    assert all(isinstance(frame, metadata_dtype) for frame in result.frame_metadata)
    assert result.data.sum() == stack_sum
    assert len(result.frame_metadata) == result.data.shape[0]


# --- ARIS loader tests ---


def create_aris_file_on_disk(path: Path, num_frames=3, override_second_frame=False):
    """
    Create an in-memory ARIS-like HDF5 file for testing.

    Do not have .ARIS  test data, so create some.

    Parameters
    ----------
    num_frames : int, optional
        Number of synthetic frames to generate (default: 3).
    override_second_frame : bool, optional
        If True, applies a per-frame pixel-scaling override to Frame 1
        to test per-frame scan parameter handling.

    Returns
    -------
    h5py.File
        An in-memory HDF5 file configured like an ARIS dataset.
    """

    with h5py.File(path, "w") as f:
        # --- DataSet structure ---
        data_group = f.create_group("/DataSet/Resolution 0")
        for i in range(num_frames):
            frame_g = data_group.create_group(f"Frame {i}")
            ch_g = frame_g.create_group("height_trace")
            ch_g.create_dataset("Image", data=np.full((4, 4), i, float))

        # --- DataSetInfo ---
        info = f.create_group("/DataSetInfo")
        info.attrs["ChannelNames"] = [b"height_trace"]

        # Global params
        gscan = info.create_group("Global/Parameters/Scan")
        gscan.attrs["FastScanSize"] = 2e-6
        gscan.attrs["ScanPoints"] = 256
        gscan.attrs["ScanRate"] = 350.0

        frames = info.create_group("Frames")

        for i in range(num_frames):
            p = frames.create_group(f"Frame {i}").create_group("Parameters/Scan")
            p.attrs["FastScanSize"] = 2e-6
            p.attrs["ScanPoints"] = 256

        if override_second_frame and num_frames > 1:
            p2 = frames["Frame 1"]["Parameters/Scan"]
            p2.attrs["FastScanSize"] = 1e-6
            p2.attrs["ScanPoints"] = 256

        series = info.create_group("Series")
        series.create_dataset("Time", data=np.arange(num_frames))

    return path


def test_get_channel_names(tmp_path):
    """Test extraction of ARIS channel names."""
    file_path = tmp_path / "test.aris"
    create_aris_file_on_disk(file_path)

    with h5py.File(file_path, "r") as f:
        info = f["/DataSetInfo"]
        channels = _get_channel_names(info)

    assert channels == ["height_trace"]


def test_global_pixel_scaling(tmp_path):
    """Test computing global pixel size in nanometres."""
    file_path = tmp_path / "test.aris"
    create_aris_file_on_disk(file_path)

    with h5py.File(file_path, "r") as f:
        info = f["/DataSetInfo"]
        pixel_size = _aris_global_pixel_to_nm_scaling_h5(info)

    assert np.isclose(pixel_size, 7.8125)


def test_sorted_frame_keys(tmp_path):
    """Test numeric sorting of ARIS frame keys."""
    file_path = tmp_path / "test.aris"
    create_aris_file_on_disk(file_path)

    with h5py.File(file_path, "r") as f:
        data = f["/DataSet/Resolution 0"]
        keys = _get_sorted_frame_keys(data)

    assert keys == ["Frame 0", "Frame 1", "Frame 2"]


def test_load_frames_without_override(tmp_path):
    """Test frame loading when no per-frame scaling override is used."""
    file_path = tmp_path / "test.aris"
    create_aris_file_on_disk(file_path)

    with h5py.File(file_path, "r") as f:
        data = f["/DataSet/Resolution 0"]
        info = f["/DataSetInfo"]
        stack, pixel_sizes = load_frames_and_scaling(data, info, "height_trace")

    assert stack.shape == (3, 4, 4)
    assert np.all(stack[2] == 2)
    assert all(np.isclose(px, pixel_sizes[0]) for px in pixel_sizes)


def test_load_frames_with_override(tmp_path):
    """Test per-frame scaling override for Frame 1."""
    file_path = tmp_path / "test.aris"
    create_aris_file_on_disk(file_path, override_second_frame=True)

    with h5py.File(file_path, "r") as f:
        data = f["/DataSet/Resolution 0"]
        info = f["/DataSetInfo"]
        stack, pixel_sizes = load_frames_and_scaling(data, info, "height_trace")

    assert not np.isclose(pixel_sizes[0], pixel_sizes[1])


def test_load_frames_reverts_to_global_pixel_size(tmp_path):
    """Test per-frame scaling falls back to global scale when missing."""
    file_path = tmp_path / "missing_frame_pixel.aris"
    create_aris_file_on_disk(file_path)

    # Set a new global FastScanSize to verify fallback behaviour
    new_global_scan_size = 2e-7  # 0.2 µm
    expected_pixel_nm = (new_global_scan_size / 256) * 1e9

    with h5py.File(file_path, "a") as f:
        # Modify global scan size
        f["/DataSetInfo/Global/Parameters/Scan"].attrs[
            "FastScanSize"
        ] = new_global_scan_size

        # Remove per-frame scan metadata (so loader must fall back)
        for i in range(3):
            try:
                del f[f"/DataSetInfo/Frames/Frame {i}/Parameters/Scan"]
            except KeyError:
                pass  # already missing -> fine

        data = f["/DataSet/Resolution 0"]
        info = f["/DataSetInfo"]
        _, pixel_sizes = load_frames_and_scaling(data, info, "height_trace")

    # Validate fallback for all frames
    for px in pixel_sizes:
        assert np.isclose(px, expected_pixel_nm)


def test_load_aris_returns_correct_type(tmp_path):
    """Test that load_aris returns an AFMImageStack."""
    file_path = tmp_path / "test.aris"
    create_aris_file_on_disk(file_path)

    result = load_aris(file_path, "height_trace")

    assert isinstance(result, AFMImageStack)
    assert result.data.shape == (3, 4, 4)


def test_load_aris_raises_missing_channel(tmp_path):
    """Test error raised when loading a missing ARIS channel."""
    file_path = tmp_path / "test.aris"
    create_aris_file_on_disk(file_path)

    with pytest.raises(ValueError):
        load_aris(file_path, "nonexistent")


def test_load_aris_timestamp_mismatch(tmp_path):
    """Test error raised when timestamps do not match frame count."""
    file_path = tmp_path / "bad.aris"
    create_aris_file_on_disk(file_path)

    # Replace timestamps with too few values
    with h5py.File(file_path, "a") as f:
        del f["/DataSetInfo/Series/Time"]
        f["/DataSetInfo/Series"].create_dataset("Time", data=np.array([0.0, 1.0]))

    with pytest.raises(ValueError):
        load_aris(file_path, "height_trace")


# --- NHF loader tests ---


@pytest.mark.parametrize(
    (
        "folder_name",
        "channel",
        "pixel_to_nm_scaling",
        "stack_shape",
        "image_dtype",
        "metadata_dtype",
        "stack_sum",
    ),
    [
        pytest.param(
            "nhf_folder_0",
            "Topography_trace",
            10.154255319148936,
            (3, 500, 500),
            float,
            dict,
            908261278.2984858,
        )
    ],
)
def test_read_nhf_valid_files(
    folder_name: str,
    channel: str,
    pixel_to_nm_scaling: float,
    stack_shape: tuple[int, int, int],
    image_dtype: type[np.floating],
    metadata_dtype: type,
    stack_sum: float,
    resource_path: Path,
) -> None:
    """Test the normal operation of loading a .nhf folder."""
    nhf_result = load_nhf_folder(resource_path / folder_name, channel)

    assert isinstance(nhf_result, AFMImageStack)
    assert nhf_result.pixel_size_nm == pixel_to_nm_scaling
    assert isinstance(nhf_result.data, np.ndarray)
    assert nhf_result.data.shape == stack_shape
    assert nhf_result.data.dtype == np.dtype(image_dtype)
    assert isinstance(nhf_result.frame_metadata, list)
    assert all(isinstance(frame, metadata_dtype) for frame in nhf_result.frame_metadata)
    assert nhf_result.data.sum() == stack_sum
    assert len(nhf_result.frame_metadata) == nhf_result.data.shape[0]

    required_keys = {"timestamp", "frame_pixel_size_nm", "line_rate"}

    for frame in nhf_result.frame_metadata:
        assert required_keys.issubset(frame.keys())


@patch("playnano.io.formats.read_nhf_folder.get_nhf_time")
@patch("playnano.io.formats.read_nhf_folder.get_image_number")
@patch("playnano.io.formats.read_nhf_folder.load_nhf")
def test_load_nhf_folder_missing_line_rate_raises(
    mock_load_nhf,
    mock_get_image_number,
    mock_get_nhf_time,
    tmp_path,
):
    """Test that load_nhf_folder raises ValueError if line_rate is missing."""
    dummy_file = tmp_path / "frame1.nhf"
    dummy_file.write_text("placeholder")

    # Prevent file I/O
    mock_get_image_number.return_value = 0
    mock_get_nhf_time.return_value = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    # Missing line_rate
    mock_load_nhf.return_value = (
        np.ones((10, 10)),
        1.0,
        None,
    )

    with pytest.raises(ValueError, match="line_rate"):
        load_nhf_folder(tmp_path, channel="Topography_trace")


@patch("playnano.io.formats.read_nhf_folder.get_nhf_time")
@patch("playnano.io.formats.read_nhf_folder.get_image_number")
@patch("playnano.io.formats.read_nhf_folder.load_nhf")
def test_load_nhf_folder_missing_timestamp_raises(
    mock_load_nhf,
    mock_get_image_number,
    mock_get_nhf_time,
    tmp_path,
):
    """Test that load_nhf_folder raises ValueError if timestamp is missing."""
    dummy_file = tmp_path / "frame1.nhf"
    dummy_file.write_text("placeholder")

    mock_get_image_number.return_value = 0
    mock_get_nhf_time.return_value = None

    mock_load_nhf.return_value = (
        np.ones((10, 10)),
        1.0,
        1.0,
    )

    with pytest.raises((TypeError, AttributeError)):
        load_nhf_folder(tmp_path, channel="Topography_trace")


@patch("playnano.io.formats.read_nhf_folder.get_nhf_time")
@patch("playnano.io.formats.read_nhf_folder.get_image_number")
@patch("playnano.io.formats.read_nhf_folder.load_nhf")
def test_load_nhf_folder_backwards_timestamps_raise(
    mock_load_nhf,
    mock_get_image_number,
    mock_get_nhf_time,
    tmp_path,
):
    """Test that load_nhf_folder raises ValueError if timestamps are not monotonic."""
    # Two frames
    (tmp_path / "frame1.nhf").write_text("placeholder")
    (tmp_path / "frame2.nhf").write_text("placeholder")

    mock_get_image_number.side_effect = [0, 1]

    t0 = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    t1 = t0 - timedelta(seconds=5)  # ← backwards time

    mock_get_nhf_time.side_effect = [t0, t1]

    mock_load_nhf.return_value = (
        np.ones((10, 10)),
        1.0,
        1.0,
    )

    with pytest.raises(ValueError, match="not monotonic"):
        load_nhf_folder(tmp_path, channel="Topography_trace")


@patch("playnano.io.formats.read_nhf_folder.get_nhf_time")
@patch("playnano.io.formats.read_nhf_folder.get_image_number")
@patch("playnano.io.formats.read_nhf_folder.load_nhf")
def test_load_nhf_folder_repeated_timestamps_warn(
    mock_load_nhf,
    mock_get_image_number,
    mock_get_nhf_time,
    tmp_path,
    caplog,
):
    """Test that load_nhf_folder logs a warning if timestamps are repeated."""
    (tmp_path / "frame1.nhf").write_text("placeholder")
    (tmp_path / "frame2.nhf").write_text("placeholder")

    mock_get_image_number.side_effect = [0, 1]

    t = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    mock_get_nhf_time.side_effect = [t, t]  # ← repeated

    mock_load_nhf.return_value = (
        np.ones((10, 10)),
        1.0,
        1.0,
    )

    with caplog.at_level("WARNING"):
        load_nhf_folder(tmp_path, channel="Topography_trace")

    assert any("repeated values" in record.message for record in caplog.records)


class DummyMeasurement:
    def __init__(self, *, created=None, measurement_name=None):
        self.attribute = {}
        if created is not None:
            self.attribute["created"] = created
        if measurement_name is not None:
            self.attribute["measurement_name"] = measurement_name


class DummyNHFFile:
    def __init__(self, measurements: dict[str, DummyMeasurement]):
        self.measurement = measurements

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


@patch("playnano.io.formats.read_nhf_folder.nhf_reader.NHFFileReader")
def test_get_nhf_time_iso_z(mock_reader):
    created = "2026-02-01T12:54:32.990Z"
    mock_reader.return_value = DummyNHFFile(
        {"Image 1": DummyMeasurement(created=created)}
    )

    t = get_nhf_time(Path("dummy.nhf"))

    assert isinstance(t, datetime)
    assert t.isoformat() == "2026-02-01T12:54:32.990000+00:00"


@patch("playnano.io.formats.read_nhf_folder.nhf_reader.NHFFileReader")
def test_get_nhf_time_iso_no_z(mock_reader):
    created = "2026-02-01T12:54:32.990"
    mock_reader.return_value = DummyNHFFile(
        {"Image 1": DummyMeasurement(created=created)}
    )

    t = get_nhf_time(Path("dummy.nhf"))

    assert isinstance(t, datetime)
    assert t.isoformat() == "2026-02-01T12:54:32.990000"


@patch("playnano.io.formats.read_nhf_folder.nhf_reader.NHFFileReader")
def test_get_nhf_time_numpy_scalar(mock_reader):
    created = np.array("2026-02-01T12:54:32.990Z")
    mock_reader.return_value = DummyNHFFile(
        {"Image 1": DummyMeasurement(created=created)}
    )

    t = get_nhf_time(Path("dummy.nhf"))

    assert isinstance(t, datetime)
    assert t.tzinfo is not None


@patch("playnano.io.formats.read_nhf_folder.nhf_reader.NHFFileReader")
def test_get_nhf_time_bytes(mock_reader):
    created = b"2026-02-01T12:54:32.990Z"
    mock_reader.return_value = DummyNHFFile(
        {"Image 1": DummyMeasurement(created=created)}
    )

    t = get_nhf_time(Path("dummy.nhf"))

    assert isinstance(t, datetime)
    assert t.isoformat() == "2026-02-01T12:54:32.990000+00:00"


@pytest.mark.parametrize(
    "measurement_name, expected",
    [
        ("Image 1", 1),
        ("Image 12", 12),
        ("Topography 3", 3),
        ("Scan_007", 7),
        ("Measurement #42", 42),
    ],
)
@patch("playnano.io.formats.read_nhf_folder.nhf_reader.NHFFileReader")
def test_get_image_number_valid_names(
    mock_reader,
    measurement_name,
    expected,
):
    mock_reader.return_value = DummyNHFFile(
        {"only": DummyMeasurement(measurement_name=measurement_name)}
    )

    result = get_image_number(Path("dummy.nhf"))

    assert result == expected


@patch("playnano.io.formats.read_nhf_folder.nhf_reader.NHFFileReader")
def test_get_image_number_invalid_measurement_name_no_digits_raises(mock_reader):
    """
    measurement_name is a string but contains no digits,
    so get_image_number must raise ValueError.
    """
    mock_reader.return_value = DummyNHFFile(
        {"Image 1": DummyMeasurement(measurement_name="Topography")}  # ← no digits
    )

    with pytest.raises(ValueError, match="invalid measurement_name"):
        get_image_number(Path("dummy.nhf"))
