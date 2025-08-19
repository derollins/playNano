"""Tests for the functions within io module."""

import json
import re
import tempfile
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import h5py
import jsonschema
import numpy as np
import pytest
import tifffile
from PIL import Image, ImageSequence

from playNano.afm_stack import AFMImageStack
from playNano.analysis.pipeline import AnalysisPipeline
from playNano.analysis.utils.common import (
    export_to_hdf5,
    make_json_safe,
    sanitize_analysis_for_logging,
)
from playNano.io.export_data import (
    check_path_is_path,
    export_bundles,
    save_h5_bundle,
    save_npz_bundle,
    save_ome_tiff_stack,
)
from playNano.io.formats.read_asd import load_asd_file
from playNano.io.formats.read_h5jpk import load_h5jpk
from playNano.io.formats.read_jpk_folder import load_jpk_folder
from playNano.io.formats.read_spm_folder import load_spm_folder
from playNano.io.gif_export import (
    create_gif_with_scale_and_timestamp,
    export_gif,
    normalize_to_uint8,
)
from playNano.io.loader import (
    get_loader_for_file,
    get_loader_for_folder,
    load_afm_stack,
)
from playNano.processing.pipeline import ProcessingPipeline


class DummyAFM:
    """A dummy AFMImageStack for testing purposes."""

    def __init__(self):
        """Initialize a dummy AFMImageStack for testing."""
        self.data = np.zeros((5, 10, 10))
        self.pixel_size_nm = 1.0
        self.frame_metadata = [{"timestamp": i} for i in range(5)]
        self.channel = "Height"
        self.file_path = Path("dummy.jpk")
        self.processed = {"raw": self.data}

    def apply(self):
        """Simulate processing with a dummy apply method."""
        return self.data + 1  # simulate filtered result

    @property
    def image_shape(self):
        """Return the shape of the image data."""
        return self.data.shape[1:]

    # Only needed if _export_* methods rely on specific attributes/methods
    def __getitem__(self, key):
        """Allow dict-like access to data."""
        return self.data  # fallback for dict-like access if used in your code


def test_get_loader_for_file_with_no_extension_raises(tmp_path):
    """Test that get_loader_for_file raises ValueError for files with no extension."""
    no_ext_file = tmp_path / "afm_stackfile"
    no_ext_file.write_text("dummy content")

    file_loaders = {".asd": lambda x: x, ".h5-jpk": lambda x: x}
    folder_loaders = {".jpk": lambda x: x, ".spm": lambda x: x}

    expected_pattern = re.escape(f"{no_ext_file} has no extension")
    with pytest.raises(ValueError, match=expected_pattern):
        get_loader_for_file(no_ext_file, file_loaders, folder_loaders)


def create_dummy_afm():
    """Create a dummy AFMImageStack for testing purposes."""
    return DummyAFM()


@pytest.mark.parametrize(
    "filename, expected_ext",
    [
        ("example.JPK", ".jpk"),
        ("file1.JpK", ".jpk"),
        ("file.spm", ".spm"),
    ],
)
def test_get_loader_for_folder_detects_extensions(tmp_path, filename, expected_ext):
    """Test that get_loader_for_folder detects file extension and returns loader."""
    (tmp_path / filename).touch()
    (tmp_path / "subfolder").mkdir()

    folder_loaders = {
        ".jpk": load_jpk_folder,
        ".spm": load_spm_folder,
    }

    ext, loader = get_loader_for_folder(tmp_path, folder_loaders)
    assert ext.lower() == expected_ext
    assert callable(loader)


def test_numeric_extension_treated_as_spm(tmp_path):
    """Test that numeric extensions like .001 are treated as .spm."""
    test_file = tmp_path / "image.001"
    test_file.write_text("dummy")

    # Dummy loader
    called = {}

    def fake_loader(folder_path, channel="Height"):
        called["called"] = True
        return "fake stack"

    folder_loaders = {".spm": fake_loader}

    ext, loader = get_loader_for_folder(tmp_path, folder_loaders)
    assert ext == ".spm"
    assert loader is fake_loader


def test_load_spm_folder_handles_numeric_extensions(tmp_path):
    """Test that load_spm_folder can handle numeric extensions like .001."""
    # Create dummy .001 file
    test_file = tmp_path / "frame.001"
    test_file.write_bytes(b"\\Scan Rate: 2.0\n")  # minimal valid header

    # Patch spm.load_spm to return dummy data
    import numpy as np

    dummy_img = np.ones((5, 5), dtype=np.float32)
    dummy_pixel_size = 1.0

    from playNano.io.formats import read_spm_folder

    read_spm_folder.spm.load_spm = lambda f, channel: (dummy_img, dummy_pixel_size)

    # Load
    stack = load_spm_folder(tmp_path, channel="Height")

    assert stack.data.shape == (1, 5, 5)
    assert stack.pixel_size_nm == 1.0
    assert stack.frame_metadata[0]["timestamp"] == 0.0


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
            ".spm": load_spm_folder,
        }

        ext, loader = get_loader_for_folder(tmpdir, folder_loaders)
        assert ext.lower() == ".jpk"
        assert callable(loader)


@pytest.mark.parametrize(
    "filename, expected_ext",
    [
        ("example.h5-JPK", ".h5-jpk"),
        ("file1.h5-JpK", ".h5-jpk"),
        ("file.asd", ".asd"),
    ],
)
def test_get_loader_for_file_detects_extensions(tmp_path, filename, expected_ext):
    """Test that get_loader_for_file detects file extension and returns loader."""
    (tmp_path / filename).touch()
    file_path = tmp_path / filename

    file_loaders = {
        ".h5-jpk": load_h5jpk,
        ".asd": load_asd_file,
    }
    folder_loaders = {
        ".jpk": load_jpk_folder,
        ".spm": load_spm_folder,
    }
    ext, loader = get_loader_for_file(file_path, file_loaders, folder_loaders)
    assert callable(loader)
    assert ext.lower() == expected_ext


def test_get_loader_for_file_known_extension():
    """Test get_loader_for_file returns correct loader for supported file extension."""
    fake_path = Path("/fake/path/sample.h5-jpk")

    file_loaders = {
        ".h5-jpk": load_h5jpk,
        ".asd": load_asd_file,
    }
    folder_loaders = {
        ".jpk": load_jpk_folder,
        ".spm": load_spm_folder,
    }

    ext, loader = get_loader_for_file(fake_path, file_loaders, folder_loaders)
    assert callable(loader)
    assert loader == load_h5jpk
    assert ext == ".h5-jpk"


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

    dummy_afm_stack = np.zeros((1, 10, 10))
    mock_stack = AFMImageStack(
        data=dummy_afm_stack,
        pixel_size_nm=1.0,
        channel="height_trace",
        file_path=str(tmp_path),
        frame_metadata=[{}],
    )

    with patch(
        "playNano.io.loader.load_jpk_folder", return_value=mock_stack
    ) as mock_loader:
        mock_loader.__name__ = "load_jpk_folder"
        result = load_afm_stack(tmp_path)

        mock_loader.assert_called_once_with(tmp_path, channel="height_trace")
        assert isinstance(result, AFMImageStack)
        assert result.data.shape == (1, 10, 10)


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


def test_create_gif_with_flat_data(tmp_path):
    """Test that GIF creation handles flat data without crashing."""
    image_stack = np.zeros((3, 4, 4))  # flat stack
    timestamps = [0, 1, 2]
    output_path = tmp_path / "test.gif"

    with patch("PIL.Image.Image.save") as mock_save:
        create_gif_with_scale_and_timestamp(
            image_stack=image_stack,
            pixel_size_nm=1.0,
            timestamps=timestamps,
            output_path=str(output_path),
            zmin=None,
            zmax=None,
        )

        # Verify save was called, but file wasn't actually written
        mock_save.assert_called_once()


@pytest.mark.parametrize(
    "zmin,zmax",
    [
        (0.0, 1.0),  # normal case
        ("auto", "auto"),  # auto percentiles
        (1.0, 1.0),  # flat image: triggers black frame
        (None, None),  # fallback path (normalize_to_uint8)
    ],
)
def test_create_gif_with_various_zscales(zmin, zmax):
    """Test GIF creation with various zmin/zmax values."""
    # Setup: tiny stack of 2x2 images
    stack = np.stack(
        [
            np.array([[0.0, 0.5], [0.5, 1.0]]),
            np.array([[0.2, 0.2], [0.2, 0.2]]),  # flat if zmin == zmax == 0.2
        ]
    )

    timestamps = [0.0, 1.0]

    with TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "test.gif"

        create_gif_with_scale_and_timestamp(
            image_stack=stack,
            pixel_size_nm=1.0,
            timestamps=timestamps,
            scale_bar_length_nm=50,
            output_path=str(out_path),
            duration=0.1,
            cmap_name="viridis",
            zmin=zmin,
            zmax=zmax,
        )

        assert out_path.exists()
        assert out_path.stat().st_size > 0


class DummyStack:
    """Class to create a dummy stack for testing."""

    def __init__(self, data, processed, metadata, pixel_size, path="dummy.jpk"):
        """Initialise the dummy stack."""
        self.data = data
        self.processed = processed
        self.frame_metadata = metadata
        self.pixel_size_nm = pixel_size
        self.file_path = path


@pytest.mark.parametrize(
    "raw_flag,processed_keys,expected_suffix",
    [
        (False, {"raw": np.ones((2, 2, 2)), "other": np.ones((2, 2, 2))}, "_filtered"),
        (True, {"raw": np.ones((2, 2, 2))}, ""),  # use raw
        (True, {}, ""),  # fallback to stack.data
    ],
)
def test_export_gif_modes(raw_flag, processed_keys, expected_suffix):
    """Test that export_gif creates the expected path."""
    dummy_data = np.random.rand(2, 2, 2)
    dummy_meta = [{"timestamp": 0.0}, {"timestamp": 1.0}]
    dummy_stack = DummyStack(dummy_data, processed_keys, dummy_meta, pixel_size=1.0)

    with TemporaryDirectory() as tmp:
        export_gif(
            afm_stack=dummy_stack,
            make_gif=True,
            output_folder=tmp,
            output_name="test_gif",
            scale_bar_nm=50,
            raw=raw_flag,
            zmin=None,
            zmax=None,
        )
        expected_path = Path(tmp) / f"test_gif{expected_suffix}.gif"
        assert expected_path.exists()
        assert expected_path.stat().st_size > 0


@pytest.mark.parametrize(
    "bad_timestamps",
    [
        [{}, 1.0],  # TypeError when float({}) is attempted
        ["bad", 1.0],  # ValueError when float("bad")
    ],
)
def test_fallback_to_index_on_bad_timestamp(bad_timestamps, tmp_path):
    """Test that gif timestamps fallback to frame index."""
    image_stack = np.ones((2, 2, 2), dtype=float)
    expected_len = image_stack.shape[0]

    ts = list(bad_timestamps)
    if len(ts) > expected_len:
        ts = ts[:expected_len]
    elif len(ts) < expected_len:
        ts += [0.0] * (expected_len - len(ts))

    output_path = tmp_path / "test.gif"

    with patch("playNano.io.gif_export.draw_scale_and_timestamp") as mock_draw:
        mock_draw.side_effect = lambda img, timestamp, **kwargs: img

        create_gif_with_scale_and_timestamp(
            image_stack=image_stack,
            pixel_size_nm=1.0,
            timestamps=ts,
            output_path=output_path,
            scale_bar_length_nm=50,
        )

        timestamps_used = [
            call.kwargs["timestamp"] for call in mock_draw.call_args_list
        ]
        assert timestamps_used == [0, 1]


def test_fallback_to_index_if_no_timestamps(tmp_path):
    """Test that gif_export falls back to frame index if there are no timestamps."""
    image_stack = np.ones((2, 2, 2), dtype=float)
    bad_timestamps = "invalid type"

    output_path = tmp_path / "test.gif"

    with patch("playNano.io.gif_export.draw_scale_and_timestamp") as mock_draw:
        mock_draw.side_effect = lambda img, timestamp, **kwargs: img

        create_gif_with_scale_and_timestamp(
            image_stack=image_stack,
            pixel_size_nm=1.0,
            timestamps=bad_timestamps,
            output_path=output_path,
            scale_bar_length_nm=50,
        )

        timestamps_used = [
            call.kwargs["timestamp"] for call in mock_draw.call_args_list
        ]
        assert timestamps_used == [0, 1]


def test_using_jpk_resource(resource_path):
    """Test that the jpk_folder_0 can be found."""
    resource_dir = resource_path / "jpk_folder_0"
    jpk_file = resource_dir / "jpk_sample_0.jpk"

    assert jpk_file.exists(), "Test .jpk file is missing!"


def test_using_h5jpk_resource(resource_path):
    """Test that the jpk_folder_0 can be found."""
    resource_dir = resource_path
    jpk_file = resource_dir / "sample_0.h5-jpk"

    assert jpk_file.exists(), "Test .h5-jpk file is missing!"


def test_get_loader_for_file_prioritizes_file_loader():
    """File extension loader should be used even if folder loader exists."""
    fake_path = Path("/path/sample.h5-jpk")

    ext, loader = get_loader_for_file(
        fake_path,
        file_loaders={".h5-jpk": load_h5jpk},
        folder_loaders={".jpk": load_jpk_folder},
    )
    assert loader == load_h5jpk
    assert ext == ".h5-jpk"


def test_load_afm_stack_unsupported_file(tmp_path):
    """Test that unsupported file types raise an error."""
    file = tmp_path / "unsupported_file.xyz"
    file.touch()

    with pytest.raises(ValueError, match="Unsupported file type"):
        load_afm_stack(file)


def test_get_loader_for_folder_picks_first_supported(tmp_path):
    """Should pick the first supported extension found in folder."""
    (tmp_path / "file1.spm").touch()
    (tmp_path / "file2.jpk").touch()

    ext, loader = get_loader_for_folder(
        tmp_path, {".spm": load_spm_folder, ".jpk": load_jpk_folder}
    )
    assert ext.lower() in [".asd", ".jpk"]
    assert callable(loader)


@patch("playNano.io.gif_export.create_gif_with_scale_and_timestamp")
def test_export_gif_calls_create(mock_gif):
    """Test calls create_gif_with_scale_and_timestamp with correct parameters."""
    dummy = create_dummy_afm()
    export_gif(dummy, True, Path("."), "basename", scale_bar_nm=100, raw=False)
    assert mock_gif.called


@pytest.fixture
def dummy_stack():
    """Create a dummy image stack."""
    data = np.random.rand(3, 4, 4).astype(np.float32)
    timestamps = [0.0, 1.0, 2.0]
    metadata = [{"timestamp": t} for t in timestamps]
    return data, timestamps, metadata


@pytest.fixture
def afm_stack_obj(dummy_stack):
    """Generate a dummy AFMImageStack object."""
    data, timestamps, metadata = dummy_stack
    stack = AFMImageStack(
        data=data,
        pixel_size_nm=1.0,
        frame_metadata=metadata,
        file_path="dummy_path.h5-jpk",
        channel="height_trace",
    )
    # Add raw to test raw handling
    stack.processed["raw"] = data.copy()
    return stack


def test_save_ome_tiff_stack_creates_file(dummy_stack):
    """Test that save_tiff_bundle creates a file."""
    data, timestamps, metadata = dummy_stack
    stack = AFMImageStack(
        data=data,
        pixel_size_nm=1.0,
        file_path="dummy_path.h5-jpk",
        frame_metadata=metadata,
        channel="height_trace",
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test.ome.tif"
        save_ome_tiff_stack(out_path, stack, raw=False)
        assert out_path.exists()
        with tifffile.TiffFile(out_path) as tif:
            assert tif.series[0].shape[:1] == (3,)  # 3 frames


def test_save_npz_bundle_creates_file(dummy_stack):
    """Test that save_npz_bundle creates a file."""
    data, timestamps, metadata = dummy_stack
    stack = AFMImageStack(
        data=data,
        file_path="dummy_path.h5-jpk",
        pixel_size_nm=1.0,
        frame_metadata=metadata,
        channel="height_trace",
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test"
        save_npz_bundle(out_path, stack, raw=False)
        npz_path = out_path.with_suffix(".npz")
        assert npz_path.exists()
        # Properly close file after reading
        with np.load(npz_path) as contents:
            assert "data" in contents and "pixel_size_nm" in contents
        npz_path.unlink()


def test_save_h5_bundle_creates_file(dummy_stack):
    """Test that save_h5_bundle creates a file."""
    data, timestamps, metadata = dummy_stack
    stack = AFMImageStack(
        data=data,
        pixel_size_nm=1.0,
        file_path="dummy_path.h5-jpk",
        frame_metadata=metadata,
        channel="height_trace",
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test"
        save_h5_bundle(out_path, stack, raw=False)
        h5_path = out_path.with_suffix(".h5")
        assert h5_path.exists()
        with h5py.File(h5_path, "r") as f:
            assert "data" in f
            assert f.attrs["channel"] == "height_trace"


def test_export_bundles_all_formats_unfiltered(afm_stack_obj):
    """Test that export bundles exports raw data without _filtered sufix."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        export_bundles(afm_stack_obj, out_dir, "test_stack", ["tif", "npz", "h5"])
        assert (out_dir / "test_stack.ome.tif").exists()
        assert (out_dir / "test_stack.npz").exists()
        assert (out_dir / "test_stack.h5").exists()


def test_export_bundles_all_formats_filtered(afm_stack_obj):
    """Test that export bundles exports filtered data with _filtered suffix."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        # Ensure raw data exists
        afm_stack_obj.processed["raw"] = afm_stack_obj.data.copy()
        # Simulate presence of filtered data
        afm_stack_obj.processed["filtered"] = afm_stack_obj.data.copy()
        export_bundles(
            afm_stack_obj, out_dir, "test_stack", ["tif", "npz", "h5"], raw=False
        )
        assert (out_dir / "test_stack_filtered.ome.tif").exists()
        assert (out_dir / "test_stack_filtered.npz").exists()
        assert (out_dir / "test_stack_filtered.h5").exists()


def test_accepts_path_object():
    """Returns Path unchanged if input is already a Path."""
    p = Path("/some/path")
    assert check_path_is_path(p) == p


def test_converts_string_to_path():
    """Converts string input to a Path object."""
    path_str = "/some/path"
    result = check_path_is_path(path_str)
    assert isinstance(result, Path)
    assert result == Path(path_str)


def test_raises_type_error_on_invalid_type():
    """Raises TypeError for unsupported input types."""
    with pytest.raises(TypeError):
        check_path_is_path(123)


def test_export_bundles_invalid_format_raises(afm_stack_obj):
    """Test that export budles raises and error and exits if unknown format passed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(SystemExit):
            export_bundles(afm_stack_obj, Path(tmpdir), "bad_format", ["abc"])


def test_export_bundles_raw_flag(afm_stack_obj):
    """Test that data is saved without _filtered tag when raw flag is true."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        export_bundles(afm_stack_obj, out_dir, "stack_raw", ["tif"], raw=True)
        assert (out_dir / "stack_raw.ome.tif").exists()


@pytest.fixture
def sample_record():
    """Create a sample record for testing."""
    return {
        "metadata": {
            "experiment": "test",
            "version": 1.0,
        },
        "results": {
            "values": [
                {"id": 1, "area": 100.0, "label": "A"},
                {"id": 2, "area": 200.0, "label": "B"},
            ]
        },
    }


def test_npz_export_and_reload_synthetic(tmp_path):
    """Ensure exporting and reloading a synthetic AFMImageStack as NPZ retains data."""
    # Create synthetic AFM stack
    n_frames, H, W = 3, 8, 8
    raw_data = np.random.rand(n_frames, H, W).astype(np.float32)
    processed_data = raw_data + 1.0
    meta = [{"timestamp": i} for i in range(n_frames)]

    stack = AFMImageStack(
        data=processed_data,
        pixel_size_nm=2.0,
        channel="height_trace",
        file_path=Path("synthetic.h5-jpk"),
        frame_metadata=meta,
    )
    stack.processed["raw"] = raw_data
    stack.provenance["processing"]["steps"] = ["dummy_filter"]
    stack.provenance["analysis"] = {"results": np.array([1, 2, 3])}

    # Save to NPZ
    out_path = tmp_path / "stack_export"
    save_npz_bundle(out_path, stack, raw=False)

    # Reload and verify
    loaded = load_afm_stack(out_path.with_suffix(".npz"), channel="height_trace")
    assert loaded.data.shape == (3, 8, 8)
    assert loaded.pixel_size_nm == 2.0
    assert loaded.frame_metadata == meta
    prov = loaded.provenance
    print(prov)
    assert prov["processing"]["steps"] == ["dummy_filter"]


def test_npz_export_from_real_resource(tmp_path, resource_path):
    """Test loading a real AFM file and saving to NPZ format."""

    in_path = resource_path / "sample_0.h5-jpk"
    assert in_path.exists(), "Resource file missing"

    stack = load_afm_stack(in_path)
    assert stack.data.ndim == 3
    assert stack.pixel_size_nm > 0
    assert isinstance(stack.frame_metadata, list)

    out_path = tmp_path / "real_stack"
    save_npz_bundle(out_path, stack, raw=False)

    npz = out_path.with_suffix(".npz")
    assert npz.exists()

    contents = load_afm_stack(npz, channel="height_trace")
    assert np.allclose(contents.data, stack.data)
    assert contents.pixel_size_nm == 1.171875
    assert contents.data.shape == stack.data.shape


def test_h5_export_and_reload_synthetic(tmp_path):
    """Ensure exporting and reloading a synthetic AFMImageStack as .h5 retains data."""
    # Create synthetic AFM stack
    n_frames, H, W = 3, 8, 8
    raw_data = np.random.rand(n_frames, H, W).astype(np.float32)
    processed_data = raw_data + 1.0
    meta = [{"timestamp": i} for i in range(n_frames)]

    stack = AFMImageStack(
        data=processed_data,
        pixel_size_nm=2.0,
        channel="height_trace",
        file_path=Path("synthetic.h5-jpk"),
        frame_metadata=meta,
    )
    stack.processed["raw"] = raw_data
    stack.provenance["processing"]["steps"] = ["dummy_filter"]
    stack.provenance["analysis"] = {"results": np.array([1, 2, 3])}

    # Save to H5
    out_path = tmp_path / "stack_export"
    save_h5_bundle(out_path, stack, raw=False)

    # Reload and verify
    loaded = load_afm_stack(out_path.with_suffix(".h5"), channel="height_trace")
    assert loaded.data.shape == (3, 8, 8)
    assert loaded.pixel_size_nm == 2.0
    assert loaded.frame_metadata == meta
    prov = loaded.provenance
    print(prov)
    assert prov["processing"]["steps"] == ["dummy_filter"]


def test_h5_export_from_real_resource(tmp_path, resource_path):
    """Test loading a real AFM file and saving to HDF5 format."""

    in_path = resource_path / "sample_0.h5-jpk"
    assert in_path.exists(), "Resource file missing"

    stack = load_afm_stack(in_path)
    assert stack.data.ndim == 3
    assert stack.pixel_size_nm > 0
    assert isinstance(stack.frame_metadata, list)

    out_path = tmp_path / "real_stack"
    save_h5_bundle(out_path, stack, raw=False)

    h5 = out_path.with_suffix(".h5")
    assert h5.exists()

    contents = load_afm_stack(h5, channel="height_trace")
    assert np.allclose(contents.data, stack.data)
    assert contents.pixel_size_nm == 1.171875
    assert contents.data.shape == stack.data.shape


def test_ome_tif_export_and_reload_synthetic(tmp_path):
    """Ensure exporting and reloading a synthetic AFMImageStack as TIF retains data."""
    # Create synthetic AFM stack
    n_frames, H, W = 3, 8, 8
    raw_data = np.random.rand(n_frames, H, W).astype(np.float32)
    processed_data = raw_data + 1.0
    meta = [{"timestamp": i} for i in range(n_frames)]

    stack = AFMImageStack(
        data=processed_data,
        pixel_size_nm=2.0,
        channel="height_trace",
        file_path=Path("synthetic.h5-jpk"),
        frame_metadata=meta,
    )
    stack.processed["raw"] = raw_data
    stack.provenance["processing"]["steps"] = ["dummy_filter"]
    stack.provenance["analysis"] = {"results": np.array([1, 2, 3])}

    # Save to OME TIF
    out_path = tmp_path / "stack_export.ome.tif"
    print(out_path)

    save_ome_tiff_stack(out_path, stack, raw=False)

    # Reload and verify
    loaded = load_afm_stack(out_path.with_suffix(".tif"), channel="height_trace")
    assert loaded.data.shape == (3, 8, 8)
    assert loaded.pixel_size_nm == 2.0
    assert loaded.frame_metadata == meta
    prov = loaded.provenance
    print(prov)
    assert prov["processing"]["steps"] == ["dummy_filter"]


def test_ome_tif_export_from_real_resource(tmp_path, resource_path):
    """Test loading a real AFM file and saving to OME TIF format."""

    in_path = resource_path / "sample_0.h5-jpk"
    assert in_path.exists(), "Resource file missing"

    stack = load_afm_stack(in_path)
    assert stack.data.ndim == 3
    assert stack.pixel_size_nm > 0
    assert isinstance(stack.frame_metadata, list)

    out_path = tmp_path / "real_stack.ome.tif"
    save_ome_tiff_stack(out_path, stack, raw=False)

    ome = out_path.with_suffix(".tif")
    print(ome)
    assert ome.exists()

    contents = load_afm_stack(ome, channel="height_trace")
    assert np.allclose(contents.data, stack.data)
    assert contents.pixel_size_nm == 1.171875
    assert contents.data.shape == stack.data.shape


def validate_analysis_record(record):
    """Validate structure and basic integrity of an AnalysisPipeline output record."""
    assert isinstance(record, dict)
    for key in ("environment", "analysis", "provenance"):
        assert key in record, f"Missing top-level key: {key}"
        assert isinstance(record[key], dict), f"{key} must be a dict"

    prov = record["provenance"]
    for key in ("steps", "results_by_name", "frame_times"):
        assert key in prov, f"Missing provenance key: {key}"
    assert isinstance(prov["steps"], list), "provenance['steps'] must be a list"
    assert isinstance(
        prov["results_by_name"], dict
    ), "provenance['results_by_name'] must be a dict"
    assert prov["frame_times"] is None or isinstance(prov["frame_times"], list)

    analysis_keys = set(record["analysis"].keys())
    step_analysis_keys = set()
    for step in prov["steps"]:
        for field in (
            "index",
            "name",
            "params",
            "timestamp",
            "version",
            "analysis_key",
        ):
            assert field in step, f"Step missing field: {field}"
        step_analysis_keys.add(step["analysis_key"])

    missing_keys = step_analysis_keys - analysis_keys
    assert not missing_keys, f"Missing analysis keys for steps: {missing_keys}"

    # Ensure each step output is a dict
    for key in step_analysis_keys:
        assert isinstance(
            record["analysis"][key], dict
        ), f"Step output {key} must be a dict"


@pytest.mark.parametrize(
    "ext,save_func",
    [
        (".npz", save_npz_bundle),
        (".h5", save_h5_bundle),
        (".ome.tif", save_ome_tiff_stack),
    ],
)
def test_analysis_pipeline_across_formats(tmp_path, resource_path, ext, save_func):
    """Test AnalysisPipeline from exported NPZ/HDF5/TIF data to JSON/HDF5 outputs."""
    # STEP 1: Load real AFM stack
    input_path = resource_path / "sample_0.h5-jpk"
    stack = load_afm_stack(input_path)

    # STEP 2: Export to target format
    export_path = tmp_path / f"exported_stack{ext}"
    save_func(export_path, stack, raw=False)

    # STEP 3: Reload exported stack for analysis
    reloaded = load_afm_stack(export_path, channel="height_trace")

    # STEP 4: Run analysis pipeline
    pipeline = AnalysisPipeline()
    pipeline.add("count_nonzero")
    pipeline.add("feature_detection", mask_fn="mask_threshold", threshold=1)
    pipeline.add("particle_tracking")
    record = pipeline.run(reloaded)

    # STEP 5: Validate output structure
    validate_analysis_record(record)

    step_keys = [step["analysis_key"] for step in record["provenance"]["steps"]]
    assert step_keys, "No analysis steps found in record provenance"

    # STEP 6: Save analysis results JSON + HDF5
    out_dir = tmp_path / "analysis_results"
    out_dir.mkdir()
    json_path = out_dir / "summary_analysis.json"
    h5_path = out_dir / "summary_analysis.h5"
    record_group_name = "analysis_record"

    with open(json_path, "w") as f:
        json.dump(make_json_safe(record), f, indent=2)

    export_to_hdf5(record, h5_path, dataset_name=record_group_name)

    # STEP 7: Verify JSON contents
    assert json_path.exists()
    with open(json_path) as f:
        loaded_json = json.load(f)
    # Re-validate loaded JSON structure (optional but recommended)
    validate_analysis_record(loaded_json)

    # Check JSON analysis results
    for step in record["provenance"]["steps"]:
        key = step["analysis_key"]
        assert key in loaded_json["analysis"], f"{key} missing in JSON analysis"
        expected_vals = record["analysis"][key]
        actual_vals = loaded_json["analysis"][key]
        assert "results_by_name" in record["provenance"]
        assert isinstance(record["provenance"]["results_by_name"], dict)
        assert "frame_times" in record["provenance"]
        env = record["environment"]
        assert "python_version" in env
        assert "platform" in env

        # frame_times can be None or a list/array of floats
        ft = record["provenance"]["frame_times"]
        assert ft is None or (
            isinstance(ft, (list, tuple))
            and all(isinstance(t, (float, int)) for t in ft)
        )

        # Flexible value comparisons
        for k, expected_val in expected_vals.items():
            actual_val = actual_vals.get(k)
            assert actual_val is not None, f"{key}.{k} missing in JSON analysis"
            # Skip summary-only dicts without 'values'
            if (
                isinstance(actual_val, dict)
                and "_summary" in actual_val
                and "values" not in actual_val
            ):
                continue
            # Skip lists of summaries
            if isinstance(actual_val, list) and all(
                isinstance(x, dict) and "_summary" in x for x in actual_val
            ):
                continue
            # Numeric scalar comparison
            if isinstance(expected_val, (int, float)):
                assert np.isclose(
                    expected_val, actual_val
                ), f"{key}.{k} scalar mismatch"
            # Arrays/lists comparison
            elif isinstance(expected_val, (list, np.ndarray)):
                if isinstance(actual_val, dict) and "values" in actual_val:
                    actual_val = actual_val["values"]
                assert np.allclose(
                    expected_val, actual_val
                ), f"{key}.{k} array mismatch"

    # STEP 8: Verify HDF5 contents
    assert h5_path.exists()
    with h5py.File(h5_path, "r") as f:
        assert record_group_name in f, f"{record_group_name} group missing in HDF5"
        analysis_group = f[record_group_name]["analysis"]

        for step in record["provenance"]["steps"]:
            key = step["analysis_key"]
            assert key in analysis_group, f"{key} missing in HDF5 analysis group"
            step_group = analysis_group[key]
            expected_vals = record["analysis"][key]
            assert "results_by_name" in record["provenance"]
            assert isinstance(record["provenance"]["results_by_name"], dict)
            assert "frame_times" in record["provenance"]
            env = record["environment"]
            assert "python_version" in env
            assert "platform" in env
            # frame_times can be None or a list/array of floats
            ft = record["provenance"]["frame_times"]
            assert ft is None or (
                isinstance(ft, (list, tuple))
                and all(isinstance(t, (float, int)) for t in ft)
            )

            for k, expected_val in expected_vals.items():
                # Skip non-dataset keys or summaries without raw values
                if k not in step_group:
                    continue
                ds = step_group[k]
                # If dataset contains "values" group
                if isinstance(ds, h5py.Group) and "values" in ds:
                    actual_data = ds["values"][()]
                elif isinstance(ds, h5py.Dataset):
                    actual_data = ds[()]
                else:
                    continue
                expected_array = np.array(expected_val)
                assert np.allclose(
                    actual_data, expected_array
                ), f"{key}.{k} HDF5 data mismatch"


@pytest.mark.parametrize(
    "ext,save_func",
    [
        (".npz", save_npz_bundle),
        (".h5", save_h5_bundle),
        (".ome.tif", save_ome_tiff_stack),
    ],
)
def test_analysis_pipeline_schema(
    tmp_path, resource_path, ext, save_func, analysis_pipeline_schema
):
    """Test analysis record against the JSON schema after pipeline execution."""
    input_path = resource_path / "sample_0.h5-jpk"
    stack = load_afm_stack(input_path)

    export_path = tmp_path / f"exported_stack{ext}"
    save_func(export_path, stack, raw=False)

    reloaded = load_afm_stack(export_path, channel="height_trace")

    pipeline = AnalysisPipeline()
    pipeline.add("count_nonzero")
    pipeline.add("feature_detection", mask_fn="mask_threshold", threshold=1)
    pipeline.add("particle_tracking")
    record = pipeline.run(reloaded)

    # Validate the overall record against the schema
    jsonschema.validate(instance=record, schema=analysis_pipeline_schema)


def test_pipeline_clear_and_reuse(tmp_path, resource_path):
    """Test pipeline clearing and reuse for a second analysis run."""
    input_path = resource_path / "sample_0.h5-jpk"
    stack = load_afm_stack(input_path)

    # Process

    processing_pipeline = ProcessingPipeline(stack)
    processing_pipeline.add_filter("remove_plane")
    processing_pipeline.run()

    # First run
    pipeline = AnalysisPipeline()
    pipeline.add("count_nonzero")
    pipeline.add("feature_detection", mask_fn="mask_threshold", threshold=1)
    record1 = pipeline.run(stack)
    assert record1["analysis"], "First pipeline run produced no output"

    # Clear the pipeline
    pipeline.clear()
    assert pipeline.steps == []
    assert pipeline._module_cache == {}

    # Reuse the same instance for a new run
    pipeline.add("log_blob_detection")
    pipeline.add(
        "x_means_clustering",
        detection_module="log_blob_detection",
        coord_columns=("x", "y"),
    )
    new_stack = load_afm_stack(input_path)  # simulate clean start
    processing_pipeline_2 = ProcessingPipeline(new_stack)
    processing_pipeline_2.add_filter("zero_mean")
    processing_pipeline_2.run()
    record2 = pipeline.run(new_stack)

    assert record2["analysis"], "Second pipeline run produced no output"
    expected_keys = ["step_1_log_blob_detection", "step_2_x_means_clustering"]
    actual_keys = list(record2["analysis"].keys())
    assert actual_keys == expected_keys, f"Unexpected analysis keys: {actual_keys}"

    validate_analysis_record(record2)


def test_make_json_safe_handles_numpy_types():
    """Test make_json_safe handles NumPy scalars and arrays without error."""
    raw_record = {
        "environment": {
            "python_version": np.str_("3.11"),
            "float": np.float32(1.5),
        },
        "analysis": {"step_1": {"array": np.array([1.0, 2.0])}},
        "provenance": {
            "steps": [],
            "results_by_name": {},
            "frame_times": None,
        },
    }
    safe = make_json_safe(raw_record)
    dumped = json.dumps(safe)  # Should not raise
    assert isinstance(dumped, str)


def test_sanitize_analysis_for_logging_numpy_types():
    """Test that NumPy types are sanitized into JSON-safe formats."""
    raw = {
        "array": np.array([1.0, 2.0]),
        "float": np.float32(1.5),
        "int": np.int64(2),
    }
    sanitized = sanitize_analysis_for_logging(raw)
    dumped = json.dumps(sanitized)  # Should succeed
    assert isinstance(dumped, str)


def test_validate_analysis_record_minimal():
    """Test that a minimal valid analysis record passes validation."""
    record = {
        "environment": {"python_version": "3.x", "platform": "test"},
        "analysis": {"step_1_dummy": {}},
        "provenance": {
            "steps": [
                {
                    "index": 1,
                    "name": "dummy",
                    "params": {},
                    "timestamp": "2025-01-01T00:00:00Z",
                    "version": None,
                    "analysis_key": "step_1_dummy",
                }
            ],
            "results_by_name": {"dummy": [{}]},
            "frame_times": None,
        },
    }
    validate_analysis_record(record)
