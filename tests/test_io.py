"""Tests for the functions within io module."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import h5py
import numpy as np
import pytest
import tifffile
from PIL import Image, ImageSequence

from playNano.io.export import (
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
from playNano.playback import vis
from playNano.playback.vis import pad_to_square, play_stack_cv
from playNano.processing.pipeline import ProcessingPipeline
from playNano.stack.afm_stack import AFMImageStack


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
    loader = get_loader_for_file(file_path, file_loaders, folder_loaders)
    assert callable(loader)


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


def test_pad_to_square_rectangular_image():
    """Pad a rectangular image to a square with correct centering."""
    # Create a 2x4 grayscale image with distinct values
    img = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.uint8)
    out = pad_to_square(img, border_color=0)
    # Output should be 4x4, with original centered vertically
    assert out.shape == (4, 4)
    # Top and bottom rows should be zeros
    assert np.all(out[0] == 0) and np.all(out[3] == 0)
    # Middle rows should match original
    assert np.array_equal(out[1, :], img[0]) and np.array_equal(out[2, :], img[1])


def test_pad_to_square_square_image():
    """Return the same image when it's already square."""
    img = np.ones((3, 3), dtype=np.uint8) * 5
    out = pad_to_square(img, border_color=7)
    assert out.shape == (3, 3)
    assert np.array_equal(out, img)


class DummyWindow:
    """Class for testing the cv2 window."""

    pass


def test_play_stack_cv_exits_on_escape(monkeypatch, tmp_path):
    """play_stack_cv should exit immediately when ESC key is pressed."""
    # Create a dummy AFMImageStack with a 2-frame stack of 2x2 images
    stack = AFMImageStack(
        data=np.zeros((2, 2, 2)),
        pixel_size_nm=1.0,
        channel="height_trace",
        file_path=tmp_path,
        frame_metadata=[{}, {}],
    )

    # Monkey­patch all cv2 window functions to do nothing / return values
    monkeypatch.setattr(cv2, "namedWindow", lambda *args, **kwargs: None)
    monkeypatch.setattr(cv2, "resizeWindow", lambda *args, **kwargs: None)
    monkeypatch.setattr(cv2, "setWindowProperty", lambda *args, **kwargs: None)
    # getWindowImageRect returns (x, y, width, height) matching image size (2x2)
    monkeypatch.setattr(cv2, "getWindowImageRect", lambda *args, **kwargs: (0, 0, 2, 2))
    # cv2.normalize is used in rendering; just return the source array
    monkeypatch.setattr(cv2, "normalize", lambda src, dst, a, b, norm_type: src)
    # <— add a stub for cv2.cvtColor
    monkeypatch.setattr(cv2, "cvtColor", lambda src, code: src)

    # Resize and imshow do nothing
    monkeypatch.setattr(cv2, "resize", lambda src, dsize, interpolation: src)
    monkeypatch.setattr(cv2, "imshow", lambda *args, **kwargs: None)
    # waitKey returns ESC code (27)
    monkeypatch.setattr(cv2, "waitKey", lambda delay: 27)
    monkeypatch.setattr(cv2, "destroyWindow", lambda name: None)

    # Call play_stack_cv: should run loop once and exit without error
    play_stack_cv(stack, fps=10.0)


def test_play_stack_cv_flatten_and_toggle(monkeypatch, tmp_path):
    """play_stack_cv should flatten on 'f' and toggle on SPACE."""
    # Create a dummy 2-frame stack of 2x2 images
    img0 = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    img1 = np.array([[5, 6], [7, 8]], dtype=np.uint8)
    stack = AFMImageStack(
        data=np.stack([img0, img1], axis=0),
        pixel_size_nm=1.0,
        channel="height_trace",
        file_path=tmp_path,
        frame_metadata=[{"timestamp": 0.0}, {"timestamp": 1.0}],
    )

    # Track if pipeline.run() was called
    calls = {"flattened": False}

    def fake_run(self):
        calls["flattened"] = True
        # Simulate a flattened stack of all 9’s
        self.stack.processed["raw"] = self.stack.data.copy()
        flat = np.full_like(self.stack.data, 9)
        self.stack.data = flat
        return flat

    # Patch run() on the ProcessingPipeline class
    monkeypatch.setattr(ProcessingPipeline, "run", fake_run)

    # Patch OpenCV functions
    monkeypatch.setattr(cv2, "namedWindow", lambda *args, **kwargs: None)
    monkeypatch.setattr(cv2, "resizeWindow", lambda *args, **kwargs: None)
    monkeypatch.setattr(cv2, "setWindowProperty", lambda *args, **kwargs: None)
    monkeypatch.setattr(cv2, "getWindowImageRect", lambda *args, **kwargs: (0, 0, 2, 2))
    monkeypatch.setattr(cv2, "normalize", lambda src, dst, a, b, norm_type: src)
    # <— add a stub for cv2.cvtColor
    monkeypatch.setattr(cv2, "cvtColor", lambda src, code: src)
    monkeypatch.setattr(cv2, "resize", lambda src, dsize, interpolation: src)
    monkeypatch.setattr(cv2, "imshow", lambda *args, **kwargs: None)
    monkeypatch.setattr(cv2, "destroyWindow", lambda name: None)

    # Simulate key presses: 'f' (flatten), SPACE (toggle), ESC (exit)
    key_sequence = [ord("f"), 32, 27]
    monkeypatch.setattr(cv2, "waitKey", lambda delay: key_sequence.pop(0))

    # Run the function
    play_stack_cv(stack, fps=5.0)

    # Check that flatten was called
    assert calls["flattened"] is True


def test_pad_to_square_with_border_color():
    """pad_to_square applies custom border_color correctly."""
    img = np.zeros((1, 2), dtype=np.uint8)
    out = pad_to_square(img, border_color=5)
    # Output is 2x2; bottom row should be border_color
    assert out.shape == (2, 2)
    assert np.all(out[1] == 5)
    # Top row should contain original values
    assert out[0, 0] == 0 and out[0, 1] == 0


def test_normalize_to_uint8_handles_negative_values():
    """Test normalize_to_uint8 properly rescales negative values."""
    img = np.array([[-10, 0], [10, 20]], dtype=np.float32)
    out = normalize_to_uint8(img)
    assert out.min() == 0
    assert out.max() == 255


def test_normalize_to_uint8_large_image():
    """Test normalize_to_uint8 on large arrays."""
    img = np.random.rand(1000, 1000) * 100
    out = normalize_to_uint8(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_create_gif_with_mismatched_timestamps(tmp_path):
    """Test GIF creation with fewer timestamps than frames."""
    stack = np.random.rand(3, 10, 10)
    timestamps = [0.0]  # only 1 timestamp
    output_path = tmp_path / "bad_timestamps.gif"

    with pytest.raises(ValueError, match="timestamps.*length"):
        create_gif_with_scale_and_timestamp(
            image_stack=stack,
            pixel_size_nm=1.0,
            timestamps=timestamps,
            scale_bar_length_nm=5,
            output_path=output_path,
            duration=0.2,
            cmap_name="afmhot",
        )


def test_pad_to_square_color_image():
    """Ensure color images (H, W, 3) are padded correctly, only 2D."""
    img = np.ones((2, 4), dtype=np.uint8) * 100
    out = pad_to_square(img, border_color=0)
    assert out.shape == (4, 4)
    assert np.all(out[0] == 0)
    assert np.all(out[-1] == 0)


def test_pad_to_square_rgb_square_image():
    """Ensure no padding occurs on square RGB image."""
    img = np.ones((5, 5), dtype=np.uint8) * 255
    out = pad_to_square(img, border_color=0)
    assert out.shape == img.shape
    assert np.array_equal(out, img)


def test_get_loader_for_file_prioritizes_file_loader():
    """File extension loader should be used even if folder loader exists."""
    fake_path = Path("/path/sample.h5-jpk")

    loader = get_loader_for_file(
        fake_path,
        file_loaders={".h5-jpk": load_h5jpk},
        folder_loaders={".jpk": load_jpk_folder},
    )
    assert loader == load_h5jpk


def test_load_afm_stack_unsupported_file(tmp_path):
    """Test that unsupported file types raise an error."""
    file = tmp_path / "unsupported_file.xyz"
    file.touch()

    with pytest.raises(ValueError, match="Unsupported file type"):
        load_afm_stack(file)


def test_play_stack_cv_skips_on_other_keys(monkeypatch, tmp_path):
    """Pressing an unhandled key should just continue playback."""
    stack = AFMImageStack(
        data=np.random.rand(2, 5, 5),
        pixel_size_nm=1.0,
        channel="height_trace",
        file_path=tmp_path,
        frame_metadata=[{"timestamp": 0.0}, {"timestamp": 1.0}],
    )

    monkeypatch.setattr(cv2, "namedWindow", lambda *args, **kwargs: None)
    monkeypatch.setattr(cv2, "resizeWindow", lambda *args, **kwargs: None)
    monkeypatch.setattr(cv2, "setWindowProperty", lambda *args, **kwargs: None)
    monkeypatch.setattr(cv2, "getWindowImageRect", lambda *args, **kwargs: (0, 0, 5, 5))
    monkeypatch.setattr(cv2, "normalize", lambda src, dst, a, b, norm_type: src)
    monkeypatch.setattr(cv2, "cvtColor", lambda src, code: src)
    monkeypatch.setattr(cv2, "resize", lambda src, dsize, interpolation: src)
    monkeypatch.setattr(cv2, "imshow", lambda *args, **kwargs: None)
    monkeypatch.setattr(cv2, "destroyWindow", lambda name: None)

    keys = [ord("x"), 27]  # random key + ESC to exit
    monkeypatch.setattr(cv2, "waitKey", lambda delay: keys.pop(0))

    play_stack_cv(stack, fps=5.0)


def test_get_loader_for_folder_picks_first_supported(tmp_path):
    """Should pick the first supported extension found in folder."""
    (tmp_path / "file1.spm").touch()
    (tmp_path / "file2.jpk").touch()

    ext, loader = get_loader_for_folder(
        tmp_path, {".spm": load_spm_folder, ".jpk": load_jpk_folder}
    )
    assert ext.lower() in [".asd", ".jpk"]
    assert callable(loader)


def test_prepare_output_directory(tmp_path):
    """Test that prepare_output_directory creates a directory and returns its path."""
    path = vis.prepare_output_directory(str(tmp_path))
    assert path.exists()
    assert path.is_dir()


@patch("playNano.io.gif_export.create_gif_with_scale_and_timestamp")
def test_export_gif_calls_create(mock_gif):
    """Test calls create_gif_with_scale_and_timestamp with correct parameters."""
    dummy = create_dummy_afm()
    export_gif(dummy, True, Path("."), "basename", scale_bar_nm=100, raw=False)
    assert mock_gif.called


@pytest.fixture
def dummy_stack():
    data = np.random.rand(3, 4, 4).astype(np.float32)
    timestamps = [0.0, 1.0, 2.0]
    metadata = [{"timestamp": t} for t in timestamps]
    return data, timestamps, metadata


@pytest.fixture
def afm_stack_obj(dummy_stack):
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
    data, timestamps, _ = dummy_stack
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test.ome.tif"
        save_ome_tiff_stack(out_path, data, 1.0, timestamps)
        assert out_path.exists()
        with tifffile.TiffFile(out_path) as tif:
            assert tif.series[0].shape[:1] == (3,)  # 3 frames


def test_save_npz_bundle_creates_file(dummy_stack):
    data, timestamps, _ = dummy_stack
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test"
        save_npz_bundle(out_path, data, 1.0, timestamps)
        npz_path = out_path.with_suffix(".npz")
        assert npz_path.exists()
        # Properly close file after reading
        with np.load(npz_path) as contents:
            assert "data" in contents and "pixel_size_nm" in contents
        npz_path.unlink()


def test_save_h5_bundle_creates_file(dummy_stack):
    data, timestamps, metadata = dummy_stack
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test"
        save_h5_bundle(out_path, data, 1.0, timestamps, metadata)
        h5_path = out_path.with_suffix(".h5")
        assert h5_path.exists()
        with h5py.File(h5_path, "r") as f:
            assert "data" in f
            assert f.attrs["channel"] == "height_trace"


def test_export_bundles_all_formats_unfiltered(afm_stack_obj):
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        export_bundles(afm_stack_obj, out_dir, "test_stack", ["tif", "npz", "h5"])
        assert (out_dir / "test_stack.ome.tif").exists()
        assert (out_dir / "test_stack.npz").exists()
        assert (out_dir / "test_stack.h5").exists()


def test_export_bundles_all_formats_filtered(afm_stack_obj):
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


def test_export_bundles_invalid_format_raises(afm_stack_obj):
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(SystemExit):
            export_bundles(afm_stack_obj, Path(tmpdir), "bad_format", ["abc"])


def test_export_bundles_raw_flag(afm_stack_obj):
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        export_bundles(afm_stack_obj, out_dir, "stack_raw", ["tif"], raw=True)
        assert (out_dir / "stack_raw.ome.tif").exists()
