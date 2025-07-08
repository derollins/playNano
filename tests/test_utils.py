"""Tests for the utility functions."""

import importlib.metadata
import platform
import re
import sys
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pytest

from playNano.utils.io_utils import (
    compute_zscale_range,
    convert_height_units_to_nm,
    guess_height_data_units,
    normalize_to_uint8,
    pad_to_square,
)
from playNano.utils.system_info import gather_environment_info
from playNano.utils.time_utils import utc_now_iso


def test_pad_to_square():
    """Test for pad to square function."""
    img = np.ones((50, 100), dtype=np.uint8) * 128
    square = pad_to_square(img)
    assert square.shape[0] == square.shape[1]
    assert square.shape[0] == 100
    assert np.all(square[25:75, :] == 128)


def test_normalize_to_uint8():
    """Test of normalization function."""
    img = np.linspace(0, 1, 100).reshape(10, 10)
    norm_img = normalize_to_uint8(img)
    assert norm_img.dtype == np.uint8
    assert norm_img.min() == 0
    assert norm_img.max() == 255


class TestGuessHeightDataUnits(unittest.TestCase):
    """Tests for the guess_height_data_units function."""

    def test_picometers(self):
        """Detects picometer scale data range."""
        data = np.array([0, 2e5])  # range 2e5 > 1e4 → 'pm'
        self.assertEqual(guess_height_data_units(data), "pm")

    def test_nanometers(self):
        """Detects nanometer scale data range."""
        data = np.array([0, 5e-2])  # 1e-2 < 5e-2 <= 1e4 → 'nm'
        self.assertEqual(guess_height_data_units(data), "nm")

    def test_micrometers(self):
        """Detects micrometer scale data range."""
        data = np.array([0, 5e-3])  # 1e-4 < 5e-3 <= 1e-2 → 'um'
        self.assertEqual(guess_height_data_units(data), "um")

    def test_millimeters(self):
        """Detects millimeter scale data range."""
        data = np.array([0, 5e-5])  # 1e-5 < 5e-5 <= 1e-4 → 'mm'
        self.assertEqual(guess_height_data_units(data), "mm")

    def test_meters(self):
        """Detects meter scale data range."""
        data = np.array([0, 5e-6])  # <= 1e-5 → 'm'
        self.assertEqual(guess_height_data_units(data), "m")

    def test_zero_range(self):
        """Handles zero range data by falling back to 'm'."""
        data = np.full((10,), 42)  # all constant values → 'm' fallback
        self.assertEqual(guess_height_data_units(data), "m")

    def test_non_finite_values(self):
        """Ignores non-finite values when guessing units."""
        data = np.array([np.nan, np.inf, -np.inf, 10, 20])
        self.assertEqual(guess_height_data_units(data), "nm")  # range = 10

    def test_no_finite_raises(self):
        """Raises ValueError if no finite data values exist."""
        data = np.array([np.nan, np.inf, -np.inf])
        with self.assertRaises(ValueError):
            guess_height_data_units(data)


def test_convert_height_units_to_nm():
    """Test conversion of height units to nanometers."""
    data = np.array([[1e-3, 2e-3]])  # pretend this is in meters
    expected = np.array([[1e6, 2e6]])  # nanometers
    result = convert_height_units_to_nm(data, "m")
    np.testing.assert_allclose(result, expected)


# ---Time Utils---


def test_utc_now_iso():
    """Test that utc_now_iso() returns the correct ISO 8601 UTC format."""
    fixed_time = datetime(2025, 6, 25, 12, 0, 0, tzinfo=timezone.utc)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_time

    with patch("playNano.utils.time_utils.datetime", FixedDateTime):
        result = utc_now_iso()
        assert result == "2025-06-25T12:00:00Z"


# --- System Info ---


def test_gather_environment_info_keys():
    """Test that gather_environment_info returns required top-level keys."""
    info = gather_environment_info()

    # Check required keys
    assert "timestamp" in info
    assert "python_version" in info
    assert "platform" in info
    assert "playNano_version" in info


def test_timestamp_format():
    """Test that the timestamp is in ISO 8601 format and ends with 'Z'."""
    info = gather_environment_info()
    assert info["timestamp"].endswith("Z")
    iso8601_regex = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z"
    assert re.match(iso8601_regex, info["timestamp"])


def test_python_version():
    """Test that the reported Python version matches the running interpreter."""
    info = gather_environment_info()
    assert sys.version.split()[0] in info["python_version"]


def test_platform_info():
    """Test that the platform info contains the current system name."""
    info = gather_environment_info()
    assert platform.system().lower() in info["platform"].lower()


@pytest.mark.parametrize("pkg", ["numpy", "scipy", "pandas"])
def test_dependency_versions(pkg):
    """Test that dependency versions are correctly included or omitted if missing."""
    info = gather_environment_info()
    try:
        expected_version = importlib.metadata.version(pkg)
        assert info[f"{pkg}_version"] == expected_version
    except importlib.metadata.PackageNotFoundError:
        assert f"{pkg}_version" not in info


def test_playNano_not_installed():
    """Test that playNano version is None if the package is not installed."""
    with patch("importlib.metadata.version") as mock_version:

        def side_effect(pkg):
            if pkg == "playNano":
                raise importlib.metadata.PackageNotFoundError
            return "1.0.0"

        mock_version.side_effect = side_effect

        info = gather_environment_info()
        assert info["playNano_version"] is None


def test_missing_dependency_skipped():
    """Test that missing dependencies are skipped from the environment info."""
    with patch("importlib.metadata.version") as mock_version:

        def side_effect(pkg):
            if pkg == "scipy":
                raise importlib.metadata.PackageNotFoundError
            return "1.0.0"

        mock_version.side_effect = side_effect

        info = gather_environment_info()
        assert "scipy_version" not in info
        assert "numpy_version" in info  # assuming numpy is in KEY_DEPENDENCIES


# --- Test IO utils ---


@pytest.mark.parametrize("zmin", [{"bad": "dict"}, object(), [1, 2]])
def test_compute_zscale_invalid_zmin_type(zmin):
    """Test that compute_zscale raises and error if zmin is invalid."""
    data = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="zmin must be a float, 'auto', or None."):
        compute_zscale_range(data, zmin=zmin, zmax="auto")


@pytest.mark.parametrize("zmax", [{"bad": "dict"}, object(), [1, 2]])
def test_compute_zscale_invalid_zmax_type(zmax):
    """Test that compute_zscale raises and error if zmax is invalid."""
    data = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="zmax must be a float, 'auto', or None."):
        compute_zscale_range(data, zmin="auto", zmax=zmax)


def test_compute_zscale_zmin_greater_than_zmax():
    """Test that compute_zscale raises a error is zmax is smaller than zmin."""
    data = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="zmin must be less than or equal to zmax."):
        compute_zscale_range(data, zmin=5.0, zmax=1.0)


def test_compute_zscale_valid_manual_values():
    """Test that manual z limits are returned by compute_zscale."""
    data = np.array([[1, 2], [3, 4]])
    zmin, zmax = compute_zscale_range(data, zmin=1.0, zmax=4.0)
    assert zmin == 1.0
    assert zmax == 4.0
