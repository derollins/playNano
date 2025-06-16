"""Tests for the utility functions."""

import unittest

import numpy as np

from playNano.utils.io_utils import (
    convert_height_units_to_nm,
    guess_height_data_units,
    normalize_to_uint8,
    pad_to_square,
)


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
