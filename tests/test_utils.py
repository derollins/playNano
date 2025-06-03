"""Tests for the utility functions."""

import numpy as np

from playNano.utils import normalize_to_uint8, pad_to_square


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
