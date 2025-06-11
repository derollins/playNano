"""Tests for filters and masking functions in playNano.processing."""

import numpy as np
import pytest
from scipy.ndimage import generate_binary_structure
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import playNano.processing.filters as filters
import playNano.processing.mask_generators as mask_gen
from playNano.processing.masked_filters import (
    polynomial_flatten_masked,
    remove_plane_masked,
    row_median_align_masked,
)

structure = generate_binary_structure(rank=2, connectivity=2)  # 8-connectivity

# Tests for playNano.processing.filters module


def test_row_median_align_basic():
    """Test the row meadian align function."""
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    aligned = filters.row_median_align(data)
    # Check shape unchanged
    assert aligned.shape == data.shape
    # Each row median should now be zero
    row_medians = np.median(aligned, axis=1)
    assert np.allclose(row_medians, 0)


def test_remove_plane_exact_plane():
    """Test the remove plane fucntion with a perfect plane."""
    h, w = 10, 10
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    data = 2 * X + 3 * Y + 5  # perfect plane
    corrected = filters.remove_plane(data)
    assert np.allclose(corrected, 0.0, atol=1e-6)


def test_remove_plane_removes_tilt_with_noise():
    """Test the remove plane funciton with noise."""
    # create a tilted plane: z = 2x + 3y + 5
    h, w = 10, 10
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    data = 2 * X + 3 * Y + 5
    # Add some noise
    data_noisy = data + np.random.normal(0, 0.1, size=data.shape)
    corrected = filters.remove_plane(data_noisy)
    # After correction, mean trend should be close to zero
    plane = np.median(corrected)
    assert abs(plane) < 1.2e-2


def test_polynomial_flatten_basic():
    """Test the flattening of a basic quadratic surface."""
    # Create data with a quadratic surface: z = 1 + 2x + 3y + 4x^2 + 5xy + 6y^2
    h, w = 10, 10
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    data = 1 + 2 * X + 3 * Y + 4 * X**2 + 5 * X * Y + 6 * Y**2
    flattened = filters.polynomial_flatten(data, order=2)
    # After flattening, mean should be near zero
    assert abs(np.mean(flattened)) < 1e-6


def test_polynomial_flatten_various_orders():
    """Test the flattening of various polynominal surfaces."""
    # Create a synthetic surface with known polynomial terms
    h, w = 20, 20
    X, Y = np.meshgrid(np.arange(w), np.arange(h))

    # Generate data: plane + quadratic + cubic terms
    data_linear = 3 + 2 * X + 5 * Y  # order=1 exact
    data_quadratic = (
        data_linear + 1.5 * X**2 - 0.5 * X * Y + 2 * Y**2
    )  # order=2 exact
    data_cubic = (
        data_quadratic
        + 0.1 * X**3
        - 0.2 * X**2 * Y
        + 0.3 * X * Y**2
        - 0.4 * Y**3
    )  # order=3 exact

    # Test order=1 flattening recovers zero residual for linear surface
    residual_1 = filters.polynomial_flatten(data_linear, order=1)
    assert np.allclose(
        residual_1, 0, atol=1e-8
    ), "Order 1 flattening failed on linear data"  # noqa

    # Test order=2 flattening recovers zero residual for quadratic surface
    residual_2 = filters.polynomial_flatten(data_quadratic, order=2)
    assert np.allclose(
        residual_2, 0, atol=1e-8
    ), "Order 2 flattening failed on quadratic data"  # noqa

    # Test order=3 flattening recovers zero residual for cubic surface
    residual_3 = filters.polynomial_flatten(data_cubic, order=3)
    assert np.allclose(
        residual_3, 0, atol=1e-7
    ), "Order 3 flattening failed on cubic data"  # noqa

    # Test error on invalid order
    with pytest.raises(ValueError):
        filters.polynomial_flatten(data_cubic, order=0)

    with pytest.raises(ValueError):
        filters.polynomial_flatten(data_cubic, order=-1)

    # Test error on bad shape
    with pytest.raises(ValueError):
        filters.polynomial_flatten(np.ones((10, 10, 10)), order=2)


def test_zero_mean_no_mask():
    """Test the zero_mean function without a mask."""
    data = np.array([[1, 2], [3, 4]], dtype=float)
    zeroed = filters.zero_mean(data)
    # mean of output should be zero
    assert abs(np.mean(zeroed)) < 1e-12


def test_zero_mean_with_mask():
    """Test the zero_mean function with a mask."""
    data = np.array([[1, 2], [3, 4]], dtype=float)
    mask = np.array([[False, True], [False, False]])
    zeroed = filters.zero_mean(data, mask=mask)
    # mean of unmasked pixels should be ~0
    assert np.allclose(np.mean(zeroed[~mask]), 0)
    # masked pixels unaffected by mean calc


def test_zero_mean_mask_all_masked():
    """Test the error for zero_mean when all pixels are masked."""
    data = np.ones((3, 3))
    mask = np.ones_like(data, dtype=bool)  # all True, exclude all pixels
    with pytest.raises(ValueError):
        filters.zero_mean(data, mask=mask)


def test_gaussian_filter_smooths():
    """Test the smoothing of the gaussian filter."""
    np.random.seed(0)
    data = np.random.normal(size=(20, 20))
    smoothed = filters.gaussian_filter(data, sigma=2)
    # Variance should decrease after smoothing
    assert smoothed.var() < data.var()


def test_remove_plane_removes_slope():
    """Test the plane removal fucntion removes a slope."""
    h, w = 10, 10
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    data = 2 * X + 3 * Y + 5 + np.random.normal(0, 0.1, size=(h, w))
    corrected = filters.remove_plane(data)

    # Re-fit to check residual trend
    Xf = X.ravel()
    Yf = Y.ravel()
    Zf = corrected.ravel()
    features = np.stack((Xf, Yf)).T
    model = LinearRegression()
    model.fit(features, Zf)

    assert abs(model.coef_[0]) < 0.05  # slope in x
    assert abs(model.coef_[1]) < 0.05  # slope in y


def test_register_filters_keys():
    """Test that filters are registered correctly."""
    keys = filters.register_filters().keys()
    expected = {
        "remove_plane",
        "row_median_align",
        "zero_mean",
        "polynomial_flatten",
        "gaussian_filter",
    }
    assert set(keys) == expected


# Tests for playNano.processing.mask_generators module


def test_mask_threshold_basic():
    """Tests that the threshold mask fuctnion works as expected."""
    data = np.array([[0.1, 0.5], [1.2, -1.5]])
    mask = mask_gen.mask_threshold(data, threshold=1.0)
    expected = np.array([[False, False], [True, True]])
    assert np.array_equal(mask, expected)


def test_mask_mean_offset_std_range():
    """Test that the mean offset mask works as expected."""
    data = np.array([0.0, 0.0, 0.0, 10.0])
    mask = mask_gen.mask_mean_offset(data, factor=1.0)
    # Only the outlier (10.0) should be masked
    expected = np.array([False, False, False, True])
    assert np.array_equal(mask, expected)


def test_mask_morphological_basic():
    """Test the morphological mask thresholds and closed."""
    data = np.array(
        [
            [0.0, 1.2, 0.0],
            [0.0, 1.3, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    mask = mask_gen.mask_morphological(data, threshold=1.0, structure_size=3)

    # The mask will include the pixels above threshold, after closing.
    # Check the expected True pixels manually:
    expected_mask = np.array(
        [
            [False, False, False],
            [False, True, False],
            [False, False, False],
        ]
    )

    np.testing.assert_array_equal(mask, expected_mask)


def test_mask_morphological_fills_small_holes():
    """Test that the morphological mask fills in small holes."""
    data = np.array(
        [
            [1.1, 1.1, 1.1, 1.1, 1.1],
            [1.1, 0.0, 0.0, 0.0, 1.1],
            [1.1, 0.0, 1.1, 0.0, 1.1],
            [1.1, 0.0, 0.0, 0.0, 1.1],
            [1.1, 1.1, 1.1, 1.1, 1.1],
        ]
    )
    mask = mask_gen.mask_morphological(data, threshold=1.0, structure_size=3)

    # The mask after closing should fill some holes but not all.
    expected_mask = np.array(
        [
            [False, False, False, False, False],
            [False, True, True, True, False],
            [False, True, True, True, False],
            [False, True, True, True, False],
            [False, False, False, False, False],
        ]
    )
    np.testing.assert_array_equal(mask, expected_mask)
    # sum is 9 True pixels
    assert np.sum(mask) == 9


def test_mask_adaptive_blocks():
    """Test the adaptive mask."""
    data = np.zeros((10, 10))
    data[0:5, 0:5] = np.random.normal(loc=10, scale=1, size=(5, 5))
    data[5:, :] = 0
    data[:, 5:] = 0
    mask = mask_gen.mask_adaptive(data, block_size=5, offset=1.0)
    assert np.any(mask[0:5, 0:5])
    assert np.all(~mask[5:, :])
    assert np.all(~mask[:, 5:])


def test_register_masking_returns_all():
    """Test the maskign functions are correctly registered."""
    mask_funcs = mask_gen.register_masking()
    assert "mask_threshold" in mask_funcs
    assert callable(mask_funcs["mask_threshold"])
    assert "mask_adaptive" in mask_funcs


# Test masked filters fucntions


def make_simple_plane_data(h=5, w=5):
    """Create a 2D array with a simple tilted plane: z = 2*x + 3*y + 5."""
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    return 2 * X + 3 * Y + 5


def test_remove_plane_masked_basic():
    """Test the remove_plane_masked function with masked data."""
    data = make_simple_plane_data()
    # Mask center pixel as foreground, rest as background
    mask = np.zeros_like(data, dtype=bool)
    mask[2, 2] = True

    result = remove_plane_masked(data, mask)
    # Since data is a perfect plane, background fitting subtracts all plane
    # -> zeros at background
    assert np.allclose(result[~mask], 0, atol=1e-12)
    # Foreground pixel (masked) will be data value minus predicted plane value,
    # should be close to zero as well
    assert abs(result[2, 2]) < 1e-10


def test_remove_plane_masked_shape_mismatch():
    """Test a ValueError is raised if the mask and data not the same shape."""
    data = np.zeros((4, 4))
    mask = np.zeros((5, 5), dtype=bool)
    with pytest.raises(ValueError):
        remove_plane_masked(data, mask)


def test_remove_plane_masked_not_enough_bg_points():
    """Test for ValueError if not enough points for plane removal after masking."""
    data = np.ones((4, 4))
    # All pixels masked except 2
    mask = np.ones((4, 4), dtype=bool)
    mask[0, 0] = False
    mask[1, 1] = False
    with pytest.raises(ValueError):
        remove_plane_masked(data, mask)


def test_polynomial_flatten_masked_basic():
    """Test the polynomial_flatten function with masked data."""
    data = make_simple_plane_data()
    # Use no mask: all background
    mask = np.zeros_like(data, dtype=bool)
    result = polynomial_flatten_masked(data, mask, order=2)
    # The fitted polynomial should remove the plane approx perfectly,
    # so result near zero
    assert np.allclose(result, 0, atol=1e-12)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_polynomial_flatten_masked_orders(order: int):
    """Test polynomial flattening for different orders using masked background."""
    h, w = 64, 64
    X, Y = np.meshgrid(np.arange(w), np.arange(h))

    # Create polynomial background using scikit-learn
    coords = np.stack([X.ravel(), Y.ravel()], axis=1)
    poly = PolynomialFeatures(order)
    A = poly.fit_transform(coords)
    # Random but deterministic coefficients
    rng = np.random.default_rng(seed=42)
    coeff = rng.normal(scale=1.0, size=A.shape[1])
    background = (A @ coeff).reshape(h, w)

    # Add a localized Gaussian bump as the foreground (to be masked out)
    bump = np.exp(-((X - 32) ** 2 + (Y - 32) ** 2) / (2 * 5**2)) * 10.0
    data = background + bump

    # Foreground mask excludes bump region from fitting
    mask = bump > 1.0

    # Apply flattening
    flattened = polynomial_flatten_masked(data, mask, order=order)

    # Assert shape and residual
    residual = flattened[~mask]
    assert flattened.shape == data.shape
    assert np.abs(residual).mean() < 1.0, f"Flattening failed for order={order}"


def test_polynomial_flatten_masked_shape_mismatch():
    """Test a ValueError is raised if the mask and data not the same shape."""
    data = np.zeros((4, 4))
    mask = np.zeros((5, 5), dtype=bool)
    with pytest.raises(ValueError):
        polynomial_flatten_masked(data, mask)


def test_row_median_align_masked_basic():
    """Test the row_median_align function with masked data."""
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    # Mask middle pixel in each row
    mask = np.zeros_like(data, dtype=bool)
    mask[:, 1] = True

    result = row_median_align_masked(data, mask)
    # For each row, median of unmasked pixels is median([1,3])=2, [4,6]=5, [7,9]=8
    expected = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    assert np.allclose(result, expected)


def test_row_median_align_masked_fully_masked_row():
    """Test that row median align works with a fully masked row."""
    data = np.array([[1, 2], [3, 4]], dtype=float)
    mask = np.zeros_like(data, dtype=bool)
    # Fully mask first row
    mask[0, :] = True
    result = row_median_align_masked(data, mask)
    # First row median defaults to 0, so unchanged;
    # #second row median is median([3,4])=3.5
    expected = np.array([[1, 2], [-0.5, 0.5]])
    assert np.allclose(result, expected)


def test_row_median_align_masked_shape_mismatch():
    """Test that row median align raises error is mask and data shape are different."""
    data = np.zeros((4, 4))
    mask = np.zeros((5, 5), dtype=bool)
    with pytest.raises(ValueError):
        row_median_align_masked(data, mask)
