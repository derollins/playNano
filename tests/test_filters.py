"""Tests for filters and masking functions in playnano.processing."""

import logging

import numpy as np
import pytest
from scipy.ndimage import generate_binary_structure
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import playnano.processing.filters as filters
import playnano.processing.mask_generators as mask_gen
from playnano.processing.masked_filters import (
    polynomial_flatten_masked,
    remove_plane_masked,
    row_median_align_masked,
    zero_mean_masked,
    zero_median_masked,
)

structure = generate_binary_structure(rank=2, connectivity=2)  # 8-connectivity

# Tests for playnano.processing.filters module


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
    """Test the remove plane function with a perfect plane."""
    h, w = 10, 10
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    data = 2 * X + 3 * Y + 5  # perfect plane
    corrected = filters.remove_plane(data)
    assert np.allclose(corrected, 0.0, atol=1e-6)


def test_remove_plane_removes_tilt_with_noise():
    """Test the remove plane function with noise."""
    # create a tilted plane: z = 2x + 3y + 5
    h, w = 10, 10
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    data = 2 * X + 3 * Y + 5
    # Add some noise
    data_noisy = data + np.random.normal(0, 0.1, size=data.shape)
    corrected = filters.remove_plane(data_noisy)
    # After correction, mean trend should be close to zero
    plane = np.median(corrected)
    assert abs(plane) < 2.5e-2


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


def test_zero_median_no_mask():
    """Test the zero_median function without a mask."""
    data = np.array([[1, 2], [3, 4]], dtype=float)
    zeroed = filters.zero_median(data)
    # median of output should be zero
    assert abs(np.median(zeroed)) < 1e-12


def test_gaussian_filter_smooths():
    """Test the smoothing of the gaussian filter."""
    np.random.seed(0)
    data = np.random.normal(size=(20, 20))
    smoothed = filters.gaussian_filter(data, sigma=2)
    # Variance should decrease after smoothing
    assert smoothed.var() < data.var()


def test_remove_plane_removes_slope():
    """Test the plane removal function removes a slope."""
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


def test_vertical_flip():
    """Test that vertical_flip correctly flips a 2 by 2 array vertically."""
    data = np.array([[1, 2], [3, 4]], dtype=float)
    flipped = filters.vertical_flip(data)
    expected = np.array([[3, 4], [1, 2]], dtype=float)
    assert np.array_equal(flipped, expected)


def test_register_filters_keys():
    """Test that filters are registered correctly."""
    keys = filters.register_filters().keys()
    expected = {
        "remove_plane",
        "row_median_align",
        "zero_mean",
        "zero_median",
        "polynomial_flatten",
        "gaussian_filter",
        "vertical_flip",
    }
    assert set(keys) == expected


# Tests for playnano.processing.mask_generators module


def test_mask_threshold_basic():
    """Tests that the threshold mask fuctnion works as expected."""
    data = np.array([[0.1, 0.5], [1.2, -1.5]])
    mask = mask_gen.mask_threshold(data, threshold=1.0)
    expected = np.array([[False, False], [True, False]])
    assert np.array_equal(mask, expected)


def test_mask_below_threshold_basic():
    """Tests that the threshold mask fuctnion works as expected."""
    data = np.array([[0.1, 0.5], [1.2, -1.5]])
    mask = mask_gen.mask_below_threshold(data, threshold=1.0)
    expected = np.array([[True, True], [False, True]])
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


# Test masked filters functions


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


def test_zero_mean_masked_basic():
    """Test zero mean on a simple image with a single pixel mask."""
    # Simple 3x3 image
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    mask = np.array(
        [[False, False, False], [False, True, False], [False, False, False]]
    )

    # Background pixels are all except center
    expected_bg = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=float)
    expected_mean = expected_bg.mean()
    expected = data - expected_mean

    result = zero_mean_masked(data, mask)
    np.testing.assert_allclose(result, expected)
    # mean of unmasked pixels should be ~0
    assert np.allclose(np.mean(result[~mask]), 0)


def test_zero_mean_masked_no_mask(caplog):
    """Test warning is logged zero_mean output returned when no mask is supplied."""
    # Simple 3x3 image
    caplog.set_level(logging.WARNING, logger="playnano.processing.masked_filters")

    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    mask = None
    result = zero_mean_masked(data, mask)
    assert np.mean(result) == 0
    assert any(
        "Masked zero_mean filter selected but no mask found. Applying unmasked."
        in rec.message
        for rec in caplog.records
    )


def test_zero_mean_masked_all_foreground():
    """Test that error is raised if whole image is masked."""
    data = np.ones((2, 2))
    mask = np.ones_like(data, dtype=bool)  # all foreground
    with pytest.raises(ValueError, match="Mask excludes all pixels"):
        zero_mean_masked(data, mask)


def test_zero_mean_masked_shape_mismatch():
    """Test that error is raised if mask and data are different shapes."""
    data = np.ones((2, 2))
    mask = np.ones((3, 3), dtype=bool)
    with pytest.raises(ValueError, match="Mask must have same shape"):
        zero_mean_masked(data, mask)


def test_zero_median_masked_basic():
    """Test zero median on a simple image with a single pixel mask."""
    # Simple 3x3 image
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    mask = np.array(
        [[False, False, False], [False, True, False], [False, False, False]]
    )

    # Background pixels are all except center
    expected_bg = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=float)
    expected_median = np.median(expected_bg)
    expected = data - expected_median

    result = zero_median_masked(data, mask)
    np.testing.assert_allclose(result, expected)
    # median of unmasked pixels should be ~0
    assert np.allclose(np.median(result[~mask]), 0)


def test_zero_median_masked_no_mask(caplog):
    """Test warning is logged zero_median output returned when no mask is supplied."""
    # Simple 3x3 image
    caplog.set_level(logging.WARNING, logger="playnano.processing.masked_filters")

    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    mask = None
    result = zero_median_masked(data, mask)
    assert np.median(result) == 0
    assert any(
        "Masked zero_median filter selected but no mask found. Applying unmasked."
        in rec.message
        for rec in caplog.records
    )


def test_zero_median_masked_all_foreground():
    """Test that error is raised if whole image is masked."""
    data = np.ones((2, 2))
    mask = np.ones_like(data, dtype=bool)  # all foreground
    with pytest.raises(ValueError, match="Mask excludes all pixels"):
        zero_median_masked(data, mask)


def test_zero_median_masked_shape_mismatch():
    """Test that error is raised if mask and data are different shapes."""
    data = np.ones((2, 2))
    mask = np.ones((3, 3), dtype=bool)
    with pytest.raises(ValueError, match="Mask must have same shape"):
        zero_median_masked(data, mask)


# --- Polynomial and plane fits: scaled coordinates and NaN pixels (#105) ---


def _polynomial_surface(n: int, order: int, seed: int = 0) -> np.ndarray:
    """Return an exact polynomial surface of total degree ``order`` on n x n."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n] / (n - 1)  # 0..1, so no term dominates
    surface = np.zeros((n, n))
    for total in range(order + 1):
        for i in range(total + 1):
            surface += rng.normal() * x**i * y ** (total - i)
    return surface * 10.0


@pytest.mark.parametrize(
    ("n", "order"),
    [(128, o) for o in range(1, 7)]
    + [(512, o) for o in range(1, 7)]
    + [(1024, o) for o in (2, 3, 4)],
)
@pytest.mark.parametrize("masked", [False, True])
def test_polynomial_flatten_removes_exact_surface(n, order, masked):
    """An exact polynomial of the fitted order is removed to round-off level."""
    data = _polynomial_surface(n, order)
    if masked:
        mask = np.zeros_like(data, dtype=bool)
        mask[n // 3 : n // 2, n // 3 : n // 2] = True
        flattened = polynomial_flatten_masked(data, mask, order=order)
    else:
        flattened = filters.polynomial_flatten(data, order=order)
    assert np.abs(flattened).max() < 1e-8 * np.abs(data).max()


@pytest.mark.parametrize("order", [2, 3, 4, 5, 6])
def test_polynomial_flatten_residual_is_noise_on_large_frame(order):
    """On a 512 px frame the residual is the noise, at every order."""
    rng = np.random.default_rng(0)
    n = 512
    y, x = np.mgrid[0:n, 0:n]
    curve = 1e-5 * (x - n * 0.4) ** 2 + 2e-5 * (y - n * 0.6) ** 2
    data = curve + rng.normal(scale=0.01, size=(n, n))
    assert filters.polynomial_flatten(data, order=order).std() == pytest.approx(
        0.01, rel=0.02
    )


def test_remove_plane_matches_previous_scikit_learn_fit():
    """``remove_plane`` gives the same result as the old scikit-learn fit."""
    rng = np.random.default_rng(1)
    h, w = 37, 53
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    data = 0.3 * X - 0.7 * Y + 4 + rng.normal(size=(h, w))
    features = np.stack([X.ravel(), Y.ravel()], axis=1)
    model = LinearRegression().fit(features, data.ravel())
    expected = data - model.predict(features).reshape(h, w)

    np.testing.assert_allclose(filters.remove_plane(data), expected, atol=1e-9)

    mask = np.zeros_like(data, dtype=bool)
    mask[10:20, 10:30] = True
    model = LinearRegression().fit(features[~mask.ravel()], data[~mask])
    expected = data - model.predict(features).reshape(h, w)
    np.testing.assert_allclose(remove_plane_masked(data, mask), expected, atol=1e-9)


@pytest.mark.parametrize(
    "flatten",
    [
        lambda d: filters.remove_plane(d),
        lambda d: filters.polynomial_flatten(d, order=3),
        lambda d: remove_plane_masked(d, np.zeros(d.shape, dtype=bool)),
        lambda d: polynomial_flatten_masked(d, np.zeros(d.shape, dtype=bool), 3),
    ],
    ids=["remove_plane", "polynomial_flatten", "remove_plane_masked", "poly_masked"],
)
def test_flatten_ignores_nan_pixels(flatten):
    """Leave NaN pixels out of the fit and keep them NaN; flatten the rest."""
    data = _polynomial_surface(64, 1)  # a plane, exact for every order >= 1
    data[5, 7] = np.nan
    data[40:44, 0] = np.nan

    flattened = flatten(data)

    assert np.array_equal(np.isnan(flattened), np.isnan(data))
    assert np.nanmax(np.abs(flattened)) < 1e-9


def test_flatten_with_too_few_finite_pixels_raises():
    """Too few usable pixels gives a clear error, counting NaN and masked."""
    data = np.full((10, 10), np.nan)
    data[0, :5] = 1.0  # 5 finite pixels; order 2 needs 6
    with pytest.raises(ValueError, match="5 found, at least 6 needed"):
        filters.polynomial_flatten(data, order=2)

    data = np.arange(100.0).reshape(10, 10)
    mask = np.ones_like(data, dtype=bool)
    mask[0, :2] = False
    mask[5, 5] = False
    data[5, 5] = np.nan  # 2 usable pixels; a plane needs 3
    with pytest.raises(ValueError, match="2 found, at least 3 needed"):
        remove_plane_masked(data, mask)


def test_flatten_filter_versions():
    """Changed fits have new versions, recorded in processing provenance."""
    assert filters.remove_plane.__version__ == "0.1.1"
    assert remove_plane_masked.__version__ == "0.1.1"
    assert filters.polynomial_flatten.__version__ == "0.2.0"
    assert polynomial_flatten_masked.__version__ == "0.2.0"
