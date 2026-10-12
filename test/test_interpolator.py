"""Tests for ``rbfgpu.RBFInterpolator``: interpolation and polynomial tail."""

import warnings

import numpy as np
import pytest

from rbfgpu import RBFInterpolator, available_kernels, monomial_exponents

from ._util import to_numpy

#: Scale invariant kernels do not need an explicit shape parameter; the others
#: are given a narrow enough epsilon to keep the Gram matrix well conditioned.
SCALE_INVARIANT = ("linear", "thin_plate_spline", "cubic", "quintic")
KERNEL_EPSILON = {
    name: (None if name in SCALE_INVARIANT else 2.0) for name in available_kernels()
}
ALL_KERNELS = sorted(KERNEL_EPSILON)

#: Tolerance for "exact" interpolation: the linear system is solved to round-off,
#: but a wide kernel can amplify that round-off by a few orders of magnitude.
EXACT_TOL = 1e-8


def build(y, d, **kwargs) -> RBFInterpolator:
    """Build an interpolator, silencing the expected ill-posed degree warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return RBFInterpolator(y, d, **kwargs)


@pytest.mark.parametrize("kernel", ALL_KERNELS)
def test_interpolates_the_data_at_the_nodes(backend, kernel, points2d, values2d):
    rbf = build(points2d, values2d, kernel=kernel, epsilon=KERNEL_EPSILON[kernel])
    residual = to_numpy(rbf(points2d)) - values2d
    assert np.max(np.abs(residual)) < EXACT_TOL


@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
def test_works_in_several_dimensions(backend, rng, ndim):
    y = rng.random((40, ndim))
    d = np.sin(2 * y.sum(axis=1))
    rbf = RBFInterpolator(y, d)
    assert np.max(np.abs(to_numpy(rbf(y)) - d)) < EXACT_TOL
    assert rbf.ndim == ndim


def test_one_dimensional_problem_accepts_flat_arrays(backend, rng):
    y = np.sort(rng.random(30))
    d = np.cos(5 * y)
    rbf = RBFInterpolator(y, d)
    assert rbf.ndim == 1 and rbf.npoints == 30
    assert to_numpy(rbf(np.linspace(0, 1, 11))).shape == (11,)
    assert np.max(np.abs(to_numpy(rbf(y)) - d)) < EXACT_TOL


def test_flat_x_is_a_single_point_when_ndim_is_larger(backend, points2d, values2d):
    rbf = RBFInterpolator(points2d, values2d)
    assert to_numpy(rbf(np.array([0.5, 0.5]))).shape == (1,)


def test_vector_valued_data(backend, points2d, rng):
    d = rng.random((60, 3))
    rbf = RBFInterpolator(points2d, d)
    out = to_numpy(rbf(points2d))
    assert out.shape == (60, 3)
    assert np.max(np.abs(out - d)) < EXACT_TOL
    assert rbf.output_dim == 3


def test_data_with_trailing_dimensions_keeps_its_shape(backend, points2d, rng, grid2d):
    d = rng.random((60, 2, 3))
    rbf = RBFInterpolator(points2d, d)
    assert rbf.d_shape == (2, 3)
    assert to_numpy(rbf(grid2d)).shape == (200, 2, 3)
    assert np.max(np.abs(to_numpy(rbf(points2d)) - d)) < EXACT_TOL


def test_data_with_trailing_dimensions_matches_the_flat_problem(backend, points2d, rng):
    d = rng.random((60, 2, 3))
    grid = rng.random((25, 2))
    wide = RBFInterpolator(points2d, d)
    flat = RBFInterpolator(points2d, d.reshape(60, -1))
    np.testing.assert_allclose(
        to_numpy(wide(grid)).reshape(25, -1), to_numpy(flat(grid)), rtol=1e-12
    )


def test_integer_input_is_promoted_to_float64(backend):
    y = np.arange(12).reshape(-1, 1)
    d = np.arange(12) ** 2
    rbf = RBFInterpolator(y, d, degree=2)
    assert rbf.d.dtype == np.float64 and rbf.y.dtype == np.float64
    assert to_numpy(rbf(y)).dtype == np.float64
    assert np.max(np.abs(to_numpy(rbf(y)) - d)) < EXACT_TOL


# ---------------------------------------------------------------------------
# Polynomial tail
# ---------------------------------------------------------------------------
def test_reproduces_a_plane_exactly(backend, points2d, grid2d):
    """The degree-1 tail reproduces a linear field, also outside the data hull."""
    plane = 1.5 - 2.0 * points2d[:, 0] + 0.5 * points2d[:, 1]
    expected = 1.5 - 2.0 * grid2d[:, 0] + 0.5 * grid2d[:, 1]
    for kernel in ("thin_plate_spline", "linear", "cubic", "gaussian", "multiquadric"):
        rbf = RBFInterpolator(
            points2d, plane, kernel=kernel, degree=1, epsilon=KERNEL_EPSILON[kernel]
        )
        assert np.max(np.abs(to_numpy(rbf(grid2d)) - expected)) < EXACT_TOL


def test_reproduces_a_quadratic_with_degree_two(backend, points2d, grid2d):
    quad = (
        1.0
        + 0.5 * points2d[:, 0]
        - points2d[:, 1]
        + 2 * points2d[:, 0] ** 2
        - 3 * points2d[:, 0] * points2d[:, 1]
        + 0.25 * points2d[:, 1] ** 2
    )
    expected = (
        1.0
        + 0.5 * grid2d[:, 0]
        - grid2d[:, 1]
        + 2 * grid2d[:, 0] ** 2
        - 3 * grid2d[:, 0] * grid2d[:, 1]
        + 0.25 * grid2d[:, 1] ** 2
    )
    rbf = RBFInterpolator(points2d, quad, degree=2)
    assert np.max(np.abs(to_numpy(rbf(grid2d)) - expected)) < EXACT_TOL


@pytest.mark.parametrize("degree", [-1, 0, 1, 2, 3])
def test_degree_sets_the_number_of_polynomial_terms(backend, points2d, values2d, degree):
    rbf = build(points2d, values2d, degree=degree)
    powers = monomial_exponents(2, degree)
    assert rbf.degree == degree
    assert rbf.n_poly == len(powers)
    assert rbf._powers == powers
    if degree == -1:
        assert rbf.coeffs_poly is None
    else:
        assert to_numpy(rbf.coeffs_poly).shape == (len(powers), 1)
    assert to_numpy(rbf.coeffs_rbf).shape == (60, 1)


def test_degree_minus_one_still_interpolates(backend, points2d, values2d):
    rbf = build(points2d, values2d, degree=-1)
    assert np.max(np.abs(to_numpy(rbf(points2d)) - values2d)) < EXACT_TOL


def test_a_degree_below_the_kernel_minimum_warns(backend, points2d, values2d):
    """SciPy's rule: warn for -1 < degree < min_degree, never for degree=-1."""
    with pytest.warns(UserWarning, match="`degree` should not be below 1"):
        RBFInterpolator(points2d, values2d, degree=0)
    with pytest.warns(UserWarning, match="`degree` should not be below 2"):
        RBFInterpolator(points2d, values2d, kernel="quintic", degree=1)


def test_degree_minus_one_never_warns(backend, points2d, values2d):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        rbf = RBFInterpolator(points2d, values2d, degree=-1)
    assert rbf.n_poly == 0 and rbf.coeffs_poly is None
    assert np.max(np.abs(to_numpy(rbf(points2d)) - values2d)) < EXACT_TOL


def test_a_warned_degree_still_produces_an_interpolant(backend, points2d, values2d):
    with pytest.warns(UserWarning):
        rbf = RBFInterpolator(points2d, values2d, degree=0)
    assert rbf.n_poly == 1
    assert np.max(np.abs(to_numpy(rbf(points2d)) - values2d)) < EXACT_TOL


def test_the_default_degree_never_warns(backend, points2d, values2d):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        for kernel in ALL_KERNELS:
            rbf = RBFInterpolator(
                points2d, values2d, kernel=kernel, epsilon=KERNEL_EPSILON[kernel]
            )
            assert rbf.degree == max(rbf._kernel.min_degree, 0)
