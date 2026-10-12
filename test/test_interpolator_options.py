"""Tests for the ``RBFInterpolator`` options: chunking and smoothing."""

import numpy as np
import pytest
import xupy as xp

from rbfgpu import RBFInterpolator, default_epsilon, monomial_exponents

from ._util import reference_polynomial, to_numpy
from .test_interpolator import EXACT_TOL


# ---------------------------------------------------------------------------
# Chunked evaluation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("chunk_size", [1, 3, 17, 60, 199, 200, 10**6])
def test_explicit_chunk_size_does_not_change_the_result(
    backend, points2d, values2d, grid2d, chunk_size
):
    rbf = RBFInterpolator(points2d, values2d)
    reference = to_numpy(rbf(grid2d))
    # Chunking changes the shape of the BLAS calls, so the two paths agree to
    # round-off rather than bit for bit.
    np.testing.assert_allclose(
        to_numpy(rbf(grid2d, chunk_size=chunk_size)), reference, rtol=1e-10, atol=1e-12
    )


def test_a_tiny_memory_budget_forces_many_chunks(backend, points2d, values2d, grid2d):
    rbf = RBFInterpolator(points2d, values2d, max_chunk_bytes=1024, min_chunk_size=1)
    assert rbf._resolve_chunk_size(200, None) < 200
    np.testing.assert_allclose(
        to_numpy(rbf(grid2d)),
        to_numpy(rbf(grid2d, chunk_size=200)),
        rtol=1e-10,
        atol=1e-12,
    )


def test_memory_efficient_false_evaluates_in_one_batch(backend, points2d, values2d):
    rbf = RBFInterpolator(points2d, values2d, memory_efficient=False)
    assert rbf._resolve_chunk_size(123456, None) == 123456


def test_resolve_chunk_size_rules(backend, points2d, values2d):
    rbf = RBFInterpolator(points2d, values2d)
    # An explicit size is clamped to [1, n_eval].
    assert rbf._resolve_chunk_size(100, 10) == 10
    assert rbf._resolve_chunk_size(100, 0) == 1
    assert rbf._resolve_chunk_size(100, -5) == 1
    assert rbf._resolve_chunk_size(100, 10**6) == 100
    # The estimate is never below the configured floor and never above n_eval.
    small = RBFInterpolator(points2d, values2d, max_chunk_bytes=1, min_chunk_size=8)
    assert small._resolve_chunk_size(100, None) == 8
    assert small._resolve_chunk_size(3, None) == 3
    # A bigger budget gives a bigger (or equal) chunk.
    big = RBFInterpolator(points2d, values2d, max_chunk_bytes=10**9, min_chunk_size=1)
    assert big._resolve_chunk_size(10**6, None) >= small._resolve_chunk_size(10**6, None)


def test_chunking_a_multi_output_problem(backend, points2d, rng):
    d = rng.random((60, 4))
    x = rng.random((37, 2))
    rbf = RBFInterpolator(points2d, d, max_chunk_bytes=4096, min_chunk_size=1)
    reference = RBFInterpolator(points2d, d)(x)
    np.testing.assert_allclose(
        to_numpy(rbf(x)), to_numpy(reference), rtol=1e-10, atol=1e-12
    )


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------
def test_zero_smoothing_interpolates_exactly(backend, points2d, values2d):
    rbf = RBFInterpolator(points2d, values2d, smoothing=0.0)
    assert np.max(np.abs(to_numpy(rbf(points2d)) - values2d)) < EXACT_TOL


def test_smoothing_relaxes_the_fit(backend, points2d, values2d):
    residuals = []
    for smoothing in (0.0, 1e-2, 1e0, 1e2):
        rbf = RBFInterpolator(points2d, values2d, smoothing=smoothing)
        residuals.append(float(np.max(np.abs(to_numpy(rbf(points2d)) - values2d))))
    assert residuals[0] < EXACT_TOL
    assert residuals[1] > residuals[0]
    assert residuals[2] > residuals[1]
    assert residuals[3] > residuals[2]


def test_array_smoothing_matches_the_equivalent_scalar(backend, points2d, values2d):
    scalar = RBFInterpolator(points2d, values2d, smoothing=0.25)
    array = RBFInterpolator(points2d, values2d, smoothing=np.full(60, 0.25))
    np.testing.assert_allclose(
        to_numpy(scalar.smoothing), to_numpy(array.smoothing), rtol=0, atol=0
    )
    np.testing.assert_allclose(
        to_numpy(array(points2d)), to_numpy(scalar(points2d)), rtol=1e-12
    )


def test_heavy_smoothing_approaches_a_polynomial_least_squares_fit(
    backend, points2d, values2d, grid2d
):
    rbf = RBFInterpolator(points2d, values2d, smoothing=1e8, degree=1)
    powers = monomial_exponents(2, 1)
    coeffs, *_ = np.linalg.lstsq(
        reference_polynomial(points2d, powers), values2d, rcond=None
    )
    expected = reference_polynomial(grid2d, powers) @ coeffs
    np.testing.assert_allclose(to_numpy(rbf(grid2d)), expected, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# Shape parameter
# ---------------------------------------------------------------------------
def test_epsilon_defaults_to_one_for_scale_invariant_kernels(backend, points2d, values2d):
    for kernel in ("linear", "thin_plate_spline", "cubic", "quintic"):
        assert RBFInterpolator(points2d, values2d, kernel=kernel).epsilon == 1.0


def test_epsilon_defaults_to_the_data_driven_value(backend, points2d, values2d):
    expected = default_epsilon(points2d)
    for kernel in ("gaussian", "multiquadric", "inverse_quadratic", "matern3_2"):
        got = RBFInterpolator(points2d, values2d, kernel=kernel).epsilon
        assert got == pytest.approx(expected, rel=1e-12)


def test_explicit_epsilon_is_kept_and_changes_the_interpolant(
    backend, points2d, values2d, grid2d
):
    narrow = RBFInterpolator(points2d, values2d, kernel="gaussian", epsilon=5.0)
    wide = RBFInterpolator(points2d, values2d, kernel="gaussian", epsilon=0.5)
    assert narrow.epsilon == 5.0 and wide.epsilon == 0.5
    assert not np.allclose(to_numpy(narrow(grid2d)), to_numpy(wide(grid2d)))


def test_a_narrow_gaussian_interpolates_better_than_a_wide_one(
    backend, points2d, values2d
):
    """The default epsilon is a heuristic, not a guarantee: conditioning matters."""
    narrow = RBFInterpolator(points2d, values2d, kernel="gaussian", epsilon=2.0)
    wide = RBFInterpolator(points2d, values2d, kernel="gaussian", epsilon=0.1)
    narrow_residual = np.max(np.abs(to_numpy(narrow(points2d)) - values2d))
    wide_residual = np.max(np.abs(to_numpy(wide(points2d)) - values2d))
    assert narrow_residual < wide_residual


# ---------------------------------------------------------------------------
# Custom kernels
# ---------------------------------------------------------------------------
def _matern1_2(r):
    return xp.exp(-r)


def test_a_callable_kernel_needs_epsilon_and_degree(backend, points2d, values2d):
    with pytest.raises(ValueError, match="`epsilon` must be given explicitly"):
        RBFInterpolator(points2d, values2d, kernel=_matern1_2, degree=-1)
    with pytest.raises(ValueError, match="`degree` must be given explicitly"):
        RBFInterpolator(points2d, values2d, kernel=_matern1_2, epsilon=1.0)


def test_a_callable_kernel_is_used(backend, points2d, values2d):
    rbf = RBFInterpolator(
        points2d, values2d, kernel=_matern1_2, epsilon=1.0, degree=-1
    )
    assert rbf.kernel == "_matern1_2"
    assert np.max(np.abs(to_numpy(rbf(points2d)) - values2d)) < EXACT_TOL
    # ... and it agrees with the built-in kernel of the same shape.
    builtin = RBFInterpolator(
        points2d, values2d, kernel="matern1_2", epsilon=1.0, degree=-1
    )
    np.testing.assert_allclose(
        to_numpy(rbf(points2d)), to_numpy(builtin(points2d)), rtol=1e-10
    )
