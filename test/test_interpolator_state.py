"""Tests for the state export and the backend handling of ``RBFInterpolator``."""

import numpy as np
import pytest

from rbfgpu import RBFInterpolator

from ._util import reference_interpolant, to_numpy
from .test_interpolator import EXACT_TOL


# ---------------------------------------------------------------------------
# State export
# ---------------------------------------------------------------------------
def test_to_cpu_returns_host_arrays_with_the_expected_keys(backend, points2d, values2d):
    state = RBFInterpolator(points2d, values2d).to_cpu()
    assert set(state) == {
        "y",
        "coeffs_rbf",
        "coeffs_poly",
        "powers",
        "degree",
        "kernel",
        "epsilon",
    }
    for key in ("y", "coeffs_rbf", "coeffs_poly"):
        assert isinstance(state[key], np.ndarray), key
    assert state["y"].shape == (60, 2)
    assert state["coeffs_rbf"].shape == (60, 1)
    assert state["coeffs_poly"].shape == (3, 1)
    assert state["kernel"] == "thin_plate_spline"
    assert state["degree"] == 1 and state["epsilon"] == 1.0


def test_to_cpu_state_reproduces_the_interpolant(backend, points2d, values2d, grid2d):
    """The exported state is re-evaluated by an independent NumPy implementation."""
    rbf = RBFInterpolator(points2d, values2d)
    np.testing.assert_allclose(
        reference_interpolant(rbf.to_cpu(), grid2d)[:, 0],
        to_numpy(rbf(grid2d)),
        rtol=1e-6,
        atol=1e-10,
    )


@pytest.mark.parametrize("kernel", ["gaussian", "quintic", "matern5_2"])
def test_to_cpu_state_reproduces_every_kernel(
    backend, points2d, values2d, grid2d, kernel
):
    epsilon = None if kernel == "quintic" else 2.0
    rbf = RBFInterpolator(points2d, values2d, kernel=kernel, epsilon=epsilon)
    np.testing.assert_allclose(
        reference_interpolant(rbf.to_cpu(), grid2d)[:, 0],
        to_numpy(rbf(grid2d)),
        rtol=1e-6,
        atol=1e-10,
    )


def test_to_cpu_without_a_polynomial_tail(backend, points2d, values2d, grid2d):
    rbf = RBFInterpolator(points2d, values2d, kernel="gaussian", epsilon=2.0, degree=-1)
    state = rbf.to_cpu()
    assert state["coeffs_poly"] is None
    assert np.max(np.abs(to_numpy(rbf(points2d)) - values2d)) < EXACT_TOL
    np.testing.assert_allclose(
        reference_interpolant(state, grid2d)[:, 0],
        to_numpy(rbf(grid2d)),
        rtol=1e-6,
        atol=1e-10,
    )


def test_to_cpu_of_a_vector_valued_problem(backend, points2d, rng):
    d = rng.random((60, 3))
    state = RBFInterpolator(points2d, d).to_cpu()
    assert state["coeffs_rbf"].shape == (60, 3)
    assert state["coeffs_poly"].shape == (3, 3)


# ---------------------------------------------------------------------------
# Backend handling
# ---------------------------------------------------------------------------
def test_the_interpolant_lives_on_the_active_backend(backend, points2d, values2d):
    rbf = RBFInterpolator(points2d, values2d)
    assert isinstance(rbf.y, np.ndarray) == (backend == "cpu")
    assert rbf.on_gpu == (backend == "gpu")
    assert isinstance(to_numpy(rbf(points2d)), np.ndarray)
    assert isinstance(rbf(points2d), np.ndarray) == (backend == "cpu")


def test_repr_describes_the_interpolant(backend, points2d, values2d):
    text = repr(RBFInterpolator(points2d, values2d))
    assert text.startswith("RBFInterpolator(")
    for fragment in (
        "kernel='thin_plate_spline'",
        "npoints=60",
        "ndim=2",
        "output_dim=1",
        "degree=1",
        "epsilon=1.0",
        f"backend='{backend}'",
    ):
        assert fragment in text, fragment


def test_two_identical_builds_agree(backend, points2d, values2d, grid2d):
    first = RBFInterpolator(points2d, values2d)
    second = RBFInterpolator(points2d, values2d)
    np.testing.assert_array_equal(to_numpy(first(grid2d)), to_numpy(second(grid2d)))


@pytest.mark.slow
def test_a_large_problem_is_chunked_and_still_exact(backend, rng):
    """The regime the package exists for: thousands of points, wide evaluation."""
    y = rng.random((2000, 2))
    d = np.sin(4 * y[:, 0]) * np.cos(4 * y[:, 1])
    rbf = RBFInterpolator(y, d)
    assert rbf._resolve_chunk_size(50000, None) < 50000
    assert np.max(np.abs(to_numpy(rbf(y)) - d)) < 1e-8

    x = rng.random((50000, 2))
    chunked = to_numpy(rbf(x, chunk_size=1000))
    np.testing.assert_allclose(
        chunked, to_numpy(rbf(x, chunk_size=50000)), rtol=1e-9, atol=1e-12
    )
