"""Tests for the input validation of ``rbfgpu.RBFInterpolator``."""

import re

import numpy as np
import pytest

from rbfgpu import RBFInterpolator


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"kernel": "nope"}, "Unknown kernel 'nope'"),
        ({"degree": -2}, "`degree` must be >= -1"),
        ({"kernel": "gaussian", "epsilon": 0.0}, "positive finite"),
        ({"kernel": "gaussian", "epsilon": -1.0}, "positive finite"),
        ({"kernel": "gaussian", "epsilon": float("nan")}, "positive finite"),
        ({"kernel": "gaussian", "epsilon": float("inf")}, "positive finite"),
        ({"smoothing": -1.0}, "`smoothing` must be non-negative"),
        ({"smoothing": np.zeros(5)}, "must be a scalar or an array of shape"),
        ({"max_chunk_bytes": 0}, "`max_chunk_bytes` must be positive"),
        ({"max_chunk_bytes": -1}, "`max_chunk_bytes` must be positive"),
        ({"min_chunk_size": 0}, "`min_chunk_size` must be positive"),
    ],
)
def test_invalid_options_raise_value_error(backend, points2d, values2d, kwargs, match):
    with pytest.raises(ValueError, match=re.escape(match)):
        RBFInterpolator(points2d, values2d, **kwargs)


def test_a_non_string_non_callable_kernel_raises_type_error(backend, points2d, values2d):
    with pytest.raises(TypeError, match="must be a kernel name"):
        RBFInterpolator(points2d, values2d, kernel=3.5)


def test_data_with_the_wrong_length_raises(backend, points2d, values2d):
    with pytest.raises(ValueError, match="same length along its first axis"):
        RBFInterpolator(points2d, values2d[:-1])


def test_an_empty_point_cloud_raises(backend):
    with pytest.raises(ValueError, match="at least one point"):
        RBFInterpolator(np.empty((0, 2)), np.empty(0))


def test_a_three_dimensional_y_raises(backend, rng):
    with pytest.raises(ValueError, match="1-D or 2-D array"):
        RBFInterpolator(rng.random((4, 2, 2)), rng.random(4))


def test_evaluation_points_with_the_wrong_dimension_raise(
    backend, points2d, values2d, rng
):
    rbf = RBFInterpolator(points2d, values2d)
    with pytest.raises(ValueError, match="second axis of `x` to have length 2"):
        rbf(rng.random((5, 3)))
    with pytest.raises(ValueError, match="2-D array of shape"):
        rbf(rng.random((5, 2, 2)))


def test_a_singular_system_raises_runtime_error(backend, points2d):
    """LAPACK raises, cuSOLVER returns NaNs: both must end up as a RuntimeError."""
    y = np.vstack([points2d[0], points2d[0]])
    with pytest.raises(RuntimeError, match="singular"):
        RBFInterpolator(y, np.zeros(2))


def test_a_singular_system_raises_whatever_the_data(backend, points2d, rng):
    """The check must not depend on the right hand side being zero."""
    y = np.vstack([points2d[:5], points2d[0]])
    with pytest.raises(RuntimeError, match="singular"):
        RBFInterpolator(y, rng.random(6))


def test_building_does_not_modify_the_inputs(backend, points2d, values2d):
    y_copy, d_copy = points2d.copy(), values2d.copy()
    RBFInterpolator(points2d, values2d)
    np.testing.assert_array_equal(points2d, y_copy)
    np.testing.assert_array_equal(values2d, d_copy)
