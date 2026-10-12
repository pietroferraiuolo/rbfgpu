"""Tests for ``rbfgpu._kernels``."""

import numpy as np
import pytest

from rbfgpu import KERNELS, KERNEL_ALIASES, Kernel, available_kernels, resolve_kernel
from rbfgpu._backend import xp

from ._util import numpy_kernels, to_numpy

# name -> (phi(0), phi(1), phi(2))
EXPECTED_VALUES = {
    "linear": (0.0, -1.0, -2.0),
    "thin_plate_spline": (0.0, 0.0, 4 * np.log(2)),
    "cubic": (0.0, 1.0, 8.0),
    "quintic": (0.0, -1.0, -32.0),
    "multiquadric": (-1.0, -np.sqrt(2), -np.sqrt(5)),
    "inverse_multiquadric": (1.0, 1 / np.sqrt(2), 1 / np.sqrt(5)),
    "inverse_quadratic": (1.0, 0.5, 0.2),
    "gaussian": (1.0, np.exp(-1), np.exp(-4)),
    "matern1_2": (1.0, np.exp(-1), np.exp(-2)),
    "matern3_2": (
        1.0,
        (1 + np.sqrt(3)) * np.exp(-np.sqrt(3)),
        (1 + 2 * np.sqrt(3)) * np.exp(-2 * np.sqrt(3)),
    ),
    "matern5_2": (
        1.0,
        (1 + np.sqrt(5) + 5 / 3) * np.exp(-np.sqrt(5)),
        (1 + 2 * np.sqrt(5) + 20 / 3) * np.exp(-2 * np.sqrt(5)),
    ),
}

# Minimum polynomial degree documented by SciPy for the conditionally positive
# definite kernels; -1 means "no constraint" (positive definite kernels).
EXPECTED_MIN_DEGREE = {
    "linear": 0,
    "thin_plate_spline": 1,
    "cubic": 1,
    "quintic": 2,
    "multiquadric": 0,
    "inverse_multiquadric": -1,
    "inverse_quadratic": -1,
    "gaussian": -1,
    "matern1_2": -1,
    "matern3_2": -1,
    "matern5_2": -1,
}

EXPECTED_SCALE_INVARIANT = {"linear", "thin_plate_spline", "cubic", "quintic"}


def test_registry_is_complete_and_consistent():
    assert available_kernels() == tuple(sorted(EXPECTED_VALUES))
    for name, kernel in KERNELS.items():
        assert isinstance(kernel, Kernel)
        assert kernel.name == name
        assert callable(kernel.function)


@pytest.mark.parametrize("name", sorted(EXPECTED_VALUES))
def test_kernel_values(name):
    phi = KERNELS[name].function
    r = xp.asarray([0.0, 1.0, 2.0])
    got = to_numpy(phi(r))
    np.testing.assert_allclose(got, EXPECTED_VALUES[name], rtol=1e-13, atol=1e-14)
    assert np.all(np.isfinite(got))


@pytest.mark.parametrize("name", sorted(EXPECTED_VALUES))
def test_kernel_matches_the_numpy_reference(name, rng):
    """Cross-check against an independent NumPy implementation."""
    r = np.concatenate([[0.0], rng.random(50) * 5, [1e-8, 1e8]])
    got = to_numpy(KERNELS[name].function(xp.asarray(r)))
    np.testing.assert_allclose(got, numpy_kernels()[name](r), rtol=1e-12, atol=1e-14)


def test_thin_plate_spline_has_no_nan_at_zero():
    r = xp.asarray([0.0, 0.0, 1e-300, 1.0])
    got = to_numpy(KERNELS["thin_plate_spline"].function(r))
    assert np.all(np.isfinite(got))
    assert got[0] == 0.0 and got[1] == 0.0 and got[2] == 0.0


@pytest.mark.parametrize("name, min_degree", sorted(EXPECTED_MIN_DEGREE.items()))
def test_min_degree_table(name, min_degree):
    assert KERNELS[name].min_degree == min_degree


@pytest.mark.parametrize("name", sorted(EXPECTED_MIN_DEGREE))
def test_default_degree(name):
    kernel = KERNELS[name]
    expected = max(kernel.min_degree, 0)
    assert kernel.default_degree == expected
    assert kernel.scale_invariant == (name in EXPECTED_SCALE_INVARIANT)


@pytest.mark.parametrize("alias, canonical", sorted(KERNEL_ALIASES.items()))
def test_aliases_resolve(alias, canonical):
    assert resolve_kernel(alias).name == canonical


def test_kernel_names_are_case_insensitive_and_stripped():
    assert resolve_kernel("  Thin_Plate_Spline ").name == "thin_plate_spline"
    assert resolve_kernel("GAUSSIAN").name == "gaussian"


def test_unknown_kernel_raises_value_error():
    with pytest.raises(ValueError, match="Unknown kernel 'nope'") as error:
        resolve_kernel("nope")
    assert "thin_plate_spline" in str(error.value)


def test_non_string_non_callable_raises_type_error():
    with pytest.raises(TypeError, match="must be a kernel name"):
        resolve_kernel(3.5)


def test_callable_is_wrapped_and_requires_explicit_parameters():
    def my_kernel(r):
        return xp.exp(-r)

    kernel = resolve_kernel(my_kernel)
    assert kernel.name == "my_kernel"
    assert kernel.function is my_kernel
    assert kernel.min_degree is None
    assert kernel.default_degree is None
    assert kernel.scale_invariant is False


def test_callable_without_a_name_is_still_usable():
    kernel = resolve_kernel(lambda r: -r)
    assert kernel.name == "<lambda>"
    assert kernel.min_degree is None
