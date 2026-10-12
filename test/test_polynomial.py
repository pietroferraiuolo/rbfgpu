"""Tests for ``rbfgpu._polynomial``."""

import math

import numpy as np
import pytest

from rbfgpu import monomial_exponents, polynomial_matrix

from ._util import reference_polynomial, to_numpy

CASES = [
    (1, 0),
    (1, 1),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
    (2, 3),
    (3, 1),
    (3, 2),
    (4, 2),
]


@pytest.mark.parametrize("ndim, degree", CASES)
def test_number_of_monomials(ndim, degree):
    powers = monomial_exponents(ndim, degree)
    assert len(powers) == math.comb(ndim + degree, degree)


@pytest.mark.parametrize("ndim, degree", CASES)
def test_exponents_are_valid(ndim, degree):
    powers = monomial_exponents(ndim, degree)
    assert len(set(powers)) == len(powers), "duplicated monomials"
    for exponents in powers:
        assert len(exponents) == ndim
        assert all(e >= 0 for e in exponents)
        assert sum(exponents) <= degree


def test_no_polynomial_for_degree_minus_one():
    assert monomial_exponents(2, -1) == ()
    assert monomial_exponents(1, -1) == ()


def test_degree_zero_is_the_constant_term():
    assert monomial_exponents(3, 0) == ((0, 0, 0),)


@pytest.mark.parametrize("degree", [-2, -5])
def test_degree_below_minus_one_raises(degree):
    with pytest.raises(ValueError, match="`degree` must be >= -1"):
        monomial_exponents(2, degree)


def test_ndim_below_one_raises():
    with pytest.raises(ValueError, match="`ndim` must be >= 1"):
        monomial_exponents(0, 1)


def test_matrix_matches_reference(backend, rng):
    x = rng.random((23, 3))
    for degree in (0, 1, 2):
        powers = monomial_exponents(3, degree)
        got = to_numpy(polynomial_matrix(x, powers))
        np.testing.assert_allclose(
            got, reference_polynomial(x, powers), rtol=1e-12, atol=1e-12
        )


def test_matrix_shape(backend, points2d):
    powers = monomial_exponents(2, 2)
    assert to_numpy(polynomial_matrix(points2d, powers)).shape == (60, len(powers))


def test_empty_powers_gives_an_empty_matrix(backend, points2d):
    out = to_numpy(polynomial_matrix(points2d, ()))
    assert out.shape == (60, 0)


def test_degree_one_spans_one_x_and_y(backend, rng):
    x = rng.random((13, 2))
    got = to_numpy(polynomial_matrix(x, monomial_exponents(2, 1)))
    assert got.shape == (13, 3)
    np.testing.assert_allclose(np.ones(13), got[:, 0])
    remaining = {tuple(np.round(got[:, j], 12)) for j in (1, 2)}
    expected = {tuple(np.round(x[:, k], 12)) for k in (0, 1)}
    assert remaining == expected


def test_degree_zero_is_all_ones(backend, points2d):
    got = to_numpy(polynomial_matrix(points2d, monomial_exponents(2, 0)))
    np.testing.assert_allclose(got, np.ones((60, 1)))


def test_integer_input_keeps_its_dtype(backend):
    x = np.arange(12, dtype=np.int32).reshape(6, 2)
    got = to_numpy(polynomial_matrix(x, monomial_exponents(2, 1)))
    assert got.dtype == np.int32


def test_powers_with_the_wrong_dimension_raise(backend, points2d):
    with pytest.raises(ValueError, match="3-dimensional exponents"):
        polynomial_matrix(points2d, ((0, 0, 0),))
