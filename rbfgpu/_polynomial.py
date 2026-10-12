"""
Polynomial tail of the RBF interpolant.

The RBF part alone cannot extrapolate and, for conditionally positive definite
kernels, is not even well posed: a polynomial of degree ``degree`` is added to
the interpolant and constrained to be orthogonal to the RBF coefficients
(``P(y)^T a = 0``).  The basis used here is the plain monomial basis
``x**p`` with ``|p| <= degree``.

Any basis spanning the same polynomial space gives the same interpolant, so the
column order below is arbitrary (it differs from SciPy's, which also shifts and
scales the coordinates): the coefficients are consistent because the very same
``powers`` are used to build and to evaluate the system.
"""

from __future__ import annotations

import itertools

from ._backend import xp

__all__ = ["monomial_exponents", "polynomial_matrix"]


def monomial_exponents(ndim: int, degree: int) -> tuple:
    """
    Exponent tuples of every monomial of total degree <= ``degree``.

    Parameters
    ----------
    ndim : int
        Number of coordinates of the points.
    degree : int
        Polynomial degree.  ``-1`` means "no polynomial terms at all", ``0``
        only the constant term.

    Returns
    -------
    powers : tuple of tuple of int
        ``len(powers)`` is the number of polynomial terms, i.e.
        ``C(ndim + degree, degree)``.  Empty for ``degree == -1``.

    Raises
    ------
    ValueError
        If ``degree < -1``, or ``ndim < 1``.
    """
    if not isinstance(degree, (int,)) or isinstance(degree, bool):
        degree = int(degree)
    if degree < -1:
        raise ValueError(f"`degree` must be >= -1 (-1 means no polynomial), got {degree}.")
    if ndim < 1:
        raise ValueError(f"`ndim` must be >= 1, got {ndim}.")
    if degree == -1:
        return ()

    return tuple(
        powers
        for powers in itertools.product(range(degree + 1), repeat=ndim)
        if sum(powers) <= degree
    )


def polynomial_matrix(x, powers) -> object:
    """
    Evaluate a monomial basis at the points ``x``.

    Parameters
    ----------
    x : array, shape (n, d)
        Point coordinates.
    powers : sequence of tuple of int
        Exponents of the monomials, as returned by :func:`monomial_exponents`.

    Returns
    -------
    P : array, shape (n, len(powers))
        ``P[i, j]`` is ``prod_k x[i, k] ** powers[j][k]``.  For an empty
        ``powers`` an ``(n, 0)`` array is returned.
    """
    x = xp.asarray(x)
    n, ndim = x.shape

    if len(powers) == 0:
        return xp.empty((n, 0), dtype=x.dtype)

    columns = []
    for exponents in powers:
        if len(exponents) != ndim:
            raise ValueError(
                f"`powers` holds {len(exponents)}-dimensional exponents but `x` "
                f"has {ndim} columns."
            )
        column = xp.ones(n, dtype=x.dtype)
        for axis, exponent in enumerate(exponents):
            if exponent == 0:
                continue
            if exponent == 1:
                column = column * x[:, axis]
            else:
                column = column * x[:, axis] ** exponent
        columns.append(column)

    return xp.column_stack(columns)
