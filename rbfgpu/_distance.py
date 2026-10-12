"""
Pairwise Euclidean distances on the active XuPy backend.

Distances are the only O(n*m) quantity in an RBF interpolant, so they are
computed with the dot-product identity instead of an explicit difference: it
avoids materialising the (n, m, d) broadcast temporary, which on a GPU is what
actually limits the problem size.
"""

from __future__ import annotations

import math

from ._backend import xp

__all__ = ["default_epsilon", "euclidean_distance"]

#: Memory budget (in floats) for the (block, npoints, ndim) temporary that
#: :func:`default_epsilon` allocates: 2**22 floats is 32 MiB of float64.
_EPSILON_BLOCK_FLOATS = 2**22


def euclidean_distance(x, y, epsilon=None):
    """
    Compute the (optionally scaled) Euclidean distance matrix between two clouds.

    Parameters
    ----------
    x : array, shape (n, d)
        First set of points.
    y : array, shape (m, d)
        Second set of points.
    epsilon : float, optional
        Shape parameter the distances are multiplied by before being fed to the
        kernel.  ``None`` (default) leaves the distances unscaled.

    Returns
    -------
    dist : array, shape (n, m)
        Pairwise distances, on the active backend (NumPy or CuPy).

    Notes
    -----
    The dot-product identity costs one (n, m) temporary instead of the (n, m, d)
    one an explicit difference would need, which is what makes large problems fit
    in device memory.  The price is that a distance that is exactly zero (the
    diagonal of a self-distance matrix, or duplicated points) comes out as
    round-off instead: up to about ``sqrt(machine epsilon) * ||x||``, i.e. ~1e-8
    for unit-scale coordinates.  That is harmless for every kernel here (the
    smallest kernel value it can produce is ``r**2 * log(r)`` ~ 1e-15), but
    anything that needs *exact* zeros, such as :func:`default_epsilon`, must use
    the explicit difference instead.
    """
    x = xp.asarray(x, dtype=xp.float64)
    y = xp.asarray(y, dtype=xp.float64)

    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x.y
    x_sqnorm = (x**2).sum(axis=1, keepdims=True)  # (n, 1)
    y_sqnorm = (y**2).sum(axis=1, keepdims=True)  # (m, 1)

    dist_sq = x_sqnorm + y_sqnorm.T - 2 * (x @ y.T)  # (n, m)
    # Round-off can make the above slightly negative for coincident points.
    dist_sq = xp.clip(dist_sq, 0, None)
    dist = xp.sqrt(dist_sq)

    if epsilon is not None:
        dist = dist * epsilon

    return dist


def default_epsilon(y) -> float:
    """
    Data-driven default shape parameter, for the kernels that need one.

    The classic nearest neighbour heuristic: the square root of the mean distance
    between each point and its closest distinct neighbour.  Small ``epsilon``
    means wide kernels, large ``epsilon`` means narrow ones.

    Parameters
    ----------
    y : array, shape (n, d)
        Data point coordinates, with ``n >= 2``.

    Returns
    -------
    epsilon : float
        Positive shape parameter.

    Raises
    ------
    ValueError
        If there are fewer than two points, or if the estimate comes out
        non-finite or non-positive (every point coincident with another).

    Notes
    -----
    Distances are computed with the explicit difference, block by block, so that
    the self-distances are *exactly* zero and can be discarded: the round-off of
    :func:`euclidean_distance` (~1e-8) would otherwise be mistaken for the
    nearest neighbour distance.  The blocks bound the temporary memory to a few
    tens of MiB whatever ``n`` is.  The final reduction to a Python float forces
    one host synchronisation, paid once at build time.
    """
    y = xp.asarray(y, dtype=xp.float64)
    if y.ndim == 1:
        y = xp.reshape(y, (y.shape[0], 1))
    npoints, ndim = y.shape
    if npoints < 2:
        raise ValueError(
            f"`default_epsilon` needs at least two points, got {npoints}: pass "
            "`epsilon` explicitly."
        )

    block = max(1, int(_EPSILON_BLOCK_FLOATS // max(npoints * ndim, 1)))
    nearest_sum = xp.zeros((), dtype=xp.float64)

    for start in range(0, npoints, block):
        stop = min(start + block, npoints)
        chunk = y[start:stop]

        difference = chunk[:, None, :] - y[None, :, :]  # (block, npoints, ndim)
        dist = xp.sqrt((difference**2).sum(axis=-1))
        del difference

        index = xp.arange(stop - start)
        dist[index, index + start] = xp.inf  # a point is not its own neighbour
        dist = xp.where(dist == 0, xp.inf, dist)  # ...nor are its exact copies
        nearest_sum = nearest_sum + dist.min(axis=1).sum()
        del dist

    epsilon = float(xp.sqrt(nearest_sum / npoints))
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError(
            "Cannot estimate a default shape parameter from `y` (all the points "
            "are coincident). Pass `epsilon` explicitly."
        )
    return epsilon
