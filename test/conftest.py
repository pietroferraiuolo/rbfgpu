"""Shared pytest fixtures."""

import numpy as np
import pytest
import xupy as xp

from ._util import BACKENDS


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded generator, so that the whole suite is reproducible."""
    return np.random.default_rng(20261012)


@pytest.fixture(params=BACKENDS)
def backend(request):
    """
    Run the test once per available XuPy backend.

    ``xp.backend`` is scoped to the current thread, so the global default is left
    untouched and the tests cannot interfere with each other.
    """
    with xp.backend(request.param):
        yield request.param


@pytest.fixture
def points2d(rng) -> np.ndarray:
    """60 scattered points in the unit square (well conditioned, no duplicates)."""
    return rng.random((60, 2))


@pytest.fixture
def values2d(rng, points2d) -> np.ndarray:
    """A smooth 2-D field sampled at :func:`points2d`."""
    return np.sin(3 * points2d[:, 0]) * np.cos(3 * points2d[:, 1])


@pytest.fixture
def grid2d(rng) -> np.ndarray:
    """Evaluation points, partly outside the convex hull of :func:`points2d`."""
    return rng.uniform(-0.5, 1.5, (200, 2))
