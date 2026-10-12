"""Tests for ``rbfgpu._distance``."""

import numpy as np
import pytest

from rbfgpu import default_epsilon, euclidean_distance

from ._util import reference_distance, to_numpy


def test_matches_brute_force(backend, rng):
    x = rng.random((25, 3)) * 10
    y = rng.random((17, 3)) * 10
    got = to_numpy(euclidean_distance(x, y))
    np.testing.assert_allclose(got, reference_distance(x, y), rtol=1e-12, atol=1e-12)


def test_shape_and_square_case(backend, rng):
    x = rng.random((9, 2))
    assert to_numpy(euclidean_distance(x, rng.random((4, 2)))).shape == (9, 4)
    assert to_numpy(euclidean_distance(x, x)).shape == (9, 9)


def test_symmetry_and_zero_diagonal(backend, points2d):
    dist = to_numpy(euclidean_distance(points2d, points2d))
    np.testing.assert_allclose(dist, dist.T, rtol=0, atol=0)
    # The dot-product identity cannot make a zero distance exactly zero: the
    # diagonal comes out as round-off, bounded by sqrt(machine eps) * ||y||.
    tolerance = 1e-7 * max(1.0, float(np.abs(points2d).max()))
    assert np.max(np.diag(dist)) < tolerance
    np.testing.assert_allclose(np.diag(dist), 0.0, atol=tolerance)


def test_epsilon_scales_the_distances(backend, rng):
    x, y = rng.random((11, 2)), rng.random((7, 2))
    unscaled = to_numpy(euclidean_distance(x, y))
    scaled = to_numpy(euclidean_distance(x, y, 2.5))
    np.testing.assert_allclose(scaled, 2.5 * unscaled, rtol=1e-12)


def test_duplicate_points_are_at_round_off_distance(backend, points2d):
    y = np.vstack([points2d, points2d])
    dist = to_numpy(euclidean_distance(y, y))
    n = len(points2d)
    duplicates = dist[np.arange(n), np.arange(n) + n]
    assert np.max(duplicates) < 1e-7


def test_no_negative_or_non_finite_values(backend, rng):
    # Large offsets are where the ||x||^2 + ||y||^2 - 2xy identity loses digits:
    # a zero distance comes out as ~sqrt(eps) * ||x|| instead of exactly zero.
    x = rng.random((30, 2)) * 1e4 + 1e4
    dist = to_numpy(euclidean_distance(x, x))
    assert np.all(dist >= 0)
    assert np.all(np.isfinite(dist))
    np.testing.assert_allclose(
        dist, reference_distance(x, x), rtol=1e-9, atol=1e-2
    )


def test_accepts_lists_and_int_arrays(backend):
    dist = to_numpy(euclidean_distance([[0, 0], [3, 4]], [[0, 0]]))
    np.testing.assert_allclose(dist, [[0.0], [5.0]], rtol=1e-12, atol=1e-12)
    assert dist.dtype == np.float64


def test_default_epsilon_matches_the_heuristic(backend, points2d):
    dist = reference_distance(points2d, points2d)
    dist[dist == 0.0] = np.inf
    expected = float(np.sqrt(np.mean(np.min(dist, axis=1))))
    assert default_epsilon(points2d) == pytest.approx(expected, rel=1e-12)


def test_default_epsilon_is_positive_and_finite(backend, rng):
    epsilon = default_epsilon(rng.random((50, 4)))
    assert np.isfinite(epsilon) and epsilon > 0


def test_default_epsilon_ignores_exact_duplicates(backend, points2d):
    """A duplicated point's neighbour is its next distinct one, not its copy."""
    duplicated = np.vstack([points2d, points2d])
    assert default_epsilon(duplicated) == pytest.approx(
        default_epsilon(points2d), rel=1e-9
    )


def test_default_epsilon_accepts_a_flat_array(backend):
    y = np.linspace(0.0, 1.0, 11)
    assert default_epsilon(y) == pytest.approx(default_epsilon(y[:, None]), rel=1e-12)


def test_default_epsilon_blocks_large_clouds(backend, rng):
    """Enough points to need several blocks: the result must not depend on it."""
    y = rng.random((3000, 2))
    dist = reference_distance(y, y)
    dist[dist == 0.0] = np.inf
    expected = float(np.sqrt(np.mean(np.min(dist, axis=1))))
    assert default_epsilon(y) == pytest.approx(expected, rel=1e-9)


def test_default_epsilon_raises_for_coincident_points(backend):
    with pytest.raises(ValueError, match="Cannot estimate a default shape parameter"):
        default_epsilon(np.zeros((5, 2)))


def test_default_epsilon_raises_for_a_single_point(backend):
    with pytest.raises(ValueError, match="at least two points"):
        default_epsilon(np.zeros((1, 2)))
