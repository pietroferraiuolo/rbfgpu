"""
Differential tests against ``scipy.interpolate.RBFInterpolator``.

SciPy is an optional test dependency: the whole module is skipped when it is not
installed.  These are the tests that pin rbfgpu to the reference implementation
of the same mathematical object, so they are the ones to look at first if a
kernel or a default is ever changed.
"""

import warnings

import numpy as np
import pytest

from rbfgpu import KERNELS, RBFInterpolator, available_kernels, default_epsilon

from ._util import to_numpy

scipy_interpolate = pytest.importorskip(
    "scipy.interpolate", reason="SciPy is an optional test dependency"
)
SciPyRBFInterpolator = scipy_interpolate.RBFInterpolator

# (kernel, epsilon, degree, smoothing).  epsilon and degree are always explicit
# so that the comparison never depends on either package's defaulting rules
# (those are tested separately).  Every kernel is well conditioned for
# `epsilon=2.0` on 60 points of the unit square: a wider kernel makes the system
# nearly singular, where the two implementations legitimately drift apart.
CASES = [
    ("thin_plate_spline", 1.0, 1, 0.0),
    ("thin_plate_spline", 1.0, -1, 0.0),
    ("thin_plate_spline", 0.5, 2, 0.0),
    ("linear", 1.0, 0, 0.0),
    ("linear", 2.0, 1, 0.0),
    ("cubic", 1.0, 1, 0.0),
    ("quintic", 1.0, 2, 0.0),
    ("multiquadric", 2.0, 0, 0.0),
    ("multiquadric", 2.0, 1, 1e-3),
    ("inverse_multiquadric", 2.0, -1, 0.0),
    ("inverse_quadratic", 2.0, 0, 0.0),
    ("gaussian", 5.0, 0, 0.0),
    ("gaussian", 5.0, 1, 1e-2),
    ("matern1_2", 2.0, -1, 0.0),
    ("matern3_2", 2.0, 0, 0.0),
    ("matern5_2", 2.0, 0, 1e-3),
]

#: Both implementations solve the same linear system, but with a different
#: polynomial basis (SciPy shifts and scales the coordinates first), so the
#: agreement is at round-off level amplified by the conditioning of the system.
RTOL, ATOL = 1e-6, 1e-7


def build_scipy(y, d, **kwargs):
    """Build SciPy's interpolator, skipping when this SciPy lacks the options."""
    try:
        return SciPyRBFInterpolator(y, d, **kwargs)
    except ValueError as error:  # unknown kernel on an older SciPy, bad degree...
        pytest.skip(f"SciPy refuses these options: {error}")


def build_ours(y, d, **kwargs):
    return RBFInterpolator(y, d, **kwargs)


def options_of(case) -> dict:
    kernel, epsilon, degree, smoothing = case
    return {
        "kernel": kernel,
        "epsilon": epsilon,
        "degree": degree,
        "smoothing": smoothing,
    }


@pytest.mark.parametrize("case", CASES, ids=[f"{c[0]}-d{c[2]}-s{c[3]}" for c in CASES])
def test_matches_scipy(backend, points2d, values2d, grid2d, case):
    kwargs = options_of(case)
    reference = build_scipy(points2d, values2d, **kwargs)
    ours = build_ours(points2d, values2d, **kwargs)

    # At the constraint points...
    np.testing.assert_allclose(
        to_numpy(ours(points2d)), reference(points2d), rtol=RTOL, atol=ATOL
    )
    # ...and away from them, including outside the convex hull of the data.
    np.testing.assert_allclose(
        to_numpy(ours(grid2d)), reference(grid2d), rtol=RTOL, atol=ATOL
    )


def test_matches_scipy_in_one_dimension(backend, rng):
    y = np.sort(rng.random(40))[:, None]
    d = np.sin(6 * y[:, 0]) + 0.3 * y[:, 0] ** 2
    x = np.linspace(-0.5, 1.5, 200)[:, None]
    for kernel, epsilon in (
        ("thin_plate_spline", 1.0),
        ("cubic", 1.0),
        # 40 points in [0, 1] are spaced by ~0.025: a Gaussian needs an epsilon
        # of the order of 1/spacing to stay well conditioned.
        ("gaussian", 40.0),
    ):
        kwargs = {"kernel": kernel, "epsilon": epsilon, "degree": 2}
        reference = build_scipy(y, d, **kwargs)
        np.testing.assert_allclose(
            to_numpy(build_ours(y, d, **kwargs)(x)),
            reference(x),
            rtol=RTOL,
            atol=ATOL,
        )


def test_matches_scipy_in_three_dimensions(backend, rng):
    y = rng.random((50, 3))
    d = np.cos(4 * y[:, 0]) * np.exp(-y[:, 1]) + y[:, 2]
    x = rng.random((300, 3))
    kwargs = {"kernel": "thin_plate_spline", "epsilon": 1.0, "degree": 1}
    reference = build_scipy(y, d, **kwargs)
    np.testing.assert_allclose(
        to_numpy(build_ours(y, d, **kwargs)(x)), reference(x), rtol=RTOL, atol=ATOL
    )


def test_matches_scipy_for_vector_valued_data(backend, points2d, rng, grid2d):
    d = rng.random((60, 3))
    kwargs = {"kernel": "thin_plate_spline", "epsilon": 1.0, "degree": 1}
    reference = build_scipy(points2d, d, **kwargs)
    np.testing.assert_allclose(
        to_numpy(build_ours(points2d, d, **kwargs)(grid2d)),
        reference(grid2d),
        rtol=RTOL,
        atol=ATOL,
    )


def test_matches_scipy_for_data_with_trailing_dimensions(
    backend, points2d, rng, grid2d
):
    d = rng.random((60, 2, 3))
    kwargs = {"kernel": "linear", "epsilon": 1.0, "degree": 1}
    reference = build_scipy(points2d, d, **kwargs)
    got = to_numpy(build_ours(points2d, d, **kwargs)(grid2d))
    assert got.shape == reference(grid2d).shape == (200, 2, 3)
    np.testing.assert_allclose(got, reference(grid2d), rtol=RTOL, atol=ATOL)


def test_matches_scipy_with_per_point_smoothing(
    backend, points2d, values2d, grid2d, rng
):
    smoothing = rng.random(60) * 0.1
    kwargs = {
        "kernel": "thin_plate_spline",
        "epsilon": 1.0,
        "degree": 1,
        "smoothing": smoothing,
    }
    reference = build_scipy(points2d, values2d, **kwargs)
    np.testing.assert_allclose(
        to_numpy(build_ours(points2d, values2d, **kwargs)(grid2d)),
        reference(grid2d),
        rtol=RTOL,
        atol=ATOL,
    )


# ---------------------------------------------------------------------------
# Defaulting rules
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kernel", sorted(available_kernels()))
def test_the_default_polynomial_basis_matches_scipy(
    backend, points2d, values2d, kernel
):
    """
    SciPy exposes the monomials it picked (``powers``), so both the default
    degree and the enumeration of the basis can be compared directly.
    """
    reference = build_scipy(points2d, values2d, kernel=kernel, epsilon=1.0)
    ours = build_ours(points2d, values2d, kernel=kernel, epsilon=1.0)

    scipy_powers = {tuple(int(e) for e in row) for row in np.asarray(reference.powers)}
    assert scipy_powers == set(ours._powers)
    assert len(scipy_powers) == ours.n_poly
    assert max((sum(p) for p in scipy_powers), default=-1) == ours.degree


@pytest.mark.parametrize("kernel", sorted(available_kernels()))
def test_the_minimum_degree_table_matches_scipy(backend, points2d, values2d, kernel):
    """
    Below the kernel's minimum degree SciPy warns (and stays silent for -1).
    Comparing the two warning behaviours validates rbfgpu's minimum degree table.
    """
    min_degree = KERNELS[kernel].min_degree
    degree = min_degree - 1 if min_degree > 0 else -1

    with warnings.catch_warnings(record=True) as scipy_warnings:
        warnings.simplefilter("always")
        build_scipy(points2d, values2d, kernel=kernel, epsilon=1.0, degree=degree)
    with warnings.catch_warnings(record=True) as our_warnings:
        warnings.simplefilter("always")
        build_ours(points2d, values2d, kernel=kernel, epsilon=1.0, degree=degree)

    ours = [w for w in our_warnings if w.category is UserWarning]
    theirs = [w for w in scipy_warnings if w.category is UserWarning]
    assert len(ours) == len(theirs)
    assert len(ours) == (1 if min_degree > 0 else 0)
    if ours:
        assert "should not be below" in str(ours[0].message)


@pytest.mark.parametrize(
    "kernel", ["gaussian", "multiquadric", "inverse_multiquadric"]
)
def test_rbfgpu_defaults_epsilon_where_scipy_requires_it(
    backend, points2d, values2d, kernel
):
    """
    Documented difference: recent SciPy refuses to guess a shape parameter, while
    rbfgpu falls back to the nearest neighbour heuristic.
    """
    try:
        SciPyRBFInterpolator(points2d, values2d, kernel=kernel)
    except ValueError as error:
        assert "`epsilon` must be specified" in str(error)

    ours = build_ours(points2d, values2d, kernel=kernel)
    assert ours.epsilon == pytest.approx(default_epsilon(points2d), rel=1e-12)


def test_the_default_kernel_matches_scipy(backend, points2d, values2d, grid2d):
    reference = SciPyRBFInterpolator(points2d, values2d)
    ours = RBFInterpolator(points2d, values2d)
    assert ours.kernel == reference.kernel == "thin_plate_spline"
    np.testing.assert_allclose(
        to_numpy(ours(grid2d)), reference(grid2d), rtol=RTOL, atol=ATOL
    )
