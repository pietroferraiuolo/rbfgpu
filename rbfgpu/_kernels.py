"""
Radial basis function kernels.

Every kernel maps the (already ``epsilon``-scaled) distance ``r`` onto a scalar
weight; an RBF interpolant is a linear combination of those weights centred on
the data points, plus a polynomial tail.

Names, signs and defaults follow ``scipy.interpolate.RBFInterpolator`` so that
an interpolant built here and one built by SciPy agree to round-off (see
``test/test_scipy_parity.py``).

References
----------
[1] Fasshauer, G., 2007. Meshfree Approximation Methods with Matlab.
[2] Wahba, G., 1990. Spline Models for Observational Data.
"""

from __future__ import annotations

import math
from typing import Any, Callable, NamedTuple, Optional

from ._backend import xp

__all__ = [
    "KERNELS",
    "KERNEL_ALIASES",
    "Kernel",
    "available_kernels",
    "resolve_kernel",
]


class Kernel(NamedTuple):
    """
    A radial basis function and the polynomial degree it needs.

    Attributes
    ----------
    name : str
        Canonical kernel name (also the key in :data:`KERNELS`).
    function : callable
        Maps an array of distances ``r`` to an array of weights.
    scale_invariant : bool
        True when ``epsilon`` only rescales the interpolant, so it can safely be
        left at 1 (``linear``, ``thin_plate_spline``, ``cubic``, ``quintic``).
    min_degree : int or None
        Minimum polynomial degree for a well-posed interpolant, or ``None`` for a
        user supplied callable (then both ``epsilon`` and ``degree`` must be
        given explicitly).  ``-1`` means "no constraint" (positive definite
        kernels).
    """

    name: str
    function: Callable[[Any], Any]
    scale_invariant: bool
    min_degree: Optional[int]

    @property
    def default_degree(self) -> Optional[int]:
        """
        Polynomial degree used when the caller does not pass one.

        The minimum degree required by the kernel, or 0 when the kernel needs no
        polynomial at all; ``None`` for custom callables (no default).
        """
        if self.min_degree is None:
            return None
        return max(self.min_degree, 0)


def _thin_plate_spline(r):
    """``r**2 * log(r)``, with the r -> 0 limit (0) applied elementwise."""
    # log(0) = -inf and 0 * -inf = nan: both are cleaned up to the limit value.
    with xp.errstate(divide="ignore", invalid="ignore"):
        result = r**2 * xp.log(r)
    return xp.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def _linear(r):
    """``-r``."""
    return -r


def _cubic(r):
    """``r**3``."""
    return r**3


def _quintic(r):
    """``-r**5``."""
    return -(r**5)


def _multiquadric(r):
    """``-sqrt(1 + r**2)``."""
    return -xp.sqrt(r**2 + 1)


def _inverse_multiquadric(r):
    """``1/sqrt(1 + r**2)``."""
    return 1 / xp.sqrt(r**2 + 1)


def _inverse_quadratic(r):
    """``1/(1 + r**2)``."""
    return 1 / (r**2 + 1)


def _gaussian(r):
    """``exp(-r**2)``."""
    return xp.exp(-(r**2))


def _matern1_2(r):
    """``exp(-r)``."""
    return xp.exp(-r)


def _matern3_2(r):
    """``(1 + sqrt(3) r) exp(-sqrt(3) r)``."""
    s = math.sqrt(3.0) * r
    return (1 + s) * xp.exp(-s)


def _matern5_2(r):
    """``(1 + sqrt(5) r + 5/3 r**2) exp(-sqrt(5) r)``."""
    s = math.sqrt(5.0) * r
    return (1 + s + s**2 / 3) * xp.exp(-s)


# name, function, scale invariant, minimum polynomial degree
_KERNEL_DEFS = (
    ("linear", _linear, True, 0),
    ("thin_plate_spline", _thin_plate_spline, True, 1),
    ("cubic", _cubic, True, 1),
    ("quintic", _quintic, True, 2),
    ("multiquadric", _multiquadric, False, 0),
    ("inverse_multiquadric", _inverse_multiquadric, False, -1),
    ("inverse_quadratic", _inverse_quadratic, False, -1),
    ("gaussian", _gaussian, False, -1),
    ("matern1_2", _matern1_2, False, -1),
    ("matern3_2", _matern3_2, False, -1),
    ("matern5_2", _matern5_2, False, -1),
)

KERNELS: dict = {
    name: Kernel(name, function, scale_invariant, min_degree)
    for name, function, scale_invariant, min_degree in _KERNEL_DEFS
}

#: Accepted spellings that are not canonical names.
KERNEL_ALIASES: dict = {
    "inverse_quartic": "inverse_quadratic",  # SciPy < 1.15 spelling
    "tps": "thin_plate_spline",
    "thin_plate": "thin_plate_spline",
    "imq": "inverse_multiquadric",
}


def available_kernels() -> tuple:
    """
    Canonical names of every built-in kernel.

    Returns
    -------
    names : tuple of str
        Sorted kernel names, as accepted by ``RBFInterpolator(kernel=...)``.
    """
    return tuple(sorted(KERNELS))


def resolve_kernel(kernel) -> Kernel:
    """
    Turn a kernel specification into a :class:`Kernel`.

    Parameters
    ----------
    kernel : str or callable
        A kernel name (aliases in :data:`KERNEL_ALIASES` are accepted, case
        insensitive) or a callable mapping an array of distances to an array of
        weights.

    Returns
    -------
    Kernel
        The resolved kernel.  A callable is wrapped in a ``Kernel`` whose
        ``min_degree`` is ``None``, i.e. one that requires ``epsilon`` and
        ``degree`` to be given explicitly.

    Raises
    ------
    TypeError
        If ``kernel`` is neither a string nor a callable.
    ValueError
        If the name is unknown.
    """
    if isinstance(kernel, str):
        name = kernel.strip().lower()
        name = KERNEL_ALIASES.get(name, name)
        try:
            return KERNELS[name]
        except KeyError:
            raise ValueError(
                f"Unknown kernel {kernel!r}. Available kernels: "
                f"{', '.join(available_kernels())}."
            ) from None

    if callable(kernel):
        return Kernel(
            name=getattr(kernel, "__name__", "<callable>"),
            function=kernel,
            scale_invariant=False,
            min_degree=None,
        )

    raise TypeError(
        "`kernel` must be a kernel name (str) or a callable taking the distance "
        f"array, got {type(kernel).__name__}."
    )
