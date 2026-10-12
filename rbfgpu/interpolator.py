"""
Radial Basis Function interpolation on the GPU (with a transparent CPU fallback).

This is a GPU-optimised alternative to ``scipy.interpolate.RBFInterpolator`` for
the regime where SciPy runs out of breath: a few thousand constraint points, for
which the O(n^2) Gram matrix and the O(n^3) solve dominate the runtime and fit
comfortably in device memory.

Theory
------
An RBF interpolant is a linear combination of radial basis functions centred at
the data points ``y`` plus a polynomial:

    f(x) = sum_i a_i phi(||x - y_i||) + sum_j b_j p_j(x)

where ``phi`` is the RBF kernel (e.g. the thin plate spline ``r^2 log r``) and
``p_j`` are monomials of total degree <= ``degree``.  The coefficients ``(a, b)``
solve the block system

    [K + lambda I   P] [a] = [d]
    [P^T            0] [b]   [0]

with ``K`` the RBF Gram matrix and ``lambda`` the smoothing parameter.  The
polynomial side conditions ``P^T a = 0`` are what makes the system uniquely
solvable for conditionally positive definite kernels, and what allows the
interpolant to extrapolate.

Everything runs through XuPy, so the same object is built on CuPy arrays when a
GPU is available and on NumPy arrays otherwise.

References
----------
[1] Fasshauer, G., 2007. Meshfree Approximation Methods with Matlab.
[2] Wahba, G., 1990. Spline Models for Observational Data.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from xupy import typings as _ot

from ._backend import LINALG_ERRORS, as_numpy, xp
from ._distance import default_epsilon, euclidean_distance
from ._kernels import Kernel, resolve_kernel
from ._polynomial import monomial_exponents, polynomial_matrix

__all__ = ["DEFAULT_MAX_CHUNK_BYTES", "DEFAULT_MIN_CHUNK_SIZE", "RBFInterpolator"]

#: Memory budget used to pick the evaluation chunk size (1 GiB).
DEFAULT_MAX_CHUNK_BYTES = 1024 * 1024**2
#: Floor for the estimated chunk size, so that tiny budgets still batch usefully.
DEFAULT_MIN_CHUNK_SIZE = 4096

#: Message for a system that cannot be solved, whatever the backend reported.
_SINGULAR_SYSTEM_HINT = (
    "Failed to solve the RBF system: the matrix is singular. The constraint "
    "points may be duplicated or collinear, or the kernel may be too wide or too "
    "narrow for the data. Try removing duplicated points, increasing `smoothing`, "
    "or tuning `epsilon`."
)


class RBFInterpolator:
    """
    Radial Basis Function interpolator, GPU accelerated through XuPy.

    The API mirrors ``scipy.interpolate.RBFInterpolator`` for the parts that are
    implemented here (there is no ``neighbors`` support: every evaluation point
    always uses all the constraint points).  Kernels, ``epsilon`` and ``degree``
    defaults follow SciPy, so the two agree to round-off.

    Parameters
    ----------
    y : array, shape (npoints, ndims)
        Constraint point coordinates.  A 1-D array is accepted and read as
        ``npoints`` points in one dimension.  Host arrays are moved onto the
        active backend automatically.
    d : array, shape (npoints, ...)
        Data values at ``y``.  The first axis must match ``y``; the trailing
        shape is preserved in the output.  Several outputs are solved for at
        once, in a single factorisation.
    kernel : str or callable, optional
        RBF kernel: one of ``rbfgpu.available_kernels()``, or a callable mapping
        an array of distances to an array of weights (then ``epsilon`` and
        ``degree`` must be given explicitly).  Default: 'thin_plate_spline'.
    epsilon : float, optional
        Shape parameter: distances are multiplied by it before the kernel is
        applied.  Defaults to 1 for the scale invariant kernels ('linear',
        'thin_plate_spline', 'cubic', 'quintic') and, for the others, to
        :func:`rbfgpu.default_epsilon` (the square root of the mean nearest
        neighbour distance of ``y``).  SciPy refuses to guess and requires
        ``epsilon`` for those kernels; rbfgpu estimates it instead, so pass your
        own value whenever the interpolation quality matters.
    smoothing : float or array, shape (npoints,), optional
        Tikhonov regularization.  0 (default) interpolates the data exactly;
        large values approach a least squares polynomial fit of the given
        ``degree``.
    degree : int, optional
        Degree of the added polynomial.  ``-1`` adds none.  Defaults to the
        minimum degree the kernel needs ('thin_plate_spline' and 'cubic': 1,
        'quintic': 2, 'linear' and 'multiquadric': 0), or 0 for positive
        definite kernels.  A degree strictly between -1 and that minimum warns,
        as in SciPy: the interpolant may not be uniquely solvable.
    memory_efficient : bool, optional
        If True (default), evaluation is chunked to keep the peak device memory
        below ``max_chunk_bytes``.
    max_chunk_bytes : int, optional
        Memory budget, in bytes, for the temporaries allocated while evaluating
        one chunk.  Default: 1 GiB.
    min_chunk_size : int, optional
        Smallest chunk the automatic estimate may return.  Default: 4096.

    Examples
    --------
    >>> import numpy as np
    >>> import rbfgpu
    >>> from rbfgpu import RBFInterpolator
    >>> rng = np.random.default_rng(0)
    >>> y = rng.random((1000, 2))            # 1000 constraint points in 2-D
    >>> d = np.sin(y[:, 0]) * np.cos(y[:, 1])
    >>> rbf = RBFInterpolator(y, d)          # thin plate spline, degree=1
    >>> x = rng.random((10000, 2))           # 10000 evaluation points
    >>> values = rbf(x)
    >>> values.shape
    (10000,)

    The result lives on the interpolant's backend (CuPy on a GPU), so bring it to
    the host before mixing it with NumPy arrays:

    >>> residual = rbfgpu.as_numpy(rbf(y)) - d
    >>> bool(np.max(np.abs(residual)) < 1e-9)
    True

    Notes
    -----
    Building the interpolant costs one O(npoints^2) distance/kernel evaluation
    and one O(npoints^3) solve; evaluating it costs O(n_eval * npoints).  With
    ``memory_efficient=True`` the evaluation is split into chunks, trading a
    little speed for a bounded peak memory.

    The ``epsilon`` default and the ``smoothing`` validation reduce device values
    to Python floats, i.e. they synchronise the GPU once at build time.  The
    solve synchronises anyway, so no time is lost.
    """

    def __init__(
        self,
        y: _ot.ArrayLike,
        d: _ot.ArrayLike,
        kernel: _ot.Union[str, _ot.Callable] = "thin_plate_spline",
        epsilon: _ot.Optional[float] = None,
        smoothing: _ot.Union[float, _ot.ArrayLike] = 0.0,
        degree: _ot.Optional[int] = None,
        memory_efficient: bool = True,
        max_chunk_bytes: int = DEFAULT_MAX_CHUNK_BYTES,
        min_chunk_size: int = DEFAULT_MIN_CHUNK_SIZE,
    ):
        self.y, npoints, ndim = self._as_points(y)
        self.d, self.d_shape = self._as_values(d, npoints)

        self._kernel: Kernel = resolve_kernel(kernel)
        self.kernel = self._kernel.name
        self.epsilon = self._resolve_epsilon(epsilon)
        self.degree = self._resolve_degree(degree)
        self._powers = monomial_exponents(ndim, self.degree)
        self.n_poly = len(self._powers)

        self.smoothing = self._as_smoothing(smoothing, npoints)

        self.memory_efficient = bool(memory_efficient)
        self.max_chunk_bytes = int(max_chunk_bytes)
        self.min_chunk_size = int(min_chunk_size)
        if self.max_chunk_bytes <= 0:
            raise ValueError(
                f"`max_chunk_bytes` must be positive, got {max_chunk_bytes}."
            )
        if self.min_chunk_size <= 0:
            raise ValueError(
                f"`min_chunk_size` must be positive, got {min_chunk_size}."
            )

        self.npoints = npoints
        self.ndim = ndim
        self.output_dim = self.d.shape[1]

        self._build_system()

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    @staticmethod
    def _as_points(y: _ot.ArrayLike):
        """Convert ``y`` to a float64 (npoints, ndims) array on the backend."""
        y = xp.asarray(y, dtype=xp.float64)
        if y.ndim == 1:
            y = xp.reshape(y, (y.shape[0], 1))
        if y.ndim != 2:
            raise ValueError(
                "`y` must be a 1-D or 2-D array of point coordinates, got an "
                f"array with {y.ndim} dimensions."
            )
        npoints, ndim = y.shape
        if npoints == 0:
            raise ValueError("`y` must contain at least one point.")
        return y, int(npoints), int(ndim)

    @staticmethod
    def _as_values(d: _ot.ArrayLike, npoints: int):
        """
        Convert ``d`` to a float64 (npoints, output_dim) array on the backend.

        Returns the flattened values and the trailing shape of the input, so
        that ``__call__`` can restore it.
        """
        d = xp.asarray(d, dtype=xp.float64)
        if d.ndim == 0:
            d = xp.reshape(d, (1,))
        if d.shape[0] != npoints:
            raise ValueError(
                "`d` must have the same length along its first axis as `y` "
                f"({npoints}), got {d.shape[0]}."
            )
        d_shape = tuple(int(size) for size in d.shape[1:])
        return xp.reshape(d, (npoints, -1)), d_shape

    def _resolve_epsilon(self, epsilon: _ot.Optional[float]) -> float:
        """Pick the shape parameter, following SciPy's defaulting rules."""
        if self._kernel.min_degree is None and epsilon is None:
            raise ValueError(
                "`epsilon` must be given explicitly when `kernel` is a callable."
            )
        if epsilon is None:
            # Scale invariant kernels only use epsilon to rescale the data, so
            # any value works; the others need a data-driven default.
            epsilon = 1.0 if self._kernel.scale_invariant else default_epsilon(self.y)
        epsilon = float(epsilon)
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError(f"`epsilon` must be a positive finite number, got {epsilon}.")
        return epsilon

    def _resolve_degree(self, degree: _ot.Optional[int]) -> int:
        """Pick the polynomial degree, following SciPy's defaulting rules."""
        if self._kernel.min_degree is None and degree is None:
            raise ValueError(
                "`degree` must be given explicitly when `kernel` is a callable."
            )
        if degree is None:
            return int(self._kernel.default_degree)

        degree = int(degree)
        if degree < -1:
            raise ValueError(f"`degree` must be >= -1 (-1 means no polynomial), got {degree}.")
        min_degree = self._kernel.min_degree
        if min_degree is not None and -1 < degree < min_degree:
            # Same rule as SciPy: -1 ("no polynomial at all") is an explicit,
            # legitimate choice and must not warn; a degree in between is not.
            warnings.warn(
                f"`degree` should not be below {min_degree} except -1 when "
                f"`kernel` is '{self._kernel.name}'. The interpolant may not be "
                "uniquely solvable, and the smoothing parameter may have an "
                "unintuitive effect.",
                UserWarning,
                stacklevel=3,
            )
        return degree

    @staticmethod
    def _as_smoothing(smoothing: _ot.Union[float, _ot.ArrayLike], npoints: int):
        """Broadcast a scalar ``smoothing`` or validate an array one."""
        if np.ndim(smoothing) == 0:
            smoothing = xp.full(npoints, float(smoothing), dtype=xp.float64)
        else:
            smoothing = xp.asarray(smoothing, dtype=xp.float64)
            if smoothing.shape != (npoints,):
                raise ValueError(
                    "`smoothing` must be a scalar or an array of shape "
                    f"({npoints},), got shape {tuple(smoothing.shape)}."
                )
        if bool(xp.any(smoothing < 0)):
            raise ValueError("`smoothing` must be non-negative.")
        return smoothing

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def _build_system(self) -> None:
        """
        Build and solve the RBF linear system.

        Constructs the block matrix

            [K + lambda I   P] [a] = [d]
            [P^T            0] [b]   [0]

        with ``K`` the RBF Gram matrix, ``P`` the polynomial matrix and
        ``lambda`` the smoothing, and stores the solution as ``coeffs_rbf``
        (``a``) and ``coeffs_poly`` (``b``, or None when ``degree == -1``).
        """
        n = self.npoints
        n_poly = self.n_poly
        n_sys = n + n_poly

        A = xp.zeros((n_sys, n_sys), dtype=xp.float64)

        # The kernel block is written straight into the system matrix, so that
        # the (n, n) Gram matrix and the (n_sys, n_sys) block matrix are never
        # resident on the device at the same time.
        dist = euclidean_distance(self.y, self.y, self.epsilon)
        A[:n, :n] = self._kernel.function(dist)
        del dist

        A[:n, :n] += xp.diag(self.smoothing)

        if n_poly > 0:
            P = polynomial_matrix(self.y, self._powers)
            A[:n, n:] = P
            A[n:, :n] = P.T
            del P

        b = xp.zeros((n_sys, self.output_dim), dtype=xp.float64)
        b[:n] = self.d

        try:
            coeffs = xp.linalg.solve(A, b)
        except LINALG_ERRORS as error:
            raise RuntimeError(f"{_SINGULAR_SYSTEM_HINT} ({error})") from error
        del A, b

        # LAPACK reports a singular factorisation by raising LinAlgError; cuSOLVER
        # quietly returns NaNs instead.  The solution is therefore checked
        # explicitly: handing back NaN coefficients would corrupt every later
        # evaluation without any visible cause.  Costs one host synchronisation,
        # which the solve has already paid for.
        if not bool(xp.isfinite(coeffs).all()):
            raise RuntimeError(
                f"{_SINGULAR_SYSTEM_HINT} (the solver returned non-finite "
                "coefficients)"
            )

        self.coeffs_rbf = coeffs[:n]
        self.coeffs_poly = coeffs[n:] if n_poly > 0 else None

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def _resolve_chunk_size(self, n_eval: int, chunk_size: _ot.Optional[int]) -> int:
        """
        Resolve the evaluation chunk size.

        Parameters
        ----------
        n_eval : int
            Number of evaluation points.
        chunk_size : int or None
            Explicit chunk size.  If None, a size is estimated from the
            configured memory budget (or the whole evaluation is done at once
            when ``memory_efficient`` is False).

        Returns
        -------
        int
            Effective chunk size, always in [1, n_eval].
        """
        if chunk_size is not None:
            return max(1, min(n_eval, int(chunk_size)))

        if not self.memory_efficient:
            return n_eval

        bytes_per_float = np.dtype(np.float64).itemsize
        # Per evaluation point: the (npoints,) distance and kernel arrays and
        # the matmul result (4 arrays of npoints floats, conservatively), plus
        # the polynomial matrix and the output.
        floats_per_eval = 4 * self.npoints + self.output_dim + self.n_poly + self.ndim
        bytes_per_eval = bytes_per_float * floats_per_eval
        estimated_chunk = self.max_chunk_bytes // max(bytes_per_eval, 1)
        estimated_chunk = max(self.min_chunk_size, int(estimated_chunk))
        return max(1, min(n_eval, estimated_chunk))

    def __call__(self, x: _ot.ArrayLike, chunk_size: _ot.Optional[int] = None):
        """
        Evaluate the interpolant at the points ``x``.

        Parameters
        ----------
        x : array, shape (npts, ndims)
            Evaluation points.  For a 1-D interpolation problem (``ndims == 1``)
            a flat array of ``npts`` coordinates is accepted; for ``ndims > 1``
            a flat array is read as a single point.
        chunk_size : int, optional
            Number of evaluation points per batch.  Overrides the automatic
            estimate; useful to bound the device memory on very large grids.

        Returns
        -------
        f : array
            Interpolant values at ``x``, with shape ``(npts,) + d.shape[1:]``,
            on the same backend as the interpolant.
        """
        x = self._as_evaluation_points(x)
        n_eval = x.shape[0]

        resolved_chunk_size = self._resolve_chunk_size(n_eval, chunk_size)
        if resolved_chunk_size >= n_eval:
            out = self._evaluate_batch(x)
        else:
            out = xp.empty((n_eval, self.output_dim), dtype=xp.float64)
            for start in range(0, n_eval, resolved_chunk_size):
                stop = min(start + resolved_chunk_size, n_eval)
                out[start:stop] = self._evaluate_batch(x[start:stop])

        if self.d_shape:
            return xp.reshape(out, (n_eval,) + self.d_shape)
        return out[:, 0]

    def _as_evaluation_points(self, x: _ot.ArrayLike):
        """Convert ``x`` to a float64 (npts, ndims) array and check its shape."""
        x = xp.asarray(x, dtype=xp.float64)
        if x.ndim == 1:
            if self.ndim == 1:
                x = xp.reshape(x, (-1, 1))
            else:
                x = xp.reshape(x, (1, -1))
        if x.ndim != 2:
            raise ValueError(
                "`x` must be a 2-D array of shape (npts, ndims), got an array "
                f"with {x.ndim} dimensions."
            )
        if x.shape[1] != self.ndim:
            raise ValueError(
                f"Expected the second axis of `x` to have length {self.ndim}, "
                f"got {x.shape[1]}."
            )
        return x

    def _evaluate_batch(self, x):
        """Evaluate the interpolant on one batch of points."""
        # RBF contribution: sum_i a_i phi(||x - y_i||)
        dist = euclidean_distance(x, self.y, self.epsilon)
        K = self._kernel.function(dist)
        del dist
        out = K @ self.coeffs_rbf
        del K

        # Polynomial contribution: sum_j b_j p_j(x)
        if self.coeffs_poly is not None:
            P = polynomial_matrix(x, self._powers)
            out = out + P @ self.coeffs_poly
            del P

        return out

    # ------------------------------------------------------------------
    # Inspection / export
    # ------------------------------------------------------------------
    @property
    def on_gpu(self) -> bool:
        """True when this interpolant's arrays live in GPU memory."""
        return not isinstance(self.y, np.ndarray)

    def to_cpu(self) -> dict:
        """
        Export the solved interpolant as host (NumPy) arrays.

        The returned state is everything needed to evaluate the interpolant
        without this package, for instance to hand it over to a CPU-only process
        or to store it on disk.

        Returns
        -------
        state : dict
            With keys ``y``, ``coeffs_rbf``, ``coeffs_poly`` (None when
            ``degree == -1``), ``powers``, ``degree``, ``kernel`` and
            ``epsilon``.
        """
        return {
            "y": as_numpy(self.y),
            "coeffs_rbf": as_numpy(self.coeffs_rbf),
            "coeffs_poly": (
                None if self.coeffs_poly is None else as_numpy(self.coeffs_poly)
            ),
            "powers": self._powers,
            "degree": self.degree,
            "kernel": self.kernel,
            "epsilon": self.epsilon,
        }

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(kernel={self.kernel!r}, npoints={self.npoints}, "
            f"ndim={self.ndim}, output_dim={self.output_dim}, degree={self.degree}, "
            f"epsilon={self.epsilon!r}, "
            f"backend={'gpu' if self.on_gpu else 'cpu'!r})"
        )
