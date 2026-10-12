"""
Helpers shared by the test modules.

Everything in here is re-implemented with plain NumPy, deliberately: the
reference values must not come from the code under test.
"""

import numpy as np
import pytest
import xupy as xp


def to_numpy(array) -> np.ndarray:
    """Any backend array (NumPy or CuPy) -> host NumPy array."""
    return xp.asnumpy(array)


def gpu_available() -> bool:
    """True when XuPy can actually open its CuPy backend on this machine."""
    try:
        with xp.backend("gpu"):
            pass
    except Exception:  # CuPy missing or unusable
        return False
    return True


GPU_OK = gpu_available()

#: Parametrization over the backends: "gpu" is skipped, never failed, when the
#: runner has no usable CuPy installation.
BACKENDS = ["cpu"] + (
    ["gpu"]
    if GPU_OK
    else [pytest.param("gpu", marks=pytest.mark.skip(reason="no usable GPU/CuPy"))]
)


def reference_distance(x, y, epsilon: float = 1.0) -> np.ndarray:
    """Brute force pairwise Euclidean distances (explicit broadcast difference)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    diff = x[:, None, :] - y[None, :, :]
    return epsilon * np.sqrt((diff**2).sum(axis=-1))


def reference_polynomial(x, powers) -> np.ndarray:
    """Monomial basis evaluation, with ``powers`` a sequence of exponent tuples."""
    x = np.asarray(x, dtype=float)
    if len(powers) == 0:
        return np.empty((x.shape[0], 0))
    columns = [np.prod(x ** np.asarray(p, dtype=int), axis=1) for p in powers]
    return np.stack(columns, axis=1)


def numpy_kernels() -> dict:
    """
    Independent NumPy implementation of every kernel, keyed by canonical name.

    Used to cross-check ``rbfgpu._kernels`` and to evaluate a ``to_cpu()`` state
    without importing anything else from rbfgpu.
    """
    return {
        "linear": lambda r: -r,
        "thin_plate_spline": lambda r: np.where(
            r > 0, r**2 * np.log(np.where(r > 0, r, 1.0)), 0.0
        ),
        "cubic": lambda r: r**3,
        "quintic": lambda r: -(r**5),
        "multiquadric": lambda r: -np.sqrt(1 + r**2),
        "inverse_multiquadric": lambda r: 1 / np.sqrt(1 + r**2),
        "inverse_quadratic": lambda r: 1 / (1 + r**2),
        "gaussian": lambda r: np.exp(-(r**2)),
        "matern1_2": lambda r: np.exp(-r),
        "matern3_2": lambda r: (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r),
        "matern5_2": lambda r: (
            1 + np.sqrt(5) * r + (5 / 3) * r**2
        ) * np.exp(-np.sqrt(5) * r),
    }


def reference_interpolant(state: dict, x) -> np.ndarray:
    """
    Evaluate an exported ``to_cpu()`` state with plain NumPy.

    Independent of rbfgpu: it re-computes the distances, applies the kernel and
    the monomials, and combines them with the stored coefficients.
    """
    x = np.asarray(x, dtype=float)
    r = reference_distance(x, state["y"], state["epsilon"])
    out = numpy_kernels()[state["kernel"]](r) @ state["coeffs_rbf"]
    if state["coeffs_poly"] is not None:
        out = out + reference_polynomial(x, state["powers"]) @ state["coeffs_poly"]
    return out
