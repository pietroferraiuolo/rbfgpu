"""
Backend adapter over the XuPy array namespace.

rbfGPU never talks to NumPy or CuPy directly: every array operation goes through
``xupy``, whose namespace is resolved dynamically against the active backend
(CuPy when a usable NVIDIA GPU is present, NumPy otherwise).  This module is the
only place where that duality is dealt with.

Note that, unlike the module this package started from, there is no
``if xp.on_gpu:`` branch: XuPy already provides the CPU fallback, so the very
same code path runs (and is tested) on both backends.
"""

from __future__ import annotations

import numpy as np
import xupy as xp

__all__ = [
    "LINALG_ERRORS",
    "as_numpy",
    "backend_info",
    "is_on_gpu",
    "require_gpu",
    "xp",
]


def _linalg_errors() -> tuple:
    """
    Exception types raised by ``xp.linalg.solve`` on a singular matrix.

    NumPy's and CuPy's ``LinAlgError`` are distinct classes, so both are
    collected: whichever backend is active, the solve step can catch the right
    one.
    """
    errors = [np.linalg.LinAlgError]
    try:  # pragma: no cover - depends on the machine running the tests
        import cupy

        errors.append(cupy.linalg.LinAlgError)
    except Exception:  # CuPy missing, or importable but unusable
        pass
    return tuple(errors)


LINALG_ERRORS = _linalg_errors()


def is_on_gpu() -> bool:
    """True when XuPy is serving arrays from CuPy (GPU), False for NumPy (CPU)."""
    return bool(xp.on_gpu)


def require_gpu(hint: str = "") -> None:
    """
    Raise ``RuntimeError`` unless XuPy is running on the GPU backend.

    Useful at the top of a script that is only worth running on a GPU, so that
    it fails loudly instead of silently computing on the CPU.

    Parameters
    ----------
    hint : str, optional
        Extra text appended to the error message.
    """
    if not is_on_gpu():
        message = (
            "rbfGPU is running on the NumPy (CPU) backend: no usable NVIDIA GPU "
            "or CuPy installation was found. Install a matching CuPy build with "
            "`pip install 'rbfgpu[cuda12]'` (or `[cuda13]`) and retry."
        )
        if hint:
            message = f"{message} {hint}"
        raise RuntimeError(message)


def as_numpy(array):
    """
    Move ``array`` to host memory and return it as a NumPy array.

    A no-op when XuPy is on the CPU backend (the array is returned unchanged if
    it already is an ``ndarray``).
    """
    return xp.asnumpy(array)


def backend_info() -> dict:
    """
    Describe the array backend rbfGPU is currently running on.

    Returns
    -------
    info : dict
        With keys ``on_gpu``, ``backend`` ("cupy" or "numpy"), ``device``,
        ``numpy_version``, ``xupy_version`` and ``cupy_version`` (None when
        CuPy is not importable).
    """
    on_gpu = is_on_gpu()
    info = {
        "on_gpu": on_gpu,
        "backend": "cupy" if on_gpu else "numpy",
        "device": "cpu",
        "numpy_version": np.__version__,
        "xupy_version": xp.__version__,
        "cupy_version": None,
    }
    if on_gpu:  # pragma: no cover - needs a GPU
        try:
            import cupy

            info["cupy_version"] = cupy.__version__
            device_id = cupy.cuda.runtime.getDevice()
            props = cupy.cuda.runtime.getDeviceProperties(device_id)
            name = props.get("name", b"")
            info["device"] = f"cuda:{device_id} " + (
                name.decode() if isinstance(name, bytes) else str(name)
            )
        except Exception as error:  # never let an informational call fail
            info["device"] = f"<unknown: {error}>"
    return info
