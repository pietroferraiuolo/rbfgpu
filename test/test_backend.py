"""Tests for the XuPy backend adapter (``rbfgpu._backend``)."""

import numpy as np
import pytest
import xupy as xp

import rbfgpu
from rbfgpu import as_numpy, backend_info, is_on_gpu, require_gpu
from rbfgpu._backend import LINALG_ERRORS


def test_backend_info_is_consistent_with_xupy():
    info = backend_info()
    assert set(info) == {
        "on_gpu",
        "backend",
        "device",
        "numpy_version",
        "xupy_version",
        "cupy_version",
    }
    assert isinstance(info["on_gpu"], bool)
    assert info["on_gpu"] == is_on_gpu() == bool(xp.on_gpu)
    assert info["backend"] == ("cupy" if info["on_gpu"] else "numpy")


def test_backend_info_reports_the_library_versions():
    info = backend_info()
    assert info["numpy_version"] == np.__version__
    assert info["xupy_version"] == xp.__version__
    if info["on_gpu"]:  # pragma: no cover - needs a GPU
        assert info["cupy_version"] is not None
        assert info["device"].startswith("cuda:")
    else:
        assert info["device"] == "cpu"
        assert info["cupy_version"] is None


def test_linalg_errors_always_include_the_numpy_one():
    assert np.linalg.LinAlgError in LINALG_ERRORS
    assert all(issubclass(error, Exception) for error in LINALG_ERRORS)


def test_as_numpy_moves_backend_arrays_to_the_host(backend):
    array = xp.arange(5, dtype=xp.float64)
    out = as_numpy(array)
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, np.arange(5.0))


def test_as_numpy_keeps_host_arrays():
    array = np.arange(3.0)
    out = as_numpy(array)
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, array)


def test_is_on_gpu_follows_the_backend(backend):
    assert is_on_gpu() == (backend == "gpu")


def test_require_gpu_raises_on_the_cpu_backend(backend):
    if backend != "cpu":  # pragma: no cover - needs a GPU
        pytest.skip("only meaningful without a GPU backend")
    with pytest.raises(RuntimeError, match=r"NumPy \(CPU\) backend"):
        require_gpu()


def test_require_gpu_accepts_a_hint(backend):
    if backend != "cpu":  # pragma: no cover - needs a GPU
        pytest.skip("only meaningful without a GPU backend")
    with pytest.raises(RuntimeError, match="because reasons"):
        require_gpu("because reasons")


def test_require_gpu_is_silent_on_the_gpu_backend(backend):
    if backend != "gpu":
        pytest.skip("needs a usable GPU")
    require_gpu()  # pragma: no cover - must not raise


def test_the_public_backend_helpers_are_re_exported():
    for name in ("as_numpy", "backend_info", "is_on_gpu", "require_gpu"):
        assert getattr(rbfgpu, name) is not None
        assert name in rbfgpu.__all__
