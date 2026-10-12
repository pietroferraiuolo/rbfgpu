"""Tests for the public API surface and the package metadata."""

import os
import pathlib
import re
import subprocess
import sys

import pytest

import rbfgpu


def test_version_is_a_pep440_string():
    assert isinstance(rbfgpu.__version__, str)
    assert re.fullmatch(r"\d+\.\d+\.\d+([.\w+]*)?", rbfgpu.__version__)
    from rbfgpu.__version__ import __version__ as module_version

    assert rbfgpu.__version__ == module_version


def test_everything_in_dunder_all_exists():
    assert len(rbfgpu.__all__) > 10
    for name in rbfgpu.__all__:
        assert hasattr(rbfgpu, name), name


def test_the_main_api_is_importable_from_the_top_level():
    from rbfgpu import (
        KERNELS,
        RBFInterpolator,
        available_kernels,
        default_epsilon,
        euclidean_distance,
        monomial_exponents,
        polynomial_matrix,
        resolve_kernel,
    )

    assert callable(RBFInterpolator)
    assert set(KERNELS) == set(available_kernels())
    assert resolve_kernel("tps") is KERNELS["thin_plate_spline"]
    for helper in (
        available_kernels,
        default_epsilon,
        euclidean_distance,
        monomial_exponents,
        polynomial_matrix,
    ):
        assert callable(helper)


def test_metadata_is_exposed():
    assert rbfgpu.__title__ == "rbfgpu"
    assert rbfgpu.__author_email__
    assert rbfgpu.__license__ == "MIT License"
    assert rbfgpu.__url__.startswith("https://")


def test_py_typed_is_shipped():
    assert (pathlib.Path(rbfgpu.__file__).parent / "py.typed").exists()


def test_importing_rbfgpu_needs_neither_scipy_nor_cupy():
    """
    SciPy is a test-only dependency and CuPy an optional extra: rbfgpu must
    import, and run on the NumPy fallback, with both of them unavailable.
    """
    code = (
        "import sys;"
        "sys.modules['cupy'] = None;"  # makes `import cupy` raise ImportError
        "import rbfgpu;"
        "assert 'scipy' not in sys.modules, 'scipy was imported';"
        "assert not rbfgpu.is_on_gpu(), 'a GPU backend was found without cupy';"
        "print(rbfgpu.__version__)"
    )
    environment = {**os.environ, "XUPY_NO_GPU_WARNING": "1"}
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == rbfgpu.__version__


def test_installed_metadata_matches_the_package():
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - Python < 3.8
        pytest.skip("importlib.metadata is unavailable")
    try:
        installed = version("rbfgpu")
    except PackageNotFoundError:
        pytest.skip("rbfgpu is not installed (running from a source checkout)")
    assert installed == rbfgpu.__version__
