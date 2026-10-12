"""
rbfGPU: GPU/CPU agnostic Radial Basis Function interpolation.

``rbfgpu`` interpolates scattered data with radial basis functions (thin plate
splines by default) using the `XuPy <https://pypi.org/project/xupy/>`_ array
namespace, so the very same code runs on CuPy arrays when an NVIDIA GPU is
available and on NumPy arrays otherwise.  It is the drop-in replacement for
``scipy.interpolate.RBFInterpolator`` in the regime where the O(n^2) Gram matrix
and the O(n^3) solve become the bottleneck, i.e. from a few thousand constraint
points upwards.

Examples
--------
>>> import numpy as np
>>> from rbfgpu import RBFInterpolator
>>> rng = np.random.default_rng(0)
>>> y = rng.random((1000, 2))
>>> d = np.sin(3 * y[:, 0]) * np.cos(3 * y[:, 1])
>>> rbf = RBFInterpolator(y, d)                  # thin plate spline, degree=1
>>> values = rbf(rng.random((10000, 2)))         # chunked, bounded memory
>>> values.shape
(10000,)

The backend can be inspected (and, through XuPy, switched) at any time:

>>> import rbfgpu
>>> rbfgpu.backend_info()["backend"] in ("numpy", "cupy")
True
"""

from . import __version__ as _version_module
from ._backend import as_numpy, backend_info, is_on_gpu, require_gpu
from ._distance import default_epsilon, euclidean_distance
from ._kernels import (
    KERNELS,
    KERNEL_ALIASES,
    Kernel,
    available_kernels,
    resolve_kernel,
)
from ._polynomial import monomial_exponents, polynomial_matrix
from .__version__ import __version__
from .interpolator import (
    DEFAULT_MAX_CHUNK_BYTES,
    DEFAULT_MIN_CHUNK_SIZE,
    RBFInterpolator,
)

__title__ = _version_module.__title__
__description__ = _version_module.__description__
__author__ = _version_module.__author__
__author_email__ = _version_module.__author_email__
__license__ = _version_module.__license__
__url__ = _version_module.__url__

__all__ = [
    "__version__",
    # main API
    "RBFInterpolator",
    "available_kernels",
    "KERNELS",
    "KERNEL_ALIASES",
    "Kernel",
    "resolve_kernel",
    # backend
    "backend_info",
    "is_on_gpu",
    "require_gpu",
    "as_numpy",
    # building blocks (handy for custom interpolants and for testing)
    "euclidean_distance",
    "default_epsilon",
    "monomial_exponents",
    "polynomial_matrix",
    "DEFAULT_MAX_CHUNK_BYTES",
    "DEFAULT_MIN_CHUNK_SIZE",
]
