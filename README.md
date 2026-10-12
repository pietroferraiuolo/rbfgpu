# rbfGPU

[![PyPI version](https://img.shields.io/pypi/v/rbfgpu.svg)](https://pypi.org/project/rbfgpu/)
[![Python versions](https://img.shields.io/pypi/pyversions/rbfgpu.svg)](https://pypi.org/project/rbfgpu/)
[![License: MIT](https://img.shields.io/pypi/l/rbfgpu.svg)](LICENSE)
[![Tests](https://github.com/pietroferraiuolo/rbfgpu/actions/workflows/tests.yml/badge.svg)](https://github.com/pietroferraiuolo/rbfgpu/actions/workflows/tests.yml)

Radial Basis Function interpolation of scattered N-dimensional data, running on
the GPU through [XuPy](https://pypi.org/project/xupy/) (CuPy) and falling back to
NumPy on the CPU — **same code, same API, same results**.

`rbfgpu` exists for the regime where `scipy.interpolate.RBFInterpolator` becomes
the bottleneck: from a few thousand constraint points upwards, the O(n²) Gram
matrix and the O(n³) solve dominate, and a GPU turns minutes into seconds. Below
that threshold it is still a perfectly good CPU interpolator.

```python
import numpy as np
from rbfgpu import RBFInterpolator

y = np.random.default_rng(0).random((20000, 2))     # constraint points
d = np.sin(4 * y[:, 0]) * np.cos(4 * y[:, 1])       # data at those points

rbf = RBFInterpolator(y, d)                         # thin plate spline, degree=1
x = np.random.default_rng(1).random((500000, 2))    # evaluation grid
values = rbf(x)                                     # shape (500000,)
```

## Features

- **Thin plate splines and friends**: `linear`, `thin_plate_spline`, `cubic`,
  `quintic`, `multiquadric`, `inverse_multiquadric`, `inverse_quadratic`,
  `gaussian`, `matern1_2`, `matern3_2`, `matern5_2`, or your own callable.
- **SciPy compatible semantics**: kernel names, definitions and signs, and the
  `degree` defaulting rule follow `scipy.interpolate.RBFInterpolator`, and a
  differential test suite (`test/test_scipy_parity.py`) pins the agreement. Where
  rbfgpu is more permissive (a data-driven default `epsilon`, `degree=-1`, custom
  callables) the difference is documented and tested.
- **GPU or CPU, one code path**: every array operation goes through the XuPy
  namespace, which resolves to CuPy when a usable NVIDIA GPU is present and to
  NumPy otherwise. There is no `if on_gpu:` branch to maintain, and CI tests the
  same code on both backends.
- **Bounded memory**: evaluation is chunked automatically from a byte budget
  (`max_chunk_bytes`), or explicitly via `rbf(x, chunk_size=...)`.
- **Polynomial tail of any degree** (`-1` disables it), which is what makes the
  interpolant well posed for conditionally positive definite kernels and lets it
  extrapolate.
- **Smoothing** (Tikhonov), scalar or per point, for noisy data.
- **Vector valued data**: `d` may be `(n,)`, `(n, k)` or `(n, k, l, ...)`; every
  output is solved for in a single factorisation and the trailing shape is
  preserved.
- **Exportable**: `to_cpu()` returns the solved coefficients as NumPy arrays.
- Typed (`py.typed`), no dependencies beyond `numpy` and `xupy`.

## Installation

```bash
pip install rbfgpu                # CPU (NumPy)
pip install "rbfgpu[cuda12]"      # GPU, CuPy for CUDA 12.x
pip install "rbfgpu[cuda13]"      # GPU, CuPy for CUDA 13.x
```

Pick the extra matching the "CUDA Version" reported by `nvidia-smi`. Without a
CuPy extra the package still works, on the CPU: XuPy detects the situation and
resolves its namespace to NumPy.

```python
>>> import rbfgpu
>>> rbfgpu.backend_info()          # without a usable CuPy installation
{'on_gpu': False, 'backend': 'numpy', 'device': 'cpu', 'numpy_version': '2.4.6',
 'xupy_version': '2.0.0', 'cupy_version': None}
```

On a machine with a usable CuPy, `on_gpu` is `True`, `backend` is `'cupy'` and
`device` names the GPU. `rbfgpu.require_gpu()` raises a `RuntimeError` when the
GPU is expected but missing, so a script cannot silently fall back to the CPU.

## API

### `RBFInterpolator(y, d, ...)`

| argument | default | meaning |
| --- | --- | --- |
| `y` | — | constraint point coordinates, `(npoints, ndims)` (a 1-D array means `ndims=1`) |
| `d` | — | data values at `y`, `(npoints, ...)` |
| `kernel` | `'thin_plate_spline'` | kernel name (see `available_kernels()`) or a callable `r -> phi(r)` |
| `epsilon` | `None` | shape parameter: distances are scaled by it. `1` for the scale invariant kernels, otherwise the RMS nearest neighbour distance |
| `smoothing` | `0.0` | Tikhonov regularization, scalar or `(npoints,)` |
| `degree` | `None` | polynomial degree (`-1` = none). Defaults to the kernel's minimum, or 0 |
| `memory_efficient` | `True` | chunk the evaluation to stay under the memory budget |
| `max_chunk_bytes` | 1 GiB | memory budget for the temporaries of one chunk |
| `min_chunk_size` | 4096 | floor for the automatically estimated chunk size |

Calling the object evaluates the interpolant: `rbf(x)` returns an array of shape
`(npts,) + d.shape[1:]` on the same backend as the interpolant; `rbf(x,
chunk_size=1000)` overrides the automatic chunking.

Useful attributes: `kernel`, `epsilon`, `degree`, `n_poly`, `npoints`, `ndim`,
`output_dim`, `d_shape`, `smoothing`, `coeffs_rbf`, `coeffs_poly`, `on_gpu`.

### Helpers

```python
from rbfgpu import (
    available_kernels,   # ('cubic', 'gaussian', ..., 'thin_plate_spline')
    resolve_kernel,      # name/alias/callable -> Kernel(function, min_degree, ...)
    euclidean_distance,  # (n, d) x (m, d) -> (n, m), epsilon scaled
    default_epsilon,     # RMS nearest neighbour distance of a point cloud
    monomial_exponents,  # every exponent tuple with |p| <= degree
    polynomial_matrix,   # monomial basis evaluated at some points
    backend_info, is_on_gpu, require_gpu, as_numpy,
)
```

## Kernels

| name | φ(r) | scale invariant | default `degree` |
| --- | --- | --- | --- |
| `linear` | `-r` | yes | 0 |
| `thin_plate_spline` (aliases `tps`, `thin_plate`) | `r² log r` | yes | 1 |
| `cubic` | `r³` | yes | 1 |
| `quintic` | `-r⁵` | yes | 2 |
| `multiquadric` | `-sqrt(1 + r²)` | no | 0 |
| `inverse_multiquadric` (alias `imq`) | `1/sqrt(1 + r²)` | no | 0 |
| `inverse_quadratic` (alias `inverse_quartic`) | `1/(1 + r²)` | no | 0 |
| `gaussian` | `exp(-r²)` | no | 0 |
| `matern1_2` | `exp(-r)` | no | 0 |
| `matern3_2` | `(1 + √3 r) exp(-√3 r)` | no | 0 |
| `matern5_2` | `(1 + √5 r + 5/3 r²) exp(-√5 r)` | no | 0 |

Any callable `phi(r) -> array` works too; it then requires `epsilon` and `degree`
to be passed explicitly:

```python
import xupy as xp
from rbfgpu import RBFInterpolator

rbf = RBFInterpolator(y, d, kernel=lambda r: xp.exp(-r), epsilon=1.0, degree=-1)
```

### How it works

The interpolant is a linear combination of kernels centred on the data points,
plus a polynomial tail:

```
f(x) = Σᵢ aᵢ φ(ε‖x − yᵢ‖) + Σⱼ bⱼ pⱼ(x)
```

The coefficients solve the symmetric block system

```
[K + λI   P] [a] = [d]
[Pᵀ       0] [b]   [0]
```

with `K` the kernel Gram matrix, `P` the monomial matrix, `λ` the smoothing and
`Pᵀa = 0` the side conditions that make the problem well posed for conditionally
positive definite kernels (and allow extrapolation). The Gram matrix is written
straight into the block matrix, so the `(n, n)` kernel matrix and the
`(n + n_poly)²` system are never resident in device memory at the same time.

### Choosing `epsilon` and `degree`

- The scale invariant kernels (`linear`, `thin_plate_spline`, `cubic`, `quintic`)
  ignore `epsilon`: it only rescales the data, which is equivalent to rescaling
  `smoothing`.
- The other kernels need a sensible `epsilon`. rbfgpu defaults to
  `sqrt(mean(nearest neighbour distance))` — a reasonable starting point but
  **not** a guarantee: too small an `epsilon` gives very wide kernels and an
  ill-conditioned system (the data is then interpolated poorly), too large gives
  a nearly diagonal one. Tune it, ideally by cross validation. (SciPy refuses to
  guess and raises instead.)
- `degree` below the kernel's minimum is allowed but warns: the interpolant may
  then be ill posed. `degree=-1` removes the polynomial tail entirely.
- `smoothing > 0` trades exactness for stability, and is the first thing to try
  when the build fails with `RuntimeError: ... singular matrix ...`.

## Memory and chunking

Evaluation allocates `O(n_eval × npoints)` floats. By default the evaluation grid
is split into chunks sized from a byte budget, which keeps the peak device memory
bounded without any intervention:

```python
rbf = RBFInterpolator(y, d, max_chunk_bytes=256 * 1024**2)   # 256 MiB budget
values = rbf(huge_grid)                                      # chunked automatically
values = rbf(huge_grid, chunk_size=4096)                     # or fixed explicitly
```

`memory_efficient=False` evaluates in one batch (fastest, most memory hungry), and
`min_chunk_size` keeps very small budgets from degenerating into one point at a
time.

## Backends

The backend follows XuPy, so it can be inspected and switched without touching
rbfgpu:

```python
import xupy as xp
from rbfgpu import RBFInterpolator

rbf_gpu = RBFInterpolator(y, d)          # CuPy arrays, if a GPU is available
rbf_gpu.on_gpu                           # True

with xp.backend("cpu"):                  # scoped, thread local
    rbf_cpu = RBFInterpolator(y, d)      # NumPy arrays, whatever the default is
```

Arrays never move implicitly: an interpolant keeps the backend it was built on.
`rbfgpu.as_numpy(array)` brings any backend array to the host, and
`RBFInterpolator.to_cpu()` exports the whole solved interpolant as NumPy arrays
(coefficients, centres, monomial exponents, kernel name, `epsilon`), so that it
can be stored or evaluated elsewhere.

## Relationship with SciPy

`rbfgpu.RBFInterpolator` is deliberately close to `scipy.interpolate.RBFInterpolator`:

| | rbfgpu | SciPy |
| --- | --- | --- |
| backends | CuPy (GPU) or NumPy (CPU) | NumPy, or an Array API backend when `SCIPY_ARRAY_API=1` |
| kernels | the same 11 names, plus callables | 8 names (Matérn kernels are newer), plus `LowLevelCallable` |
| `degree` default | `max(min_degree, 0)`, same rule | same |
| `epsilon` default | 1 for the scale invariant kernels, otherwise `default_epsilon(y)` | 1 for the scale invariant kernels, otherwise **required** (raises) |
| `smoothing`, vector valued `d`, trailing dimensions | yes | yes |
| `neighbors` (local interpolation with a KD-tree) | **no** | yes |
| chunked evaluation with an explicit byte budget | yes | internal memory budget |

Not implemented, by design: `neighbors`. Local interpolation needs a KD-tree
query per evaluation point, which is a different (and host side) algorithm. If you
need it, use SciPy.

### Working with the results

On the GPU backend the interpolant returns **CuPy** arrays, and CuPy refuses to
mix with NumPy operands. Bring results to the host before combining them with
NumPy data:

```python
values = rbf(x)                       # cupy.ndarray on a GPU, numpy.ndarray on a CPU
residual = rbfgpu.as_numpy(values) - d_numpy    # ok
# residual = values - d_numpy                   # TypeError on the GPU backend
```

## Benchmark

`benchmarks/bench_vs_scipy.py` times the build and the evaluation of both
implementations and reports the largest difference between the two interpolants
(they solve the same system, so it must stay at round-off level):

```bash
python benchmarks/bench_vs_scipy.py --sizes 1000 2000 4000 8000 --eval 200000
```

The speed-up depends entirely on the GPU, on the number of constraint points and
on the number of evaluation points: measure it on your own hardware rather than
trusting a table in a README.

## Development

```bash
git clone https://github.com/pietroferraiuolo/rbfgpu.git
cd rbfgpu
pip install -e ".[test]"          # add ,cuda12 to test the GPU backend as well
python -m pytest                  # the whole suite, on every available backend
python -m pytest -m "not slow"    # skip the large-problem tests
python benchmarks/bench_vs_scipy.py --sizes 500 1000
```

The `backend` fixture in `test/conftest.py` parametrizes every test over `"cpu"`
and `"gpu"`; the GPU case is skipped, never failed, when CuPy is unusable.
`test/test_scipy_parity.py` compares against SciPy and is skipped when SciPy is
not installed. Reference values in `test/_util.py` are re-implemented with plain
NumPy on purpose: they must not come from the code under test.

## Releasing

1. Bump `rbfgpu/__version__.py` and add an entry to `CHANGELOG.md`.
2. Commit, tag `vX.Y.Z` and push the tag.
3. Create the GitHub Release: `.github/workflows/publish.yml` builds the sdist and
   the wheel, runs the test suite against the built wheel, and uploads to PyPI.

The workflow authenticates with the `PYPI_API_TOKEN` repository secret (an API
token scoped to the `rbfgpu` project). To use PyPI *trusted publishing* instead,
register the repository on the project's publishing page and follow the comment
at the bottom of `publish.yml`.

## Provenance

Extracted from the `opticalib` adaptive optics framework
(`opticalib/simulator/_API/_rbf_gpu.py`), where it accelerates the interpolation
of deformable mirror influence functions, and generalised into a standalone
package.

## License

MIT — see [LICENSE](LICENSE).
