# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-12

First release. Extracted from `opticalib.simulator._API._rbf_gpu` and turned into
a standalone package.

### Added

- `RBFInterpolator`: radial basis function interpolation of scattered N-D data,
  running on CuPy through [XuPy](https://pypi.org/project/xupy/) when a GPU is
  available and on NumPy otherwise. One code path, one test suite, both backends.
- Kernels: `linear`, `thin_plate_spline`, `cubic`, `quintic`, `multiquadric`,
  `inverse_multiquadric`, `inverse_quadratic`, `gaussian`, `matern1_2`,
  `matern3_2`, `matern5_2`, plus user supplied callables. Names, signs and the
  `epsilon` / `degree` defaulting rules follow `scipy.interpolate.RBFInterpolator`.
- Polynomial tail of arbitrary `degree` (`-1` disables it), monomial basis.
- Scalar or per-point `smoothing` (Tikhonov regularization).
- Multiple outputs solved in one factorisation; `d` may have trailing dimensions,
  which are preserved in the result.
- Chunked evaluation (`memory_efficient`, `max_chunk_bytes`, `min_chunk_size`,
  `chunk_size=`) to bound the peak device memory on large evaluation grids.
- `to_cpu()` exporting the solved coefficients as NumPy arrays.
- Backend helpers: `backend_info()`, `is_on_gpu()`, `require_gpu()`, `as_numpy()`.
- Building blocks exposed for reuse: `euclidean_distance()`, `default_epsilon()`,
  `monomial_exponents()`, `polynomial_matrix()`.
- Test suite (CPU always, GPU when CuPy is usable) including differential tests
  against SciPy, a benchmark script, and GitHub Actions workflows for tests and
  for publishing to PyPI on release.

### Changed relative to the original opticalib module

- No `if xp.on_gpu:` branch any more: XuPy already provides the NumPy fallback,
  so the CPU path is the same code (and is now covered by CI on both backends).
- Kernels other than the thin plate spline are implemented, with SciPy's names,
  signs and minimum-degree table; a callable can be passed as `kernel`.
- `epsilon` is supported (SciPy's shape parameter), with a data-driven default
  for the kernels that need one.
- Unknown kernels raise `ValueError` listing the available ones instead of
  `NotImplementedError`.
- A `degree` strictly between -1 and the kernel's minimum warns (`UserWarning`,
  exactly like SciPy) instead of being refused, so `degree=-1` stays usable.
- A singular system raises `RuntimeError` on both backends: cuSOLVER returns NaNs
  where LAPACK raises, so the solution is checked for non-finite coefficients
  before being accepted.

[Unreleased]: https://github.com/pietroferraiuolo/rbfgpu/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/pietroferraiuolo/rbfgpu/releases/tag/v0.1.0
