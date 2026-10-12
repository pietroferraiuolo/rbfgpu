#!/usr/bin/env python3
"""
Benchmark rbfgpu against ``scipy.interpolate.RBFInterpolator``.

Usage
-----
    python benchmarks/bench_vs_scipy.py
    python benchmarks/bench_vs_scipy.py --sizes 500 1000 2000 4000 --eval 100000
    python benchmarks/bench_vs_scipy.py --kernel gaussian --epsilon 2 --smoothing 1e-3

For every problem size the script reports the time to build the interpolant and
the time to evaluate it, for both implementations, the resulting speed-up, and
the largest difference between the two interpolants (they solve the same linear
system, so they must agree to round-off).

SciPy is optional: without it only rbfgpu is timed.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import rbfgpu
from rbfgpu import RBFInterpolator

try:
    from scipy.interpolate import RBFInterpolator as SciPyRBFInterpolator
except ImportError:  # SciPy is not a dependency of rbfgpu
    SciPyRBFInterpolator = None


def timeit(function, repeats: int) -> tuple:
    """Best of `repeats` wall clock timings: (seconds, last result)."""
    best = float("inf")
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        best = min(best, time.perf_counter() - start)
    return best, result


def to_host(array) -> np.ndarray:
    """NumPy or CuPy array -> host array (needed to compare and to time fairly)."""
    return rbfgpu.as_numpy(array)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[500, 1000, 2000, 4000],
        help="number of constraint points to benchmark (default: 500 1000 2000 4000)",
    )
    parser.add_argument(
        "--eval",
        type=int,
        default=50000,
        help="number of evaluation points (default: 50000)",
    )
    parser.add_argument("--kernel", default="thin_plate_spline")
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--smoothing", type=float, default=0.0)
    parser.add_argument("--degree", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    options = {
        "kernel": args.kernel,
        "smoothing": args.smoothing,
    }
    if args.epsilon is not None:
        options["epsilon"] = args.epsilon
    if args.degree is not None:
        options["degree"] = args.degree

    info = rbfgpu.backend_info()
    print(f"rbfgpu {rbfgpu.__version__} on {info['backend']} ({info['device']})")
    print(f"numpy {info['numpy_version']}, xupy {info['xupy_version']}")
    if SciPyRBFInterpolator is None:
        print("scipy: NOT INSTALLED (only rbfgpu is timed)")
    else:
        import scipy

        print(f"scipy {scipy.__version__}")
    print(f"options: {options}\n")

    header = (
        f"{'npoints':>8} {'n_eval':>8} | {'build: rbfgpu':>14} {'scipy':>9} "
        f"{'x':>6} | {'eval: rbfgpu':>13} {'scipy':>9} {'x':>6} | {'max|diff|':>10}"
    )
    print(header)
    print("-" * len(header))

    for npoints in args.sizes:
        rng = np.random.default_rng(args.seed)
        y = rng.random((npoints, 2))
        d = np.sin(4 * y[:, 0]) * np.cos(4 * y[:, 1])
        x = rng.random((args.eval, 2))

        build_time, ours = timeit(
            lambda: RBFInterpolator(y, d, **options), args.repeats
        )
        eval_time, ours_values = timeit(lambda: to_host(ours(x)), args.repeats)

        if SciPyRBFInterpolator is None:
            print(
                f"{npoints:>8} {args.eval:>8} | {build_time:>14.3f} {'-':>9} "
                f"{'-':>6} | {eval_time:>13.3f} {'-':>9} {'-':>6} | {'-':>10}"
            )
            continue

        scipy_build_time, reference = timeit(
            lambda: SciPyRBFInterpolator(y, d, **options), args.repeats
        )
        scipy_eval_time, scipy_values = timeit(lambda: reference(x), args.repeats)

        diff = float(np.max(np.abs(ours_values - scipy_values)))
        print(
            f"{npoints:>8} {args.eval:>8} | {build_time:>14.3f} "
            f"{scipy_build_time:>9.3f} {scipy_build_time / build_time:>5.2f}x | "
            f"{eval_time:>13.3f} {scipy_eval_time:>9.3f} "
            f"{scipy_eval_time / eval_time:>5.2f}x | {diff:>10.2e}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
