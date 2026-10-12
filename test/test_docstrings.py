"""Run the docstring examples, so that the documented API cannot rot."""

import doctest

import pytest

import rbfgpu
from rbfgpu import interpolator

MODULES_WITH_EXAMPLES = [rbfgpu, interpolator]


@pytest.mark.parametrize(
    "module", MODULES_WITH_EXAMPLES, ids=[m.__name__ for m in MODULES_WITH_EXAMPLES]
)
def test_docstring_examples(module):
    results = doctest.testmod(
        module, verbose=False, optionflags=doctest.NORMALIZE_WHITESPACE
    )
    assert results.attempted > 0, f"no example left in {module.__name__}"
    assert results.failed == 0, f"{results.failed} doctest failure(s)"
