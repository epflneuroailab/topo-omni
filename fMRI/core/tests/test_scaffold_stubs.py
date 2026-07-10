"""Placeholder tests for the core stubs still awaiting a port (docs/DESIGN.md §6 Tier 2).

As each utility is ported, its entry is removed here and a real test replaces it:
  - paths:    --input-source / --*-root resolution (real TDD — new code).
  - froi_cv:  factored during the Marvi port (may stay local instead).
spatial_stats is PORTED — see test_spatial_stats.py.
surface is PORTED — see test_surface.py.
"""
import pytest

from core import froi_cv, paths


@pytest.mark.parametrize("fn", [
    lambda: paths.resolve(None),
    lambda: froi_cv.cross_validated_froi_response(),
])
def test_stub_raises_not_implemented(fn):
    with pytest.raises(NotImplementedError):
        fn()
