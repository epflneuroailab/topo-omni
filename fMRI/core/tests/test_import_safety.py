"""Enforce the core version-robustness invariant (docs/DESIGN.md §4).

`core` is imported under BOTH nilearn 0.10.4 (Pernet) and 0.12.1 (Marvi/Jung), so it
must NOT bind nilearn at import time — surface.py imports nilearn lazily inside
functions. These tests are real and pass at scaffold stage; they guard against a
future edit that adds a top-level `import nilearn` and silently breaks one env.
"""
import importlib
import sys


def test_core_imports_without_nilearn():
    # Drop any prior nilearn import so we observe what `core` itself pulls in.
    sys.modules.pop("nilearn", None)
    for mod in ("core", "core.spatial_stats", "core.surface", "core.paths", "core.froi_cv",
                "core.precomputed"):
        importlib.import_module(mod)
    assert "nilearn" not in sys.modules, (
        "Importing `core` pulled in nilearn at module scope — this breaks the "
        "0.10/0.12 dual-version constraint (docs/DESIGN.md §4). Import nilearn lazily "
        "inside functions in surface.py."
    )


def test_core_exposes_expected_modules():
    import core
    assert set(core.__all__) == {"spatial_stats", "surface", "paths", "froi_cv"}
