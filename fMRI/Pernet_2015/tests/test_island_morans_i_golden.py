"""Golden-master test for Fig. B3b island Moran's I (docs/DESIGN.md §6 Tier 1).

Recomputes the analysis with core.spatial_stats and asserts the DETERMINISTIC fields
reproduce the published values in
``fixtures/island_morans_i_results.golden.json`` within the calibrated tolerance.

Calibration (measure→freeze, docs/DESIGN.md §6): first green port reproduced every
deterministic field to max |Δ| ≈ 1.6e-15 (libpysal 4.8.1 / esda 2.5.1, and — notably —
under nilearn 0.12.1 reproducing a 0.10.4-produced reference). Frozen tolerance
atol=1e-9 on the I values; exact equality on integer counts.

DATA-GATED: needs the surface npz (a Pipeline-1 output, not committed — docs/DESIGN.md §5/§6)
plus libpysal/esda/nilearn + fsaverage6. Skips cleanly when unavailable; the always-on
net is core/tests/test_spatial_stats.py. Point it at the data with PERNET_RESULTS_ROOT,
else it falls back to the dev results dir.

STOCHASTIC fields (unseeded esda permutation) are intentionally NOT asserted:
lh/rh ``p_value``, ``n_sig_morans_i``, ``I_sig_only``, ``I_weighted_sig``.
"""
import importlib.util
import json
import os
from pathlib import Path

import pytest

_DATASET = Path(__file__).resolve().parent.parent          # Pernet_2015/
_FIXTURE = _DATASET / "tests" / "fixtures" / "island_morans_i_results.golden.json"
_CANDIDATE_ROOTS = [
    os.environ.get("PERNET_RESULTS_ROOT"),
    "/work/upschrimpf1/mehrer/code/20241003_pernet_2015/results",
]

ATOL = 1e-9   # frozen from calibration (observed drift ~1.6e-15)


def _has_moran():
    try:
        import esda.moran      # noqa: F401
        import libpysal.weights  # noqa: F401
        import nilearn          # noqa: F401
        return True
    except ImportError:
        return False


def _resolve_results_root():
    for root in _CANDIDATE_ROOTS:
        if not root:
            continue
        npz = Path(root) / "02_surface_projection" / "surface_data_fsaverage6.npz"
        if npz.exists():
            return Path(root)
    return None


def _load_driver():
    path = _DATASET / "analysis" / "05_island_morans_i.py"
    spec = importlib.util.spec_from_file_location("pernet_05_island_morans_i", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pytestmark = pytest.mark.skipif(not _has_moran(), reason="libpysal/esda/nilearn not installed")


@pytest.fixture(scope="module")
def result_and_gold():
    root = _resolve_results_root()
    if root is None:
        pytest.skip("surface_data_fsaverage6.npz not found (set PERNET_RESULTS_ROOT)")
    driver = _load_driver()
    result = driver.compute(root)
    gold = json.loads(_FIXTURE.read_text())
    return result, gold


# Deterministic float fields (path into nested dict, tolerance).
_FLOAT_FIELDS = [
    ("lh", "I"), ("rh", "I"),
    ("lh", "I_weighted_all"), ("rh", "I_weighted_all"),
    ("brain_mean_I",), ("brain_std_I",),
    ("stats", "t_brain_vs_topo"), ("stats", "t_brain_vs_nontopo"),
    ("stats", "p_brain_vs_topo"), ("stats", "p_brain_vs_nontopo"),
    ("avg_I_weighted",),
]
# Deterministic integer invariants (exact).
_INT_FIELDS = [
    ("lh", "n_islands"), ("rh", "n_islands"),
    ("lh", "n_vertices"), ("rh", "n_vertices"),
    ("n_islands_total",), ("n_subjects",), ("df",),
]


def _dig(d, path):
    for k in path:
        d = d[k]
    return d


@pytest.mark.parametrize("path", _FLOAT_FIELDS, ids=lambda p: ".".join(p))
def test_deterministic_float_field(result_and_gold, path):
    result, gold = result_and_gold
    assert _dig(result, path) == pytest.approx(_dig(gold, path), abs=ATOL)


@pytest.mark.parametrize("path", _INT_FIELDS, ids=lambda p: ".".join(p))
def test_integer_invariant(result_and_gold, path):
    result, gold = result_and_gold
    assert _dig(result, path) == _dig(gold, path)


def test_per_island_I_values(result_and_gold):
    result, gold = result_and_gold
    r, g = result["all_island_I_values"], gold["all_island_I_values"]
    assert len(r) == len(g)
    for rv, gv in zip(r, g):
        assert rv == pytest.approx(gv, abs=ATOL)


def test_contrast_label_corrected(result_and_gold):
    # We fixed the stale "speech_vs_nonspeech" metadata label (docs/DESIGN.md §7); the golden
    # file still carries the old string — assert we now emit the correct one.
    result, _ = result_and_gold
    assert result["contrast"] == "vocal_vs_nonvocal"
