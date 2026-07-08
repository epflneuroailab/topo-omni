"""Unit tests for core.spatial_stats (docs/DESIGN.md §6 Tier 2).

Two always-runnable, self-contained checks (no external data / no nilearn):
  - compute_fdr_threshold: pure numpy/scipy, hand-checkable → runs everywhere.
  - compute_island_morans_i on a hand-built adjacency with known clustering →
    needs libpysal/esda (skipped if absent), but no mesh fetch / no real data.

The heavy end-to-end golden master (real Pernet npz → published I values) lives in
Pernet_2015/tests/ (data-gated).
"""
import numpy as np
import pytest
from scipy import sparse, stats

from core import spatial_stats as ss


def _has_moran():
    try:
        import esda.moran  # noqa: F401
        import libpysal.weights  # noqa: F401
        return True
    except ImportError:
        return False


requires_moran = pytest.mark.skipif(not _has_moran(), reason="libpysal/esda not installed")


# ---------------------------------------------------------------------------
# compute_fdr_threshold — pure, no optional deps
# ---------------------------------------------------------------------------

def test_fdr_threshold_all_significant_recovers_min_t():
    # All vertices highly significant → BH keeps all → t-threshold ≈ smallest t.
    df = 100
    t_map = np.linspace(6.0, 12.0, 50)
    p_map = stats.t.sf(t_map, df=df)
    thr = ss.compute_fdr_threshold(t_map, p_map, q=0.05, df=df)
    assert np.isfinite(thr)
    # Threshold recovers the smallest t up to the t.ppf(1 - t.sf(·)) round-trip error
    # (~7e-10 here), which may land marginally above min-t and drop just the boundary
    # vertex — so essentially all survive, not necessarily every single one.
    assert thr == pytest.approx(t_map.min(), abs=1e-6)
    assert (t_map >= thr).sum() >= len(t_map) - 1


def test_fdr_threshold_returns_inf_when_nothing_positive():
    t_map = np.array([-1.0, -2.0, -0.5])
    p_map = np.array([0.9, 0.99, 0.8])
    assert np.isinf(ss.compute_fdr_threshold(t_map, p_map, q=0.05, df=10))


def test_fdr_threshold_returns_inf_when_none_survive():
    # Uniformly non-significant p-values → BH selects nothing → inf.
    t_map = np.array([0.1, 0.2, 0.15, 0.05])
    p_map = np.array([0.9, 0.8, 0.85, 0.95])
    assert np.isinf(ss.compute_fdr_threshold(t_map, p_map, q=0.05, df=10))


# ---------------------------------------------------------------------------
# compute_island_morans_i — hand-built two-island adjacency
# ---------------------------------------------------------------------------

def _two_chain_adjacency(block=20):
    """Two disconnected path graphs of `block` vertices each (2*block total).

    Within-block chain edges only; the two chains never touch → exactly two
    connected components (islands), each ≥ min_size.
    """
    n = 2 * block
    rows, cols = [], []
    for start in (0, block):
        for i in range(start, start + block - 1):
            rows += [i, i + 1]
            cols += [i + 1, i]
    data = np.ones(len(rows))
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))


@requires_moran
def test_island_morans_i_finds_two_smooth_islands():
    block = 20
    adj = _two_chain_adjacency(block)
    df = 38
    # Smooth linear ramp along each chain → strong positive spatial autocorrelation.
    ramp = np.linspace(6.0, 12.0, block)
    t_map = np.concatenate([ramp, ramp])
    p_map = stats.t.sf(t_map, df=df)

    res = ss.compute_island_morans_i(t_map, p_map, adj, fdr_q=0.05, min_size=8,
                                     n_permutations=99, df=df)
    assert res["n_islands"] == 2
    assert res["n_vertices"] == 2 * block
    # A monotone ramp on a chain is strongly, positively autocorrelated.
    assert res["I"] > 0.5
    assert all(d["morans_i"] > 0.5 for d in res["island_details"])


@requires_moran
def test_island_morans_i_skips_below_min_size():
    # Two chains of 5 each; min_size=8 → no island qualifies → NaN / zero islands.
    block = 5
    adj = _two_chain_adjacency(block)
    df = 8
    t_map = np.concatenate([np.linspace(6, 10, block), np.linspace(6, 10, block)])
    p_map = stats.t.sf(t_map, df=df)
    res = ss.compute_island_morans_i(t_map, p_map, adj, fdr_q=0.05, min_size=8,
                                     n_permutations=99, df=df)
    assert res["n_islands"] == 0
    assert np.isnan(res["I"])


def test_import_does_not_bind_nilearn():
    # Reinforce the core invariant locally (also covered in test_import_safety).
    import sys
    assert "nilearn" not in sys.modules or True  # tolerant: another test may have loaded it
    # The real guarantee: importing spatial_stats itself pulls no nilearn.
    import importlib
    sys.modules.pop("nilearn", None)
    importlib.reload(ss)
    assert "nilearn" not in sys.modules
