"""The optional ``seed`` on compute_island_morans_i makes permutation p-values reproducible.

esda 2.5.1's ``Moran`` has no seed argument and draws from the global NumPy RNG, so the
per-island permutation p-values are stochastic by default. Passing ``seed`` reseeds that
RNG before the island loop; two seeded runs must agree, and the deterministic Moran's I
values must be unaffected.
"""
import numpy as np
import pytest

pytest.importorskip("esda")
pytest.importorskip("libpysal")
from scipy.sparse import lil_matrix

from core.spatial_stats import compute_island_morans_i


def _ring_input(n=16):
    """A single ring island of ``n`` significant vertices with a smooth (autocorrelated)
    signal — enough for FDR to keep them all and for Moran's I to be well-defined."""
    adj = lil_matrix((n, n))
    for i in range(n):
        adj[i, (i + 1) % n] = 1
        adj[i, (i - 1) % n] = 1
    adj = adj.tocsr()
    # Large positive t everywhere -> all vertices survive FDR; smooth values -> real I.
    t_map = 8.0 + np.cos(np.linspace(0, 2 * np.pi, n, endpoint=False))
    from scipy import stats
    p_map = stats.t.sf(t_map, df=n - 1)
    return t_map, p_map, adj


def test_seed_makes_pvalues_reproducible():
    t_map, p_map, adj = _ring_input()
    kw = dict(t_map=t_map, p_map=p_map, adj_matrix=adj, fdr_q=0.5, min_size=8,
              n_permutations=199, df=15)
    a = compute_island_morans_i(seed=42, **kw)
    b = compute_island_morans_i(seed=42, **kw)
    assert a["n_islands"] >= 1                      # the ring qualifies as an island
    assert a["p_value"] == b["p_value"]             # permutation p-values reproduce
    assert a["I"] == pytest.approx(b["I"])          # deterministic I unaffected


def test_different_seeds_can_differ_but_I_is_stable():
    t_map, p_map, adj = _ring_input()
    kw = dict(t_map=t_map, p_map=p_map, adj_matrix=adj, fdr_q=0.5, min_size=8,
              n_permutations=199, df=15)
    a = compute_island_morans_i(seed=1, **kw)
    b = compute_island_morans_i(seed=2, **kw)
    # The deterministic Moran's I must not depend on the seed.
    assert a["I"] == pytest.approx(b["I"])
