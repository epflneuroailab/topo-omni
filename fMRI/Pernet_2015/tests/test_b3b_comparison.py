"""Golden-ish master for Fig. B3b step 06 — the model bars + brain-vs-model stats.

The chart itself is a visual artefact (not pinned), but the numbers it draws are
reproducible and come from the vendored per-island distributions
(data/model_island_morans_i/, sha256-pinned in data/PROVENANCE.md). This locks:

  * the model bar heights (mean island Moran's I) and SEs against the published values
    (Topo-Omni 0.5749 ± 0.0416, n=79; Non-Topo 0.2353 ± 0.0120, n=418), and
  * the brain-vs-model comparison directions given a known brain mean.

These are the two model bars the published Fig. B3b was drawn from — superseding the
orphaned point-estimate literals (0.594 / 0.126) that never matched the figure.
"""
import importlib.util
import json
from pathlib import Path

import pytest

_DATASET = Path(__file__).resolve().parent.parent


def _load_06():
    path = _DATASET / "analysis" / "06_island_morans_i_comparison.py"
    spec = importlib.util.spec_from_file_location("pernet_06_comparison", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    "name,exp_mean,exp_se,exp_n",
    [("topoomni", 0.574898, 0.041570, 79),
     ("nontopo", 0.235253, 0.012036, 418)],
)
def test_model_bars_match_published(name, exp_mean, exp_se, exp_n):
    import numpy as np
    m = _load_06()
    vals = m.load_model_islands(name)
    assert vals.size == exp_n
    assert float(np.mean(vals)) == pytest.approx(exp_mean, abs=1e-5)
    se = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
    assert se == pytest.approx(exp_se, abs=1e-5)


def test_compute_bars_directions(tmp_path):
    """With a brain mean between the two models, Topo n.s. and Non-Topo highly sig."""
    m = _load_06()
    # Stage a minimal step-05 output: brain mean 0.53, a handful of island values.
    brain = {"avg_I_unweighted": 0.53,
             "all_island_I_values": [0.4, 0.5, 0.55, 0.6, 0.5, 0.55, 0.5, 0.6]}
    d = tmp_path / "03_spatial_analysis"
    d.mkdir(parents=True)
    (d / "island_morans_i_results.json").write_text(json.dumps(brain))

    result = m.compute_bars(tmp_path)
    assert result["bars"]["topoomni"]["mean"] == pytest.approx(0.574898, abs=1e-5)
    assert result["bars"]["nontopo"]["mean"] == pytest.approx(0.235253, abs=1e-5)
    # brain (0.53) >> non-topo distribution -> significant; ~ topo distribution -> n.s.
    assert result["stats"]["nontopo"]["p_ttest"] < 1e-10
    assert result["stats"]["topoomni"]["p_ttest"] > 0.05
