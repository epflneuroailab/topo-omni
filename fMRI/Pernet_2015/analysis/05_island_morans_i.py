#!/usr/bin/env python3
"""Island Moran's I for the Pernet voice-selective brain map (Fig. B3b).

Computes island Moran's I on the vocal > non-vocal group t-map projected to
fsaverage6 (FDR q<0.05, min island 8 vtx, 999 perms) — the same metric applied to
the model maps — and compares the brain against the model point estimates.

Lineage (docs/DESIGN.md §2.4): 02 surface projection → **05** → 06 plot.
  input : <results-root>/02_surface_projection/surface_data_fsaverage6.npz
  output: <results-root>/03_spatial_analysis/island_morans_i_results.json

PORT NOTES vs dev-repo `src/05_island_morans_i.py` (@ f842b1a):
  * imports `compute_island_morans_i` / `load_surface_adjacency` from **core.spatial_stats**
    (was an absolute cross-repo path into the Marvi repo — docs/DESIGN.md §4/§7);
  * results path is parameterized via `--results-root` (docs/DESIGN.md §5 path-agnostic);
  * model comparison values read from the vendored fixture
    `Pernet_2015/data/fig_b3b_model_island_morans_i.json` (docs/DESIGN.md §7/§10), not hard-coded;
  * **corrected the stale "speech_vs_nonspeech" label to "vocal_vs_nonvocal"** — the
    dev-repo docstring/`contrast` field was a copy-paste error; the analysis always ran
    the vocal>non-vocal npz (index §2.3 note, docs/DESIGN.md §7). This is a metadata reproduction
    fix; the golden-master test pins the numeric fields, not this string.

The Moran's I values are deterministic and golden-mastered
(Pernet_2015/tests/fixtures/island_morans_i_results.golden.json, atol=1e-9). The
permutation p-values are stochastic (unseeded) — reported but not pinned (docs/DESIGN.md §6).

STATUS: ported (Stage 1). Uses its own --results-root for now; will adopt core.paths
once that is implemented.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.stats import ttest_1samp

from core.spatial_stats import compute_island_morans_i, load_surface_adjacency

# --- Fixed analysis parameters (match the published run) ---
N_SUBJECTS = 218
DF = N_SUBJECTS - 1          # 217
FDR_Q = 0.05
MIN_SIZE = 8
N_PERMUTATIONS = 999
PERM_SEED = 42               # seed the island permutation p-values (reproducible; docs/DESIGN.md §6)

_HERE = Path(__file__).resolve().parent
_DATASET = _HERE.parent      # Pernet_2015/
MODEL_VALUES_JSON = _DATASET / "data" / "fig_b3b_model_island_morans_i.json"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", type=Path, required=True,
                   help="Root holding 02_surface_projection/ (input) and 03_spatial_analysis/ (output).")
    p.add_argument("--model-values", type=Path, default=MODEL_VALUES_JSON,
                   help="Vendored model island-Moran's-I point estimates for the comparison (Fig. B3b).")
    return p


def compute(results_root: Path, model_values_path: Path = MODEL_VALUES_JSON) -> dict:
    """Run the full island-Moran's-I analysis and return the results dict.

    Kept import-friendly (no I/O side effects beyond reading inputs) so the
    golden-master test can call it directly.
    """
    surface_npz = results_root / "02_surface_projection" / "surface_data_fsaverage6.npz"
    data = np.load(surface_npz)
    t_lh, t_rh = data["t_map_lh"], data["t_map_rh"]

    # One-tailed p-values for the positive direction.
    p_lh = stats.t.sf(t_lh, df=DF)
    p_rh = stats.t.sf(t_rh, df=DF)

    adj_lh = load_surface_adjacency("L")
    adj_rh = load_surface_adjacency("R")

    results = {}
    # Distinct per-hemisphere seeds so lh/rh permutations are reproducible yet independent.
    for hemi, t_map, p_map, adj, hemi_seed in [
        ("lh", t_lh, p_lh, adj_lh, PERM_SEED),
        ("rh", t_rh, p_rh, adj_rh, PERM_SEED + 1),
    ]:
        results[hemi] = compute_island_morans_i(
            t_map=t_map, p_map=p_map, adj_matrix=adj,
            fdr_q=FDR_Q, min_size=MIN_SIZE, n_permutations=N_PERMUTATIONS, df=DF,
            seed=hemi_seed,
        )

    # Per-island I values across both hemispheres.
    all_island_I = ([d["morans_i"] for d in results["lh"]["island_details"]]
                    + [d["morans_i"] for d in results["rh"]["island_details"]])
    n_islands_total = len(all_island_I)
    brain_mean = float(np.mean(all_island_I))
    brain_std = float(np.std(all_island_I, ddof=1)) if n_islands_total > 1 else float("nan")
    brain_se = brain_std / np.sqrt(n_islands_total) if n_islands_total > 1 else float("nan")

    # Model point estimates (vendored) + one-sample t-tests brain-islands vs model.
    model = json.loads(Path(model_values_path).read_text())
    TOPO_I = model["topo_omni_I"]
    NONTOPO_I = model["nontopo_I"]
    t_vs_topo, p_vs_topo = ttest_1samp(all_island_I, TOPO_I, alternative="greater")
    t_vs_nontopo, p_vs_nontopo = ttest_1samp(all_island_I, NONTOPO_I, alternative="greater")

    I_combined = float(np.nanmean([results["lh"]["I"], results["rh"]["I"]]))
    I_w_combined = float(np.nanmean([results["lh"]["I_weighted_all"], results["rh"]["I_weighted_all"]]))

    def _serialise(v):
        if isinstance(v, (float, np.floating)):
            return float(v)
        if isinstance(v, (int, np.integer)):
            return int(v)
        return v

    return {
        "method": "island_morans_i",
        "dataset": "pernet_2015",
        "contrast": "vocal_vs_nonvocal",   # corrected from stale "speech_vs_nonspeech"
        "surface": "fsaverage6",
        "n_subjects": N_SUBJECTS,
        "df": DF,
        "fdr_q": FDR_Q,
        "min_island_size": MIN_SIZE,
        "n_permutations": N_PERMUTATIONS,
        "lh": {k: _serialise(v) for k, v in results["lh"].items() if k != "island_details"},
        "rh": {k: _serialise(v) for k, v in results["rh"].items() if k != "island_details"},
        "all_island_I_values": [float(v) for v in all_island_I],
        "brain_mean_I": brain_mean,
        "brain_std_I": brain_std,
        "brain_se_I": brain_se,
        "n_islands_total": n_islands_total,
        "stats": {
            "topo_omni_I": TOPO_I,
            "nontopo_I": NONTOPO_I,
            "t_brain_vs_topo": float(t_vs_topo),
            "p_brain_vs_topo": float(p_vs_topo),
            "t_brain_vs_nontopo": float(t_vs_nontopo),
            "p_brain_vs_nontopo": float(p_vs_nontopo),
        },
        "avg_I_unweighted": I_combined,
        "avg_I_weighted": I_w_combined,
    }


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    output = compute(args.results_root, args.model_values)

    out_json = args.results_root / "03_spatial_analysis" / "island_morans_i_results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(output, indent=2))

    print(f"Island Moran's I — brain mean {output['brain_mean_I']:.4f} "
          f"across {output['n_islands_total']} islands "
          f"(lh {output['lh']['n_islands']}, rh {output['rh']['n_islands']})")
    print(f"Saved: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
