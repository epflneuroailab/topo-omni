"""Per-cluster selectivity of the cortical sheet: each cluster's clips vs all others.

One Welch t per unit, FDR-corrected across the sheet, summarised as the mean/median t
over all units and over the units surviving q < {0.05, 0.01, 0.001}. Computed on the
*raw* early-stopped clusters (``clusters_tvals.json``), not the hand-merged ones, so
the score reflects what the discovery step actually found.

    python -m src.experiments.spacetop_discovery.calculate_selectivity \
        --model_name topo-omni-tiled32-4608
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np
import scipy.stats as stats
from tqdm import tqdm

from src.experiments.spacetop_discovery.common import (
    add_run_args, read_json, resolve_run_args, write_json,
)

_cortical_sheets = {}
# Sheets of clips the walk left in no cluster (the *_unassigned.json sidecar). They are
# baseline only: the walk scored each cluster against EVERY other clip, so dropping them
# here would score the same cluster against a different baseline.
_unassigned = None


def _bh_fallback(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values if SciPy FDR not available."""
    p = np.asarray(p, dtype=float)
    n = p.size
    order = np.argsort(p)
    q = np.empty_like(p)
    cummin = 1.0
    for rank_from_end, idx in enumerate(order[::-1], start=1):
        rank = n - rank_from_end + 1
        val = p[idx] * n / rank
        cummin = min(cummin, val)
        q[idx] = cummin
    return np.minimum(q, 1.0)


def fdr_qvalues(p: np.ndarray) -> np.ndarray:
    try:
        return stats.false_discovery_control(p)
    except Exception:
        return _bh_fallback(p)


def run_stats(target: np.ndarray, baseline: np.ndarray):
    tvals, pvals = stats.ttest_ind(target, baseline, axis=0, equal_var=False,
                                   alternative='greater')
    qvals = fdr_qvalues(pvals)
    return dict(t=tvals, p=pvals, q=qvals)


def _summarise(t, q):
    def at(thresh, fn):
        sel = t[q < thresh]
        return float(fn(sel)) if sel.size else float('nan')
    return {
        "mean_t": float(np.mean(t)),
        "mean_significant_q=0.05_t": at(0.05, np.mean),
        "mean_significant_q=0.01_t": at(0.01, np.mean),
        "mean_significant_q=0.001_t": at(0.001, np.mean),
        "median_t": float(np.median(t)),
        "median_significant_q=0.05_t": at(0.05, np.median),
        "median_significant_q=0.01_t": at(0.01, np.median),
        "median_significant_q=0.001_t": at(0.001, np.median),
    }


def process_cluster(cluster_id):
    target = _cortical_sheets[cluster_id]                       # (n_videos, n_units)
    parts = [_cortical_sheets[cid] for cid in _cortical_sheets if cid != cluster_id]
    if _unassigned is not None:
        parts.append(_unassigned)
    baseline = np.concatenate(parts, axis=0)
    result = run_stats(target.reshape(target.shape[0], -1),
                       baseline.reshape(baseline.shape[0], -1))
    return cluster_id, _summarise(result["t"], result["q"])


def main():
    parser = argparse.ArgumentParser(
        description="Per-cluster selectivity scores over the unified cortical sheet.")
    add_run_args(parser)
    parser.add_argument("--clusters_json", type=str, default=None,
                        help="Default: <output_dir>/clusters_tvals.json")
    parser.add_argument("--save_name", type=str, default="cluster_selectivity_scores_v1.json")
    parser.add_argument("--unassigned_json", type=str, default=None,
                        help="clips in no cluster, added to every baseline. Default: "
                             "<output_dir>/linkage_tree_tvals_min=10_max=500_unassigned.json "
                             "if it exists.")
    parser.add_argument("--workers", type=int, default=5,
                        help="Clusters scored in parallel. Each holds a full copy of the "
                             "baseline, so this trades RAM for wall clock.")
    args = resolve_run_args(parser.parse_args())

    out = args.output_dir
    sheets_dir = os.path.join(out, "cortical_sheets")
    manifest_path = args.clusters_json or os.path.join(out, "clusters_tvals.json")

    data = read_json(manifest_path)

    for cluster in tqdm(data, desc="Loading cortical sheets"):
        cluster_id = cluster["cluster_id"]
        sheets = []
        for video_id in cluster["video_ids"]:
            path = os.path.join(sheets_dir, f"{os.path.basename(video_id)}.npy")
            if not os.path.exists(path):
                print(f"  ! missing sheet, skipping: {path}")
                continue
            sheets.append(np.load(path))
        if sheets:
            _cortical_sheets[cluster_id] = np.stack(sheets, axis=0)

    global _unassigned
    unassigned_path = args.unassigned_json or os.path.join(
        out, "linkage_tree_tvals_min=10_max=500_unassigned.json")
    if os.path.exists(unassigned_path):
        paths = [os.path.join(sheets_dir, f"{os.path.basename(v)}.npy")
                 for v in read_json(unassigned_path)]
        extra = [np.load(p) for p in paths if os.path.exists(p)]
        if extra:
            _unassigned = np.stack(extra, axis=0)
            print(f"Baseline also includes {len(extra)} unassigned clips ({unassigned_path})")

    cluster_ids = list(_cortical_sheets.keys())
    results = []
    for i in tqdm(range(0, len(cluster_ids), args.workers),
                  desc="Computing selectivity (batches)"):
        batch = cluster_ids[i:i + args.workers]
        with Pool(processes=len(batch)) as pool:
            results.extend(pool.imap(process_cluster, batch))

    save_path = os.path.join(out, args.save_name)
    write_json(dict(results), save_path)
    print(f"Wrote {save_path}")


if __name__ == "__main__":
    main()
