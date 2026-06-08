import os
import json
import numpy as np
import scipy.stats as stats
from tqdm import tqdm
from multiprocessing import Pool

def _bh_fallback(p: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg q-values if SciPy FDR not available."""
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

_cortical_sheets = {}

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def write_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def run_stats(target: np.ndarray, baseline: np.ndarray):
    tvals, pvals = stats.ttest_ind(target, baseline, axis=0, equal_var=False, alternative='greater')
    qvals = fdr_qvalues(pvals)
    return dict(t=tvals, p=pvals, q=qvals)

def process_cluster(cluster_id):
    target = _cortical_sheets[cluster_id]  # shape: (n_videos, n_units)
    baseline = np.concatenate(
        [_cortical_sheets[cid] for cid in _cortical_sheets if cid != cluster_id], axis=0
    )  # shape: (n_videos * (n_clusters-1), n_units)
    num_target = target.shape[0]
    num_baseline = baseline.shape[0]
    stats_result = run_stats(target.reshape(num_target, -1), baseline.reshape(num_baseline, -1))
    return cluster_id, {
        "mean_t": float(np.mean(stats_result["t"])),
        "mean_significant_q=0.05_t": float(np.mean(stats_result["t"][stats_result["q"] < 0.05])),
        "mean_significant_q=0.01_t": float(np.mean(stats_result["t"][stats_result["q"] < 0.01])),
        "mean_significant_q=0.001_t": float(np.mean(stats_result["t"][stats_result["q"] < 0.001])),

        "median_t": float(np.median(stats_result["t"])),
        "median_significant_q=0.05_t": float(np.median(stats_result["t"][stats_result["q"] < 0.05])),
        "median_significant_q=0.01_t": float(np.median(stats_result["t"][stats_result["q"] < 0.01])),
        "median_significant_q=0.001_t": float(np.median(stats_result["t"][stats_result["q"] < 0.001])),

    }

if __name__ == "__main__":

    dirpath = "task-alignvideo/clustering_v2"
    cortical_sheet_paths = f"{dirpath}/cortical_sheets"
    manifest_path = f"{dirpath}/merged_clusters_tvals_v3.json"
    manifest_path = f"{dirpath}/clusters_tvals.json"

    data = read_json(manifest_path)

    for cluster in tqdm(data, desc="Loading cortical sheets"):
        cluster_id = cluster["cluster_id"]
        for video_id in cluster["video_ids"]:
            cortical_sheet_path = f"{cortical_sheet_paths}/{os.path.basename(video_id)}.npy"
            cortical_sheet = np.load(cortical_sheet_path)
            if cluster_id not in _cortical_sheets:
                _cortical_sheets[cluster_id] = []
            _cortical_sheets[cluster_id].append(cortical_sheet)
        _cortical_sheets[cluster_id] = np.stack(_cortical_sheets[cluster_id], axis=0)

    cluster_ids = list(_cortical_sheets.keys())
    results = []
    batch_size = 5
    for i in tqdm(range(0, len(cluster_ids), batch_size), desc="Computing selectivity (batches)"):
        batch = cluster_ids[i:i + batch_size]
        with Pool(processes=batch_size) as pool:
            batch_results = list(tqdm(
                pool.imap(process_cluster, batch),
                total=len(batch),
                desc=f"  Batch {i // batch_size + 1}",
                leave=False,
            ))
        results.extend(batch_results)

    selectivity_scores = dict(results)
    save_path = f"{dirpath}/cluster_selectivity_scores_v1.json"
    write_json(selectivity_scores, save_path)
