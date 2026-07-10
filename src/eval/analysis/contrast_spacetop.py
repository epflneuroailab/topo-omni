import json
import argparse
from pathlib import Path
from typing import List, Tuple

import os
import yaml
import pickle
import argparse
import numpy as np

import torch
import matplotlib.pyplot as plt

import seaborn as sns
import scipy.stats as stats
from PIL import Image
from tqdm import tqdm

from glob import glob
from skimage import measure
from scipy.ndimage import binary_opening

from scipy.sparse.csgraph import connected_components
from src.utils.smoothing import NeuronSmoothingConv
from src.utils.spatial_stats import compute_standard_morans_i, compute_island_morans_i, compute_fdr_threshold
from src.core.model_loading import unified_grid_coords, MODEL_TITLE

from dotenv import load_dotenv
load_dotenv()

SAVE_DIR = os.getenv("SAVE_DIR")
CKPT_DIR = os.getenv("CKPT_DIR")
REPO_DIR = os.getenv("REPO_DIR")

def read_json(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def remove_small_components(mask, min_size):
    labeled = measure.label(mask)
    cleaned = np.zeros_like(mask)
    for region in measure.regionprops(labeled):
        if region.area >= min_size:
            cleaned[labeled == region.label] = 1
    return cleaned

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def dump_pickle(obj, fpath: Path):
    with fpath.open("wb") as f:
        pickle.dump(obj, f)

def load_pickle(fpath: Path):
    with fpath.open("rb") as f:
        obj = pickle.load(f)
    return obj  

def write_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

P_VALUE_THRESHOLD = 0.001

# -------------------------------------------------
# Small helpers (shared)
# -------------------------------------------------

def pil_open(path: str):
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def cohen_d(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    mean1, mean2 = x1.mean(axis=0), x2.mean(axis=0)
    std1, std2 = x1.std(axis=0, ddof=1), x2.std(axis=0, ddof=1)
    n1, n2 = x1.shape[0], x2.shape[0]
    pooled = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) /
                     max(n1 + n2 - 2, 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (mean1 - mean2) / pooled
        d[~np.isfinite(d)] = 0.0
    return d


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


def attention_last_indices(attn_mask: torch.Tensor) -> torch.Tensor:
    lengths = attn_mask.sum(dim=1)
    return torch.clamp(lengths - 1, min=0).long()


def infer_grid_from_coords(coords: np.array) -> Tuple[int, int]:
    xs = coords[:, 0].astype(np.int64)
    ys = coords[:, 1].astype(np.int64)
    H = int(ys.max()) + 1
    W = int(xs.max()) + 1
    return H, W


def place_on_grid(values: np.ndarray, coords: np.array) -> np.ndarray:
    H, W = infer_grid_from_coords(coords)
    grid = np.full((H, W), np.nan, dtype=np.float32)
    ij = coords.astype(np.int64)
    grid[ij[:, 1], ij[:, 0]] = values
    return grid


def ensure_device_dtype(batch: dict, device: torch.device):
    for k, v in batch.items():
        if not isinstance(v, torch.Tensor):
            continue
        v = v.to(device)
        if device.type == "cuda" and v.is_floating_point():
            v = v.to(torch.bfloat16)
        batch[k] = v
    return batch


# -------------------------------------------------
# Core stats logic
# -------------------------------------------------

def run_stats(pos: np.ndarray, neg: np.ndarray):
    # one-sided t-test for each feature (pos > neg)
    tvals, pvals = stats.ttest_ind(pos, neg, axis=0, equal_var=False, alternative='greater')
    qvals = fdr_qvalues(pvals.flatten())
    dvals = cohen_d(pos, neg)
    return dict(
        t=tvals, 
        p=pvals, 
        q=qvals, 
        d=dvals, 
    )


def maybe_hrf_project(feats: np.ndarray,
                      coords: np.array,
                      smoother: NeuronSmoothingConv | None):
    """
    If smoothing is enabled, project per-neuron activations to a regular grid
    using the HRF Gaussian kernel. Returns (H, W, projected_feats).
    Otherwise returns (None, None, feats).
    """
    if (smoother is None) or (coords is None) or (feats is None):
        return None, None, feats
    xy = coords.astype(int)
    # positions = torch.from_numpy(xy).float().to(smoother.device)      # (N_neurons, 2)
    # activations = torch.from_numpy(feats).float().to(smoother.device)  # (N_samples, N_neurons)
    projected = smoother(xy, feats)   # projected: (N_samples, N_grid)
    # H, W = _hrf_grid_shape(gridx, gridy)
    H = smoother.height
    W = smoother.width
    return H, W, projected


def run_contrast(target_cluster_id: int, topk_pct: float = 1):

    print(f">> Running contrast for target cluster: {target_cluster_id}")

    moorans_I_results = {}

    model_name = MODEL_TITLE

    save_dir = f"{SAVE_DIR}/{model_name}/spacetop_clusters_figures/{str(target_cluster_id).zfill(2)}"
    os.makedirs(save_dir, exist_ok=True)

    clusters_dir = f"{SAVE_DIR}/{model_name}/spacetop_clusters"
    clusters_dir = f"{REPO_DIR}/topo-discover/task-alignvideo/clustering_v2/cortical_sheets"
    cluster_files = sorted(glob(f"{clusters_dir}/*.npy"))

    manifest_path = f"{REPO_DIR}/topo-discover/task-alignvideo/clustering_v2/clusters_tvals.json"

    clusters_info = read_json(manifest_path)

    # baseline_cluster_dir = f"{SAVE_DIR}/{model_name}/spacetop_clusters/spacetop_rating_v0"
    # baseline_cluster_files = sorted(glob(f"{baseline_cluster_dir}/*.npy"))

    target_data = []
    other_data = []

    target_cluster_info = [info for info in clusters_info if info["cluster_id"] == target_cluster_id][0]
    target_video_ids = set(target_cluster_info["video_ids"])
    
    other_cluster_ids = []
    for cluster_file in tqdm(cluster_files):
        data = np.load(cluster_file)
        video_id = os.path.basename(cluster_file).replace(".npy", "")
        if video_id in target_video_ids:
            target_data.append(data)
        else:
            other_data.append(data)
            other_cluster_ids.append(video_id)
    
    # for cluster_file in tqdm(baseline_cluster_files):
    #     data = np.load(cluster_file)
    #     cluster_id = os.path.basename(cluster_file).replace(".npy", "")
    #     other_data.append(data)
    #     other_cluster_ids.append(cluster_id)
            
    H, W = 304, 512
    USE_SMOOTH = True

    target_data = np.concatenate(target_data, axis=0)
    other_data = np.concatenate(other_data, axis=0)

    # drop any rows with NaN values (if any)
    target_data = target_data[~np.isnan(target_data).any(axis=1)]
    other_data = other_data[~np.isnan(other_data).any(axis=1)]

    target_data = target_data.reshape(-1, H*W)
    other_data = other_data.reshape(-1, H*W)

    print(f"Target data shape: {target_data.shape}")
    print(f"Other data shape: {other_data.shape}")

    chosen_indices = np.random.choice(other_data.shape[0], size=min(10_000, other_data.shape[0]), replace=False)
    other_cluster_ids = np.array(other_cluster_ids)
    other_cluster_ids = other_cluster_ids[chosen_indices]
    # write_json(other_cluster_ids.tolist(), f"{save_dir}/{target_cluster_id}_other_cluster_ids.json")

    other_data = other_data[chosen_indices]
    print(f"Filtered other data shape: {other_data.shape}")

    coords_lm = unified_grid_coords()

    if USE_SMOOTH:
        fwhm_mm = 4.0 # {2.0, 4.0, 8.0, 12.0, 16.0} inspired by typical fMRI smoothing kernels, but can be tuned based on expected spatial scale of clusters
        resolution_mm = 1.0 # {1.0} grid resolution in mm, can be tuned based on neuron density and desired granularity of spatial patterns
        smoother = NeuronSmoothingConv(fwhm_mm=fwhm_mm, resolution_mm=resolution_mm)
    else:
        smoother = None

    H_lm, W_lm, target_data_proj = maybe_hrf_project(target_data, coords_lm, smoother)
    _, _, other_data_proj = maybe_hrf_project(other_data, coords_lm, smoother)
    
    print(f"> Running stats for LM decoder selectivity")
    stats_lm = run_stats(target_data_proj, other_data_proj)
    
    if H_lm is not None and W_lm is not None:
        p_map = stats_lm["p"].reshape(H_lm, W_lm).copy()
        t_map = stats_lm["t"].reshape(H_lm, W_lm).copy()
    else:
        p_map = place_on_grid(stats_lm["p"], coords_lm)
        t_map = place_on_grid(stats_lm["t"], coords_lm)

    # mask audio part of the sheet by setting p-values to inf (non-significant) so they are ignored in Moran's I computation
    # p_map = np.rot90(p_map, k=1)
    # t_map = np.rot90(t_map, k=1)
    # p_map[144:, 256:] = np.inf
    # t_map[144:, 256:] = -np.inf

    # only take the top-1% of significant units to focus on strongest effects and reduce noise for Moran's I
    t_threshold = np.percentile(t_map.flatten(), 100-topk_pct)

    # fdr_t_threshold = compute_fdr_threshold(t_map.flatten(), p_map.flatten(), q=P_VALUE_THRESHOLD)

    t_map_mask = (t_map >= t_threshold)

    t_map_mask = binary_opening(t_map_mask, structure=np.ones((3, 3)))
    t_map_mask = remove_small_components(t_map_mask, min_size=20)

    # morans_I = compute_standard_morans_i(t_map)
    morans_I = compute_island_morans_i(t_map, p_map, fdr_q=P_VALUE_THRESHOLD)

    # print(f"LM Decoder Selectivity Island Moran's I for sheet: {island_morans_I_value_all:.4f} | Weighted I-value: {weighted_average_moran_I_all:.4f} |  num_sig: {num_sig}/{total_num_islands}")
    print(f"LM Decoder Selectivity Island Moran's I for sheet: {morans_I['I']}")

    moorans_I_results[target_cluster_id] = morans_I

    t_map = stats_lm["t"].copy()

    if H_lm is not None and W_lm is not None:
        grid_lm_mask = t_map.reshape(H_lm, W_lm)
    else:
        grid_lm_mask = place_on_grid(t_map, coords_lm)

    np.save(f"{save_dir}/cluster_{str(target_cluster_id).zfill(2)}_t_map.npy", grid_lm_mask)
    
    grid_lm_mask[~t_map_mask] = np.nan
  
    fig, ax = plt.subplots(figsize=(12, 6))

    # rotate grid_lm_mask 90 degrees to the left
    grid_lm_mask = np.rot90(grid_lm_mask, k=1)

    # grid_lm_mask[144:, 256:] = np.nan

    sns.heatmap(grid_lm_mask, cbar=True, cmap="viridis", ax=ax,
                linewidths=0, linecolor=None)

    # add horizontal dashed line at y = 160
    ax.axhline(y=304 - 160, color='gray', linestyle='--')

    # add vertical dashed line at x = 256 until y = 160
    ax.vlines(x=256, ymin=304 - 160, ymax=304, color='gray', linestyle='--')

    # remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # title_lm = f"Cluster {target_cluster_id.split('_')[1]} vs. Other Clusters | Moran's I: {morans_I['I']:.3f}"
    # ax.set_title(title_lm, pad=18)

    fig.tight_layout()

    out_svg = f"{save_dir}/cluster_{str(target_cluster_id).zfill(2)}_selectivity_top{int(topk_pct)}.svg"
    out_png = f"{save_dir}/cluster_{str(target_cluster_id).zfill(2)}_selectivity_top{int(topk_pct)}.png"

    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    plt.clf()
    plt.cla()

    print(f"✓ saved {out_svg}")
    print(f"✓ saved {out_png}")

    moorans_I_results = {}
    moorans_I_results["t_stats"] = {
        "median": float(np.nanmedian(stats_lm["t"])),
        "mean": float(np.nanmean(stats_lm["t"])),
        "max": float(np.nanmax(stats_lm["t"])),
    }

    # sort moorans_I_results by value descending
    # moorans_I_results = dict(sorted(moorans_I_results.items(), key=lambda item: item[1]["I"], reverse=True))
    write_json(moorans_I_results, f"{save_dir}/island_morans_I_results_rating_audio=True_top{int(topk_pct)}.json")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute spatial clustering stats for SpaceTop audio clusters.")
    parser.add_argument("--cluster_id", type=int, default=5, help="ID of the target cluster to analyze")
    parser.add_argument("--topk", type=int, default=1, help="Percentage of top-k significant units to consider")
    args = parser.parse_args()

    np.random.seed(42)

    target_cluster_ids = [args.cluster_id]
    run_contrast(target_cluster_ids[0])

    # # # parallelize run_cluster function
    # from joblib import Parallel, delayed
    # Parallel(n_jobs=10)(delayed(run_contrast)(cluster_id, topk_pct=args.topk) for cluster_id in target_cluster_ids)
