#!/usr/bin/env python3
"""
Compute group IoU of thresholded vertex masks across selected clusters, with a
random-sampling null distribution for significance testing.

Two thresholding modes are supported:

  topx  (default)
        Top N% of t-statistic vertices, computed jointly across LH and RH.
        Every cluster contributes a fixed number of vertices (~820 for top 1%
        of fsaverage6), making the comparison fair across clusters.

  fdr
        Vertices surviving Benjamini-Hochberg FDR correction at the chosen
        q-level (one-tailed, positive t-values only).  Mask size varies per
        cluster; clusters with zero FDR-significant vertices are reported but
        excluded from the IoU computation.

Group IoU is:

    IoU = |intersection of all masks| / |union of all masks|

A null distribution is built by repeatedly drawing groups of the same size
uniformly at random from all available clusters (without replacement).

Usage
-----
Top-1% (default):
    python 38_cluster_top1pct_iou.py --groups "3,14,25,31"

Top-5%:
    python 38_cluster_top1pct_iou.py --groups "3,14,25,31" --top-percentile 5.0

FDR q<0.05:
    python 38_cluster_top1pct_iou.py --groups "3,14,25,31" --threshold-mode fdr

FDR q<0.01:
    python 38_cluster_top1pct_iou.py --groups "3,14,25,31" --threshold-mode fdr --fdr-q 0.01

Multiple named groups:
    python 38_cluster_top1pct_iou.py \\
        --groups "3,14,25" "31,32,47,48" \\
        --group-names "visual" "auditory"
"""

import os
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from glob import glob

from dotenv import load_dotenv
load_dotenv()

SAVE_DIR = os.getenv("SAVE_DIR")
MODEL_NAME = "topo-omni"

CONTRAST_DIR = Path(f"{SAVE_DIR}/{MODEL_NAME}/spacetop_clusters_figures")
OUTPUT_DIR = Path(f"{SAVE_DIR}/{MODEL_NAME}/spacetop_clusters_iou")


# CONTRAST_DIR = Path(
#     '/work/upschrimpf1/mehrer/datasets/fMRI_movie_watching/spacetop/'
#     'ds005256/derivatives/cluster_contrasts_standard/group_level'
# )
# OUTPUT_DIR = Path(
#     '/work/upschrimpf1/mehrer/code/20251211_fMRI_movie_watching_spacetop/'
#     'results/overarching_handpicked_clusters'
# )

# ---------------------------------------------------------------------------
# Thresholding helpers
# ---------------------------------------------------------------------------

def _fdr_threshold(model_map: np.ndarray, q: float, df: int) -> float:
    """
    Return the Benjamini-Hochberg FDR t-threshold (one-tailed, positive direction).
    Returns np.inf if no vertex survives correction.
    """
    pos = model_map[model_map>0]
    if len(pos) == 0:
        return np.inf
    p_vals = stats.t.sf(pos, df=df)
    sorted_p = np.sort(p_vals)
    n = len(sorted_p)
    bh = sorted_p <= (np.arange(1, n + 1) / n) * q
    if not bh.any():
        return np.inf
    p_thresh = sorted_p[bh][-1]
    return float(stats.t.ppf(1.0 - p_thresh, df=df))


def topx_mask(lh: np.ndarray, rh: np.ndarray,
              top_pct: float) -> tuple[np.ndarray, np.ndarray]:
    """Binary masks for vertices in the top N% of t-statistics (LH + RH joint)."""
    threshold = np.percentile(np.concatenate([lh, rh]), 100.0 - top_pct)
    return lh >= threshold, rh >= threshold

def model_topx_mask(t_map: np.ndarray,
              top_pct: float) -> tuple[np.ndarray, np.ndarray]:
    """Binary masks for vertices in the top N% of t-statistics (LH + RH joint)."""
    threshold = np.percentile(t_map, 100.0 - top_pct)
    return t_map >= threshold

def fdr_mask(lh: np.ndarray, rh: np.ndarray,
             q: float, df: int) -> tuple[np.ndarray, np.ndarray]:
    """Binary masks for vertices surviving FDR correction at level q."""
    t_thresh = _fdr_threshold(lh, rh, q, df)
    return lh >= t_thresh, rh >= t_thresh


def model_fdr_mask(model_mask, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Binary masks for vertices surviving FDR correction at level q."""
    t_thresh = _fdr_threshold(model_mask, q, df)
    return model_mask >= t_thresh,


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_masks(contrast_dir: Path, mode: str,
                   top_pct: float = 1.0, fdr_q: float = 0.05,
                   n_subjects: int = 83) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """
    Pre-load binary vertex masks for every cluster found in contrast_dir.

    Parameters
    ----------
    mode : {'topx', 'fdr'}
    top_pct : used when mode == 'topx'
    fdr_q, n_subjects : used when mode == 'fdr'

    Returns
    -------
    lh_masks : ndarray, shape (n_clusters, n_vertices_lh)
    rh_masks : ndarray, shape (n_clusters, n_vertices_rh)
    cluster_ids : list of int
    n_sig_per_cluster : list of int  — number of vertices in the mask per cluster
    """
    lh_files = sorted(contrast_dir.glob('group_cluster-*_space-fsaverage6_hemi-L_tstat.func.gii'))
    if not lh_files:
        raise FileNotFoundError(f'No GIFTI t-stat files found in {contrast_dir}')

    cluster_ids = [int(f.name.split('_')[1].split('-')[1]) for f in lh_files]
    df = n_subjects - 1

    lh_masks, rh_masks, n_sig = [], [], []
    for f in lh_files:
        lh = nib.load(f).darrays[0].data
        rh = nib.load(str(f).replace('hemi-L', 'hemi-R')).darrays[0].data
        if mode == 'fdr':
            lh_m, rh_m = fdr_mask(lh, rh, fdr_q, df)
        else:
            lh_m, rh_m = topx_mask(lh, rh, top_pct)
        lh_masks.append(lh_m)
        rh_masks.append(rh_m)
        n_sig.append(int(lh_m.sum()) + int(rh_m.sum()))

    return np.stack(lh_masks), np.stack(rh_masks), cluster_ids, n_sig


def load_model_masks(contrast_dir: Path, mode: str,
                   top_pct: float = 1.0, fdr_q: float = 0.05) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """
    Pre-load binary vertex masks for every cluster found in contrast_dir.

    Parameters
    ----------
    mode : {'topx', 'fdr'}
    top_pct : used when mode == 'topx'
    fdr_q, n_subjects : used when mode == 'fdr'

    Returns
    -------
    lh_masks : ndarray, shape (n_clusters, n_vertices_lh)
    rh_masks : ndarray, shape (n_clusters, n_vertices_rh)
    cluster_ids : list of int
    n_sig_per_cluster : list of int  — number of vertices in the mask per cluster
    """
    model_files = sorted(glob(f'{contrast_dir.as_posix()}/*/cluster_*_t_map.npy'))
    if not model_files:
        raise FileNotFoundError(f'No t-stat files found in {contrast_dir}')

    cluster_ids = [int(os.path.basename(f).split('_')[1]) for f in model_files]

    model_masks, n_sig = [], []
    for f in model_files:
        cortical_sheet = np.load(f)
        if mode == 'fdr':
            cortical_mask = fdr_mask(cortical_sheet, fdr_q)
        else:
            cortical_mask = model_topx_mask(cortical_sheet, top_pct)
        model_masks.append(cortical_mask)
        n_sig.append(int(cortical_mask.sum()))

    return np.stack(model_masks), cluster_ids, n_sig

# ---------------------------------------------------------------------------
# IoU computation
# ---------------------------------------------------------------------------

def group_iou_from_rows(lh_arr: np.ndarray, rh_arr: np.ndarray,
                         row_indices: np.ndarray) -> float:
    """Compute group IoU for a subset of rows from pre-loaded mask arrays."""
    lh = lh_arr[row_indices]
    rh = rh_arr[row_indices]
    inter = np.all(lh, axis=0).sum() + np.all(rh, axis=0).sum()
    union = np.any(lh, axis=0).sum() + np.any(rh, axis=0).sum()
    return float(inter / union) if union > 0 else 0.0


def model_group_iou_from_rows(model_mask: np.ndarray,
                         row_indices: np.ndarray) -> float:
    """Compute group IoU for a subset of rows from pre-loaded mask arrays."""
    mask = model_mask[row_indices]
    inter = np.all(mask, axis=0).sum() 
    union = np.any(mask, axis=0).sum()
    return float(inter / union) if union > 0 else 0.0



def null_distribution(model_mask,
                       group_size: int, n_iter: int,
                       rng: np.random.Generator,
                       all_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Draw `n_iter` random groups of `group_size` and return their IoUs and
    the cluster IDs used in each iteration.

    Returns
    -------
    ious : ndarray, shape (n_iter,)
    sampled_ids : ndarray, shape (n_iter, group_size)
        Actual cluster IDs (not row indices) drawn in each iteration.
    """
    n_total = model_mask.shape[0]
    id_arr = np.array(all_ids)
    ious = np.empty(n_iter)
    sampled_ids = np.empty((n_iter, group_size), dtype=int)
    for i in range(n_iter):
        idx = rng.choice(n_total, size=group_size, replace=False)
        ious[i] = model_group_iou_from_rows(model_mask, idx)
        sampled_ids[i] = id_arr[idx]
    return ious, sampled_ids


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_null_histogram(observed_iou: float, null_ious: np.ndarray,
                         p_value: float, group_name: str,
                         cluster_ids: list[int], threshold_label: str,
                         output_path: Path) -> None:
    """Save a histogram of the null IoU distribution with the observed value marked."""
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor('white')

    ax.hist(null_ious, bins=50, color='steelblue', edgecolor='white', alpha=0.85,
            label=f'Null (n={len(null_ious):,})')
    ax.axvline(observed_iou, color='crimson', linewidth=2.0,
               label=f'Observed IoU = {observed_iou:.4f}')

    p_str = f'p = {p_value:.4f}' if p_value >= 0.0001 else 'p < 0.0001'
    ax.text(0.97, 0.95, p_str, transform=ax.transAxes,
            ha='right', va='top', fontsize=11, color='crimson')

    ids_str = ', '.join(str(c) for c in cluster_ids)
    ax.set_title(f'Group IoU null distribution — {group_name}\nClusters: {ids_str}',
                 fontsize=11)
    ax.set_xlabel(f'Group IoU ({threshold_label})', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Group IoU of thresholded vertex masks with random null distribution'
    )
    parser.add_argument(
        '--groups', type=str, nargs='+', required=True,
        help='Comma-separated cluster IDs per group, e.g. "3,14,25" "31,32,47"'
    )
    parser.add_argument(
        '--group-names', type=str, nargs='+', default=None,
        help='Labels for each group (default: G1, G2, ...)'
    )
    parser.add_argument(
        '--threshold-mode', choices=['topx', 'fdr'], default='topx',
        help='Thresholding mode: topx (top N%%, default) or fdr (FDR-corrected)'
    )
    parser.add_argument('--top-percentile', type=float, default=1.0,
                        help='Top N%% threshold, used when --threshold-mode topx (default: 1.0)')
    parser.add_argument('--fdr-q', type=float, default=0.05,
                        help='FDR q-level, used when --threshold-mode fdr (default: 0.05)')
    parser.add_argument('--n-subjects', type=int, default=83,
                        help='Number of subjects for df calculation in FDR mode (default: 83)')
    parser.add_argument('--n-iterations', type=int, default=1000,
                        help='Random samples for null distribution (default: 1000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--contrast-dir', type=str, default=str(CONTRAST_DIR))
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_DIR),
                        help='Directory for CSV and histogram PNGs')
    args = parser.parse_args()

    # Parse groups
    groups = []
    for g in args.groups:
        ids = [int(x.strip()) for x in g.split(',')]
        groups.append(ids)

    if args.group_names is not None:
        if len(args.group_names) != len(groups):
            parser.error('--group-names must have the same length as --groups')
        group_names = args.group_names
    else:
        group_names = [f'G{i+1}' for i in range(len(groups))]

    contrast_dir = Path(args.contrast_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = args.threshold_mode
    n_iter = args.n_iterations
    rng = np.random.default_rng(args.seed)

    # Build threshold label for filenames and axis labels.
    # Convention: dots replaced with 'p' (e.g. 0.05 → 0p05) to keep filenames shell-safe.
    if mode == 'fdr':
        q_str = str(args.fdr_q).replace('.', 'p')   # 0.05 → 0p05
        threshold_label = f'FDR q<{args.fdr_q}'
        threshold_tag = f'fdr{q_str}'
    else:
        pct_int = int(args.top_percentile) if args.top_percentile == int(args.top_percentile) else args.top_percentile
        threshold_label = f'Top {args.top_percentile}%'
        threshold_tag = f'top{pct_int}pct'

    file_tag = f'{threshold_tag}_n{args.n_subjects}subj_n{args.n_iterations}iter'

    print(f'Threshold mode : {threshold_label}')
    print(f'Pre-loading masks for all clusters...')
    model_masks, all_ids, n_sig_all = load_model_masks(
        contrast_dir, mode,
        top_pct=args.top_percentile,
        fdr_q=args.fdr_q,
    )
    id_to_row = {cid: i for i, cid in enumerate(all_ids)}

    # Warn about FDR clusters with zero significant vertices
    if mode == 'fdr':
        zero_sig = [cid for cid, n in zip(all_ids, n_sig_all) if n == 0]
        if zero_sig:
            print(f'  WARNING: {len(zero_sig)} clusters have 0 FDR-significant vertices'
                  f' and will contribute empty masks: {zero_sig}')

    print(f'  Loaded {len(all_ids)} clusters ({model_masks.shape[1]})')
    print()

    records = []

    for group_ids, group_name in zip(groups, group_names):
        missing = [c for c in group_ids if c not in id_to_row]
        if missing:
            print(f'[{group_name}] WARNING: cluster IDs not found: {missing}. Skipping.')
            continue

        rows = np.array([id_to_row[c] for c in group_ids])

        # Per-cluster mask sizes for this group
        per_cluster_n = [n_sig_all[r] for r in rows]
        if mode == 'fdr' and any(n == 0 for n in per_cluster_n):
            zero = [cid for cid, n in zip(group_ids, per_cluster_n) if n == 0]
            print(f'[{group_name}] WARNING: clusters with 0 FDR-sig vertices in this group: {zero}')

        observed_iou = model_group_iou_from_rows(model_masks, rows)

        print(f'[{group_name}] Clusters: {group_ids}  (n={len(group_ids)})')
        print(f'  Mask sizes   : {per_cluster_n}')
        print(f'  Observed IoU : {observed_iou:.4f}')

        print(f'  Running {n_iter} random samples (group size = {len(group_ids)})...')
        null_ious, sampled_ids = null_distribution(
            model_masks, len(group_ids), n_iter, rng, all_ids
        )
        null_mean = float(np.mean(null_ious))
        null_std = float(np.std(null_ious))
        p_value = float(np.mean(null_ious >= observed_iou))

        print(f'  Null mean    : {null_mean:.4f}  std: {null_std:.4f}')
        p_str = f'{p_value:.4f}' if p_value >= 0.0001 else '< 0.0001'
        print(f'  p-value      : {p_str}')
        print()

        # Each group gets its own subfolder: {group_name}_{id0}_{id1}_..._<threshold_tag>
        ids_tag = '_'.join(str(c) for c in group_ids)
        group_dir = output_dir / f'{group_name}_{ids_tag}_{threshold_tag}'
        group_dir.mkdir(parents=True, exist_ok=True)

        # Histogram PNG — filename includes threshold tag for disambiguation
        png_path = group_dir / f'iou_null_{group_name}_{file_tag}.png'
        plot_null_histogram(observed_iou, null_ious, p_value,
                            group_name, group_ids, threshold_label, png_path)
        print(f'  Histogram saved: {group_dir.name}/{png_path.name}')

        # Save per-iteration cluster ID selections for reproducibility
        iter_csv_path = group_dir / f'iou_null_iterations_{group_name}_{file_tag}.csv'
        iter_df = pd.DataFrame(
            sampled_ids,
            columns=[f'cluster_id_{k}' for k in range(sampled_ids.shape[1])],
        )
        iter_df.insert(0, 'iteration', np.arange(n_iter))
        iter_df['iou'] = null_ious
        iter_df.to_csv(iter_csv_path, index=False)
        print(f'  Iteration selections saved: {group_dir.name}/{iter_csv_path.name}')
        print()

        record = {
            'group_name': group_name,
            'cluster_ids': ','.join(str(c) for c in group_ids),
            'n_clusters': len(group_ids),
            'threshold_mode': mode,
            'threshold_label': threshold_label,
            'top_percentile': args.top_percentile if mode == 'topx' else None,
            'fdr_q': args.fdr_q if mode == 'fdr' else None,
            'n_subjects': args.n_subjects,
            'observed_iou': round(observed_iou, 6),
            'null_mean': round(null_mean, 6),
            'null_std': round(null_std, 6),
            'p_value': round(p_value, 6),
            'n_iterations': n_iter,
        }
        records.append(record)

        # Per-group summary CSV in the group subfolder
        summary_csv_path = group_dir / f'iou_summary_{group_name}_{file_tag}.csv'
        pd.DataFrame([record]).to_csv(summary_csv_path, index=False)
        print(f'  Summary saved: {group_dir.name}/{summary_csv_path.name}')

    if records:
        df = pd.DataFrame(records)
        print(df.to_string(index=False))


if __name__ == '__main__':
    main()
