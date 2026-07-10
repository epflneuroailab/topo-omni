#!/usr/bin/env python3
"""Render the published cluster-contrast surface maps (fsaverage6, top-10%-of-FDR HTML).

Lean port of dev `src/17_visualize_cluster_contrasts_v2.py` (Branch A) / `47` wrapper
(Branch B) (20251211_fMRI_movie_watching_spacetop @ 4066746), reduced to what the paper
shows: the **top-10%-of-FDR-significant** vertices on the inflated fsaverage6 surface,
per hemisphere, viridis colormap on a binarized sulcal background.

Dropped from the dev script (not in any Spacetop paper figure — docs/DESIGN.md §8 / index §7):
Moran's I readout (also the stale duplicate `spatial_stats.py`), the PNG montage, and
the fdr05/fdr01 variants. The FDR + top-percentile threshold math is kept byte-faithful,
so the rendered survivor set matches the published maps.

Colormap note: the published HTML uses **viridis** (verified in the on-disk files);
the index's "hot_r" text is stale. df defaults to n=78 → 77 (the published analysis;
docs/DESIGN.md §7). This is a RENDER step — not golden-mastered (docs/DESIGN.md §6).

Requires nilearn (pinned 0.12.1). Fetches fsaverage6 surfaces via nilearn.datasets.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def binarize_sulcal_map(sulc_data: np.ndarray) -> np.ndarray:
    """Binarize sulcal depth: sulci → 179/255 (gray), gyri → 1.0 (white). Crisp borders."""
    return np.where(sulc_data > 0, 179 / 255, 1.0)


def calculate_fdr_threshold(data_lh, data_rh, q=0.05, df=77):
    """One-tailed (positive) Benjamini-Hochberg FDR t-threshold over both hemispheres.

    Faithful to dev 17_v2.calculate_fdr_threshold. Returns (t_threshold, n_sig, p_thresh).
    """
    all_data = np.concatenate([data_lh, data_rh])
    positive_data = all_data[all_data > 0]
    if len(positive_data) == 0:
        return np.inf, 0, 0

    p_values = 1 - stats.t.cdf(positive_data, df=df)
    sorted_p = np.sort(p_values)
    n_tests = len(sorted_p)
    significant_mask = sorted_p <= (np.arange(1, n_tests + 1) / n_tests) * q

    if np.any(significant_mask):
        fdr_p_threshold = sorted_p[significant_mask][-1]
        fdr_t_threshold = stats.t.ppf(1 - fdr_p_threshold, df=df)
        n_sig = int(np.sum(significant_mask))
        return fdr_t_threshold, n_sig, fdr_p_threshold
    return np.inf, 0, 0


def top_of_fdr_threshold(data_lh, data_rh, top_percentile=10.0, df=77, q=0.05):
    """t-threshold at the top `top_percentile`% of FDR q<0.05 survivors (both hemis)."""
    fdr_t, _, _ = calculate_fdr_threshold(data_lh, data_rh, q=q, df=df)
    if not np.isfinite(fdr_t):
        return np.nan, 0, fdr_t
    all_data = np.concatenate([data_lh, data_rh])
    survivors = all_data[all_data >= fdr_t]
    thr = float(np.percentile(survivors, 100 - top_percentile))
    n = int(np.sum(all_data >= thr))
    return thr, n, fdr_t


def _slug(text):
    """Filesystem-safe label: spaces/plus/slashes -> hyphens."""
    out = str(text)
    for ch in " +/\\":
        out = out.replace(ch, "-")
    return out.strip("-")


def render_cluster(cluster_id, contrast_dir, output_dir, fsaverage,
                   top_percentile=10.0, n_subjects=78, figure_tag=None, label=None):
    """Render top-of-FDR inflated HTML (both hemispheres) for one cluster.

    ``figure_tag`` (e.g. "Fig6-D4") and ``label`` (the author cluster label, e.g. "animals")
    are folded into the output filename so a reviewer can tell which paper figure / cluster a
    map is without opening it: ``<tag>_cluster05_animals_lh_top10pct_inflated.html``.
    """
    from nilearn import plotting

    contrast_dir = Path(contrast_dir)
    lh_file = contrast_dir / f'group_cluster-{cluster_id:02d}_space-fsaverage6_hemi-L_tstat.func.gii'
    rh_file = contrast_dir / f'group_cluster-{cluster_id:02d}_space-fsaverage6_hemi-R_tstat.func.gii'
    if not lh_file.exists() or not rh_file.exists():
        logger.warning(f"  Missing GIFTI for cluster {cluster_id:02d}, skipping")
        return

    lh_data = nib.load(str(lh_file)).darrays[0].data
    rh_data = nib.load(str(rh_file)).darrays[0].data
    df = n_subjects - 1

    thr, n_sig, fdr_t = top_of_fdr_threshold(lh_data, rh_data, top_percentile, df=df)
    if not np.isfinite(thr):
        logger.info(f"  Cluster {cluster_id:02d}: no FDR q<0.05 survivors, skipping")
        return
    logger.info(f"  Cluster {cluster_id:02d}: FDR t>{fdr_t:.2f}; top-{top_percentile:g}% t>{thr:.2f} (n={n_sig})")

    bg = {'lh': binarize_sulcal_map(nib.load(fsaverage['sulc_left']).darrays[0].data),
          'rh': binarize_sulcal_map(nib.load(fsaverage['sulc_right']).darrays[0].data)}
    masked = {'lh': np.where(lh_data >= thr, lh_data, np.nan),
              'rh': np.where(rh_data >= thr, rh_data, np.nan)}
    vmax = float(max(np.nanmax(masked['lh']), np.nanmax(masked['rh'])))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pct_int = int(top_percentile) if top_percentile == int(top_percentile) else top_percentile
    for hemi, surf_key in [('lh', 'infl_left'), ('rh', 'infl_right')]:
        title = f"Cluster {cluster_id:02d} > All Others | Top {top_percentile:g}% of FDR | {hemi.upper()}"
        view = plotting.view_surf(
            surf_mesh=fsaverage[surf_key],
            surf_map=masked[hemi],
            bg_map=bg[hemi],
            threshold=None,
            cmap='viridis',
            symmetric_cmap=False,
            vmin=thr,
            vmax=vmax,
            title=title,
        )
        parts = ([figure_tag] if figure_tag else []) + [f"cluster{cluster_id:02d}"]
        if label:
            parts.append(_slug(label))
        parts += [hemi, f"top{pct_int}pct", "inflated"]
        out = output_dir / ("_".join(parts) + ".html")
        view.save_as_html(str(out))
        logger.info(f"    Saved: {out.name}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--contrast-dir', required=True, help='group_level/ dir with group_cluster-*_tstat GIFTIs.')
    p.add_argument('--output-dir', required=True, help='Where to write the HTML maps.')
    p.add_argument('--cluster-ids', nargs='+', type=int, default=None,
                   help='Cluster IDs to render (default: all found in --contrast-dir).')
    p.add_argument('--top-percentile', type=float, default=10.0, help='Top %% of FDR survivors (default 10).')
    p.add_argument('--n-subjects', type=int, default=78, help='For df = n-1 (default 78 → df 77, the published analysis).')
    p.add_argument('--figure-tag', default=None, help='Paper-figure tag folded into filenames (e.g. Fig6-D4).')
    p.add_argument('--cluster-labels', nargs='*', default=None, metavar='ID=LABEL',
                   help='Author cluster labels folded into filenames (e.g. 5=animals 32="natural landscapes").')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    from nilearn import datasets

    contrast_dir = Path(args.contrast_dir)
    if args.cluster_ids is not None:
        cluster_ids = args.cluster_ids
    else:
        tstat_files = sorted(contrast_dir.glob('group_cluster-*_space-fsaverage6_hemi-L_tstat.func.gii'))
        cluster_ids = [int(f.name.split('_')[1].split('-')[1]) for f in tstat_files]

    labels = {}
    for item in (args.cluster_labels or []):
        if "=" in item:
            k, _, v = item.partition("=")
            labels[int(k)] = v

    logger.info(f"Rendering {len(cluster_ids)} cluster(s) from {contrast_dir}")
    fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage6')
    for cid in cluster_ids:
        render_cluster(cid, contrast_dir, args.output_dir, fsaverage,
                       top_percentile=args.top_percentile, n_subjects=args.n_subjects,
                       figure_tag=args.figure_tag, label=labels.get(cid))
    logger.info("Done.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
