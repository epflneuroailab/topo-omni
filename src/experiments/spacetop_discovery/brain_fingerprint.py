"""Functional fingerprint of discovered SpaceTop clusters on the cortical surface.

The brain-side counterpart of cluster_fingerprint.py. Each cluster's group-level
cluster-vs-rest t-map (fsaverage6, one GIFTI per hemisphere) is scored inside each EMFL
localizer ROI of Marvi et al. (2025), per hemisphere:

  r        spatial Pearson r between the cluster t-map and the ROI mask over the ROI's
           hemisphere (point-biserial). Like the model sheet's spatial r it ignores the map's
           overall level, which matters here: the two clusters that survive FDR are positive
           over most of cortex, so their raw ROI means are high in every ROI
  r_vis    the same r computed only over high-level visual cortex (the union of the five
           ROIs in that hemisphere): which visual ROI a cluster favours, the closer analogue
           of the model's r, which is computed inside the localizer's own band
  mean_t   mean t of the cluster contrast over the ROI's vertices (reported, not ranked on)
  dice     overlap (Dice) of the ROI with the cluster's top-10% vertices by t, over both
           hemispheres; threshold-free, so it exists for every cluster
  cover    share of the ROI inside the cluster's displayed map (the top 10% of the vertices
           significant at FDR q < .05, as in the discovery figure), using the per-hemisphere
           FDR thresholds of the cluster's summary JSON; only clusters with FDR survivors

As on the model sheet, every cluster is z-scored against the run's OTHER clusters, so a
cluster is credited only for a response beyond what any set of clips evokes in that ROI.
The model fingerprint (cluster_fingerprint.json) is then compared cluster by cluster with
the brain one, for the five localizers both sides share (Pearson across clusters, on the
spatial r of each side; bilateral r on the brain side).

    python -m src.experiments.spacetop_discovery.brain_fingerprint \\
        --model_name topo-omni-tiled32-4608 --run_tag disjoint2s_min60 --focus 7,10 \\
        --group_dir results/spacetop_fmri/cluster_contrasts_new16clusters_exclbrokenruns/group_level \\
        --roi_dir results/spacetop_fmri/roi_masks_emfl_fsaverage6

The group maps carry values on every vertex (no medial-wall mask), so the top-10% masks are
taken over all vertices of both hemispheres.
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import nibabel as nib
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.experiments.spacetop_discovery.common import add_run_args, read_json, resolve_run_args

# brain ROI file key, display name, name of the matching model localizer
ROIS = [("faces", "Faces", "Faces"), ("bodies", "Bodies", "Bodies"), ("scenes", "Scenes", "Scenes"),
        ("loc", "LOC", "Objects"), ("vwfa", "VWFA", "Words")]
HEMIS = ("L", "R")


def load_gii(path):
    return np.asarray(nib.load(path).darrays[0].data, np.float64).ravel()


def top_mask(t_by_hemi, pct=10.0, keep=None):
    """Top `pct`% of the vertices by t over both hemispheres (restricted to `keep` if given)."""
    t = np.concatenate([t_by_hemi[h] for h in HEMIS])
    ok = np.isfinite(t) if keep is None else np.concatenate([keep[h] for h in HEMIS]) & np.isfinite(t)
    out = np.zeros(len(t), bool)
    if ok.sum():
        k = max(1, int(round(ok.sum() * pct / 100)))
        idx = np.nonzero(ok)[0]
        out[idx[np.argsort(t[idx])[-k:]]] = True
    n = len(t_by_hemi["L"])
    return {"L": out[:n], "R": out[n:]}


def z_vs_others(M):
    Z = np.full_like(M, np.nan)
    for i in range(M.shape[0]):
        rest = np.delete(M, i, axis=0)
        Z[i] = (M[i] - np.nanmean(rest, 0)) / np.nanstd(rest, 0, ddof=1)
    return Z


def heatmap(ax, Z, T, rows, cols, focus, fmt, title, cbar_label, fig, clip=None):
    lim = np.nanmax(np.abs(Z)) if clip is None else clip
    im = ax.imshow(Z, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"cluster {c}" for c in rows], fontsize=7.5)
    for c in focus:
        if c in rows:
            ax.get_yticklabels()[rows.index(c)].set_fontweight("bold")
    for i in range(len(rows)):
        for j in range(len(cols)):
            if np.isfinite(T[i, j]):
                ax.text(j, i, format(T[i, j], fmt), ha="center", va="center", fontsize=5.5,
                        color="white" if abs(Z[i, j]) > 0.6 * lim else "black")
    fig.colorbar(im, ax=ax, label=cbar_label, fraction=0.05)
    ax.set_title(title, fontsize=8.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_run_args(ap)
    ap.add_argument("--group_dir", required=True,
                    help="holds group_cluster-XX_space-fsaverage6_hemi-{L,R}_tstat.func.gii")
    ap.add_argument("--roi_dir", required=True,
                    help="holds roi-<name>_space-fsaverage6_hemi-{L,R}_mask.func.gii")
    ap.add_argument("--focus", default="7,10")
    args = resolve_run_args(ap.parse_args())
    focus = [int(x) for x in args.focus.split(",") if x]
    out_dir = os.path.join(args.output_dir, "cluster_maps")
    os.makedirs(out_dir, exist_ok=True)

    # --- inputs
    files = glob.glob(os.path.join(args.group_dir, "group_cluster-*_space-fsaverage6_hemi-L_tstat.func.gii"))
    clusters = sorted(int(re.search(r"cluster-(\d+)_", f).group(1)) for f in files)
    if not clusters:
        raise SystemExit(f"no group t-maps in {args.group_dir}")
    tmaps = {c: {h: load_gii(os.path.join(args.group_dir,
                                          f"group_cluster-{c:02d}_space-fsaverage6_hemi-{h}_tstat.func.gii"))
                 for h in HEMIS} for c in clusters}
    masks, cols, col_key = {}, [], []
    for key, name, _ in ROIS:
        for h in HEMIS:
            m = load_gii(os.path.join(args.roi_dir, f"roi-{key}_space-fsaverage6_hemi-{h}_mask.func.gii")) > 0.5
            masks[(key, h)] = m
            if m.any():                       # there is no right-hemisphere VWFA ROI
                cols.append(f"{name} {h}"); col_key.append((key, h))
    summ = {}
    for c in clusters:
        p = os.path.join(args.group_dir, f"group_cluster-{c:02d}_space-fsaverage6_summary.json")
        summ[c] = read_json(p) if os.path.exists(p) else {}
    parcels = {}
    for p in sorted(glob.glob(os.path.join(args.roi_dir, "parcel-*_space-fsaverage6_hemi-*_mask.func.gii"))):
        name, h = re.search(r"parcel-(\w+)_space-fsaverage6_hemi-([LR])_", os.path.basename(p)).groups()
        m = load_gii(p) > 0.5
        if m.any():
            parcels[f"{name.upper()} {h}"] = (h, m)
    n_vert = {h: len(tmaps[clusters[0]][h]) for h in HEMIS}
    for (key, h), m in masks.items():
        assert len(m) == n_vert[h], (key, h, len(m), n_vert[h])

    # --- per hemisphere
    MT = np.full((len(clusters), len(cols)), np.nan)
    DI = np.full_like(MT, np.nan); CO = np.full_like(MT, np.nan); RR = np.full_like(MT, np.nan)
    RV = np.full_like(MT, np.nan)
    vis = {h: np.any([masks[(k, h)] for k, _, _ in ROIS], axis=0) for h in HEMIS}
    n_sig = {}
    for i, c in enumerate(clusters):
        top = top_mask(tmaps[c])
        disp = None
        if summ[c].get("n_sig_total", 0) > 0:
            sig = {h: tmaps[c][h] >= summ[c][f"fdr_t_{h.lower()}h"] for h in HEMIS}
            n_sig[c] = int(sum(v.sum() for v in sig.values()))
            disp = top_mask(tmaps[c], keep=sig)
        for j, (key, h) in enumerate(col_key):
            m = masks[(key, h)]
            MT[i, j] = np.nanmean(tmaps[c][h][m])
            RR[i, j] = np.corrcoef(tmaps[c][h], m.astype(float))[0, 1]
            RV[i, j] = np.corrcoef(tmaps[c][h][vis[h]], m[vis[h]].astype(float))[0, 1]
            DI[i, j] = 2 * (top[h] & m).sum() / (top[h].sum() + m.sum())
            if disp is not None:
                CO[i, j] = (disp[h] & m).sum() / m.sum()
    ZMT = z_vs_others(MT)
    ZDI = z_vs_others(DI)
    ZRR = z_vs_others(RR)
    ZRV = z_vs_others(RV)

    # parcels (the Marvi et al. fROI search spaces that make up each ROI)
    pcols = list(parcels)
    PT = np.array([[np.corrcoef(tmaps[c][h], m.astype(float))[0, 1] for (h, m) in parcels.values()]
                   for c in clusters]) if pcols else np.zeros((len(clusters), 0))
    ZPT = z_vs_others(PT) if pcols else PT

    # --- bilateral, for the comparison with the model
    roi_names = [n for _, n, _ in ROIS]
    BI = np.full((len(clusters), len(ROIS)), np.nan); BV = np.full_like(BI, np.nan)
    for i, c in enumerate(clusters):
        for j, (key, _, _) in enumerate(ROIS):
            t = np.concatenate([tmaps[c][h] for h in HEMIS])
            m = np.concatenate([masks[(key, h)] for h in HEMIS]).astype(float)
            v = np.concatenate([vis[h] for h in HEMIS])
            BI[i, j] = np.corrcoef(t, m)[0, 1]
            BV[i, j] = np.corrcoef(t[v], m[v])[0, 1]
    ZBI = z_vs_others(BI); ZBV = z_vs_others(BV)

    model_path = os.path.join(out_dir, "cluster_fingerprint.json")
    comp = {}
    if os.path.exists(model_path):
        mf = read_json(model_path)
        mr = np.array(mf["r"]); mz = np.array(mf["z_r"])
        mrow = {c: k for k, c in enumerate(mf["clusters"])}
        common = [c for c in clusters if c in mrow]
        MR = np.array([[mr[mrow[c], mf["localizers"].index(loc)] for _, _, loc in ROIS] for c in common])
        MZ = np.array([[mz[mrow[c], mf["localizers"].index(loc)] for _, _, loc in ROIS] for c in common])
        rows_b = [clusters.index(c) for c in common]
        for j, (key, name, loc) in enumerate(ROIS):
            pr, p = stats.pearsonr(MR[:, j], BI[rows_b, j])
            pr_v, p_v = stats.pearsonr(MR[:, j], BV[rows_b, j])
            sr, sp = stats.spearmanr(MR[:, j], BI[rows_b, j])
            sr_v, sp_v = stats.spearmanr(MR[:, j], BV[rows_b, j])
            comp[name] = dict(model_localizer=loc, pearson_r=float(pr), p=float(p),
                              pearson_r_visual=float(pr_v), p_visual=float(p_v),
                              spearman_rho=float(sr), spearman_p=float(sp),
                              spearman_rho_visual=float(sr_v), spearman_p_visual=float(sp_v),
                              n_clusters=len(common))
    else:
        common = []

    res = dict(clusters=clusters, columns=cols, r=RR.tolist(), z_r=ZRR.tolist(),
               r_visual=RV.tolist(), z_r_visual=ZRV.tolist(),
               mean_t=MT.tolist(), z_mean_t=ZMT.tolist(),
               dice_top10=DI.tolist(), z_dice_top10=ZDI.tolist(), cover_displayed=CO.tolist(),
               n_fdr_significant=n_sig,
               n_subjects={c: summ[c].get("n_subjects") for c in clusters},
               df={c: summ[c].get("df") for c in clusters},
               labels={c: summ[c].get("target_id") for c in clusters},
               parcels=dict(columns=pcols, r=PT.tolist(), z_r=ZPT.tolist()),
               bilateral=dict(rois=roi_names, r=BI.tolist(), z_r=ZBI.tolist(),
                              r_visual=BV.tolist(), z_r_visual=ZBV.tolist()),
               roi_n_vertices={f"{k} {h}": int(m.sum()) for (k, h), m in masks.items()},
               model_vs_brain=comp)
    path = os.path.join(out_dir, "brain_fingerprint.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)

    # --- print
    print(f"{'':>12}" + "".join(f"{n:>10}" for n in cols))
    for c in focus:
        if c not in clusters:
            continue
        i = clusters.index(c)
        print(f"cl {c:>3} r    " + "".join(f"{RR[i, j]:>10.3f}" for j in range(len(cols))))
        print(f"cl {c:>3} zr   " + "".join(f"{ZRR[i, j]:>10.1f}" for j in range(len(cols))))
        print(f"cl {c:>3} rvis " + "".join(f"{RV[i, j]:>10.3f}" for j in range(len(cols))))
        print(f"cl {c:>3} zvis " + "".join(f"{ZRV[i, j]:>10.1f}" for j in range(len(cols))))
        print(f"cl {c:>3} t    " + "".join(f"{MT[i, j]:>10.2f}" for j in range(len(cols))))
        print(f"cl {c:>3} dice " + "".join(f"{DI[i, j]:>10.3f}" for j in range(len(cols))))
        print(f"cl {c:>3} zdic " + "".join(f"{ZDI[i, j]:>10.1f}" for j in range(len(cols))))
        if np.isfinite(CO[i]).any():
            print(f"cl {c:>3} cov  " + "".join(f"{CO[i, j]:>10.2f}" for j in range(len(cols))))
        if pcols:
            print(f"cl {c:>3} parcels: " + ", ".join(f"{n} r={PT[i, k]:+.3f} (z={ZPT[i, k]:+.1f})"
                                                   for k, n in enumerate(pcols)))
    for name, v in comp.items():
        print(f"model vs brain, {name:<7} (model {v['model_localizer']}): Pearson r = {v['pearson_r']:+.2f}, "
              f"p = {v['p']:.3f}; within visual ROIs r = {v['pearson_r_visual']:+.2f}, p = {v['p_visual']:.3f}; "
              f"n = {v['n_clusters']}")

    # --- figure 1: per hemisphere
    fig, axs = plt.subplots(1, 2, figsize=(2 * (0.62 * len(cols) + 2), 0.34 * len(clusters) + 1.8), dpi=200)
    heatmap(axs[0], ZRR, RR, clusters, cols, focus, ".2f",
            "(a) Cluster t-map vs ROI mask, over the whole hemisphere\n"
            "(cell text: spatial r; colour: z against the other clusters)",
            "z vs other clusters (clipped at \u00b13)", fig, clip=3)
    heatmap(axs[1], ZRV, RV, clusters, cols, focus, ".2f",
            "(b) The same, within high-level visual cortex (union of the five ROIs)\n"
            "(cell text: spatial r; colour: z against the other clusters)",
            "z vs other clusters (clipped at \u00b13)", fig, clip=3)
    fig.suptitle("Cluster fMRI maps vs EMFL localizer ROIs (Marvi et al., 2025), fsaverage6", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "brain_fingerprint.png"), bbox_inches="tight")
    plt.close(fig)

    # --- figure 2: model next to brain, the five shared localizers
    if common:
        fig, axs = plt.subplots(1, 3, figsize=(15, 0.34 * len(common) + 2.4), dpi=200)
        heatmap(axs[0], MZ, MR, common, [loc for _, _, loc in ROIS], focus, ".2f",
                "Model sheet, within the vision encoder\n(cell text: spatial r with localizer map; colour: z)",
                "z vs other clusters", fig)
        heatmap(axs[1], ZBI[rows_b], BI[rows_b], common,
                [f"{n}\nr={comp[n]['pearson_r']:+.2f}" for n in roi_names], focus, ".2f",
                "Human cortex, whole hemispheres\n(cell text: spatial r with ROI mask; colour: z)",
                "z vs other clusters (clipped at ±3)", fig, clip=3)
        heatmap(axs[2], ZBV[rows_b], BV[rows_b], common,
                [f"{n}\nr={comp[n]['pearson_r_visual']:+.2f}" for n in roi_names], focus, ".2f",
                "Human cortex, within high-level visual cortex\n(cell text: spatial r with ROI mask; colour: z)",
                "z vs other clusters (clipped at ±3)", fig, clip=3)
        fig.suptitle("Model vs brain fingerprint (r under each ROI: Pearson across clusters)", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "model_vs_brain_fingerprint.png"), bbox_inches="tight")
        plt.close(fig)
    print(f"\n-> {path}\n-> {os.path.join(out_dir, 'brain_fingerprint.png')}")


if __name__ == "__main__":
    main()
