"""Model vs brain: mean cluster t inside matched ROIs, correlated across discovered clusters.

For every discovered cluster and each of the five visual localizers both sides share:

  model  mean of the cluster-vs-rest Welch t-map (3 mm-smoothed sheets, as in
         cluster_fingerprint.py) over the localizer's model region: the top 1% of the
         vision band by localizer t, the rule the paper's response profiles use
         (response_profiles.marvi.get_selectivity_mask), here on the full localizer data
  brain  mean of the cluster's group-level t-map over the EMFL ROI mask, both hemispheres
         pooled (VWFA: left only)

then a Pearson correlation across clusters per ROI. The brain contrasts carry a large
per-cluster global offset that ROI means inherit, so a second version subtracts from each brain
ROI mean the cluster's mean over the same hemisphere(s) (both, or the left for VWFA); the model
values are left as they are.

    python -m src.experiments.spacetop_discovery.model_brain_roi_mean_t \\
        --model_name topo-omni-tiled32-4608 --run_tag disjoint2s_min60 --focus 7,10 \\
        --group_dir results/spacetop_fmri/cluster_contrasts_new16clusters_exclbrokenruns/group_level \\
        --roi_dir results/spacetop_fmri/roi_masks_emfl_fsaverage6
"""
import argparse
import json
import os

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.spatial_utils import DEFAULT_SHEET_LAYOUT as SHEET
from src.experiments.spacetop_discovery import cluster_null as cn
from src.experiments.spacetop_discovery.common import add_run_args, read_json, resolve_run_args
from src.experiments.spacetop_discovery.brain_roi_subject_test import HEMIS, load_gii
from src.experiments.response_profiles import marvi

H, W = SHEET.n_row, SHEET.n_col
# get_selectivity_mask reads module globals that marvi.py only sets in its own main()
marvi.H, marvi.W = H, W
THK, MID = SHEET.thinker_rows, SHEET.n_col // 2
# brain ROI file key, display name, model localizer category
PAIRS = [("faces", "Faces", "faces"), ("bodies", "Bodies", "bodies"), ("scenes", "Scenes", "scenes"),
         ("loc", "LOC", "objects"), ("vwfa", "VWFA", "vwfa")]


def display(t_flat):
    return np.rot90(np.asarray(t_flat, np.float64).reshape(W, H), k=1)


def model_roi_means(model_name, out, fwhm=3.0, top_k_pct=1.0, band="vision"):
    """Per cluster: mean cluster-vs-rest t over each localizer's model region, and over the band.

    Cluster maps use the same machinery as cluster_fingerprint.py; model regions are the top
    `top_k_pct`% by localizer t (response_profiles.marvi rule) of the vision band
    (``band="vision"``) or of the whole sheet (``band="whole"``: all three components).
    Returns (cluster ids, [cluster x ROI] means, band means, {ROI: mask}).
    """
    out_dir = os.path.join(out, "cluster_maps")
    clips = read_json(os.path.join(out, "clips_used.json"))
    clusters = {c["cluster_id"]: c["video_ids"] for c in read_json(os.path.join(out, "clusters_tvals.json"))}
    cn._init(os.path.join(out_dir, f"smoothed_sheets_fwhm{fwhm}.npy"))
    X = cn._X
    cn._S1 = np.zeros(X.shape[1]); cn._S2 = np.zeros(X.shape[1])
    for i in range(0, X.shape[0], 256):
        a = np.asarray(X[i:i + 256], np.float64)
        cn._S1 += a.sum(0); cn._S2 += np.einsum("ij,ij->j", a, a)
    pos = {v: k for k, v in enumerate(clips)}
    cl_ids = sorted(clusters)
    cl_maps = {c: display(cn.welch_t(np.array([pos[v] for v in clusters[c]]))) for c in cl_ids}
    region = np.zeros((H, W), bool)
    if band == "vision":
        region[THK:, :MID] = True
    elif band == "whole":
        region[:] = True
    else:
        raise ValueError(band)
    model_masks = {}
    for key, name, cat in PAIRS:
        m, _ = marvi.get_selectivity_mask(model_name, cat, "vision", anatomical_constraint=(band == "vision"),
                                          top_k_pct=top_k_pct, odd_or_even=None, fwhm_mm=fwhm)
        assert m.shape == (H, W) and not (m & ~region).any(), (cat, f"mask outside the {band} band")
        model_masks[name] = m
    M = np.array([[cl_maps[c][model_masks[n]].mean() for _, n, _ in PAIRS] for c in cl_ids])
    model_global = np.array([cl_maps[c][region].mean() for c in cl_ids])
    return cl_ids, M, model_global, model_masks


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_run_args(ap)
    ap.add_argument("--group_dir", required=True)
    ap.add_argument("--roi_dir", required=True)
    ap.add_argument("--fwhm", type=float, default=3.0)
    ap.add_argument("--top_k_pct", type=float, default=1.0)
    ap.add_argument("--model_band", choices=("vision", "whole"), default="vision",
                    help="take the model's top-k units within the vision encoder, or over the whole sheet")
    ap.add_argument("--focus", default="7,10")
    ap.add_argument("--exclude", default="",
                    help="clusters left out of every correlation (drawn hollow); outputs get an _excl suffix")
    args = resolve_run_args(ap.parse_args())
    excl = [int(x) for x in args.exclude.split(",") if x]
    tag = ("_excl-" + "-".join(map(str, excl))) if excl else ""
    if args.top_k_pct != 1.0 or args.model_band != "vision":
        tag = f"_top{args.top_k_pct:g}" + ("_whole" if args.model_band == "whole" else "") + tag
    focus = [int(x) for x in args.focus.split(",") if x]
    out = args.output_dir
    out_dir = os.path.join(out, "cluster_maps")

    cl_ids, M, model_global, model_masks = model_roi_means(args.model_name, out, args.fwhm, args.top_k_pct,
                                                           args.model_band)

    # --- brain: group t-maps and ROI masks
    G = {c: {h: load_gii(os.path.join(args.group_dir, f"group_cluster-{c:02d}_space-fsaverage6_hemi-{h}_tstat.func.gii"))
             for h in HEMIS} for c in cl_ids}
    brain_masks = {(k, h): load_gii(os.path.join(args.roi_dir, f"roi-{k}_space-fsaverage6_hemi-{h}_mask.func.gii")) > 0.5
                   for k, _, _ in PAIRS for h in HEMIS}

    B = np.array([[np.concatenate([G[c][h][brain_masks[(k, h)]] for h in HEMIS]).mean() for k, _, _ in PAIRS]
                  for c in cl_ids])
    B_hemi = {h: np.array([[G[c][h][brain_masks[(k, h)]].mean() if brain_masks[(k, h)].any() else np.nan
                            for k, _, _ in PAIRS] for c in cl_ids]) for h in HEMIS}
    brain_global = np.array([np.concatenate([G[c][h] for h in HEMIS]).mean() for c in cl_ids])
    # the cluster's mean over the hemisphere(s) each ROI covers, and the ROI mean relative to it
    roi_hemis = [[h for h in HEMIS if brain_masks[(k, h)].any()] for k, _, _ in PAIRS]
    Bg = np.array([[np.concatenate([G[c][h] for h in hs]).mean() for hs in roi_hemis] for c in cl_ids])
    Bc = B - Bg

    names = [n for _, n, _ in PAIRS]
    res = dict(clusters=cl_ids, rois=names, model_localizers=[cat for _, _, cat in PAIRS],
               model_roi_n_units={n: int(model_masks[n].sum()) for n in names},
               brain_roi_n_vertices={f"{k} {h}": int(m.sum()) for (k, h), m in brain_masks.items()},
               model_mean_t=M.tolist(), brain_mean_t=B.tolist(),
               brain_mean_t_by_hemi={h: v.tolist() for h, v in B_hemi.items()},
               model_vision_band_mean_t=model_global.tolist(), brain_whole_cortex_mean_t=brain_global.tolist(),
               brain_mean_t_minus_global=Bc.tolist(), brain_global_per_roi=Bg.tolist(),
               pearson={}, leave_one_out={}, pearson_global_removed={}, leave_one_out_global_removed={},
               global_offset_check={})
    keep = np.array([c not in excl for c in cl_ids])
    Mk, Bk, Bck = M[keep], B[keep], Bc[keep]
    res["excluded_clusters"] = excl
    for j, n in enumerate(names):
        r, p = stats.pearsonr(Mk[:, j], Bk[:, j])
        loo = [stats.pearsonr(np.delete(Mk[:, j], i), np.delete(Bk[:, j], i))[0] for i in range(int(keep.sum()))]
        res["pearson"][n] = dict(r=float(r), p=float(p), n=int(keep.sum()))
        res["leave_one_out"][n] = dict(min=float(min(loo)), max=float(max(loo)))
        r2, p2 = stats.pearsonr(Mk[:, j], Bck[:, j])
        loo2 = [stats.pearsonr(np.delete(Mk[:, j], i), np.delete(Bck[:, j], i))[0] for i in range(int(keep.sum()))]
        res["pearson_global_removed"][n] = dict(r=float(r2), p=float(p2), n=int(keep.sum()))
        res["leave_one_out_global_removed"][n] = dict(min=float(min(loo2)), max=float(max(loo2)))
        res["global_offset_check"][n] = dict(
            brain_roi_vs_brain_global_r=float(np.corrcoef(B[:, j], brain_global)[0, 1]),
            model_roi_vs_model_global_r=float(np.corrcoef(M[:, j], model_global)[0, 1]))
    r_g, p_g = stats.pearsonr(model_global, brain_global)
    res["global_vs_global"] = dict(r=float(r_g), p=float(p_g))
    path = os.path.join(out_dir, f"model_brain_roi_mean_t{tag}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)

    # --- print
    print("model ROI sizes (units): " + ", ".join(f"{n} {model_masks[n].sum()}" for n in names))
    for j, n in enumerate(names):
        v = res["pearson"][n]; l = res["leave_one_out"][n]; g = res["global_offset_check"][n]
        print(f"{n:<7} Pearson r = {v['r']:+.2f}, p = {v['p']:.3f} (n = {v['n']}); leave-one-out [{l['min']:+.2f}, "
              f"{l['max']:+.2f}] | brain ROI vs brain global r = {g['brain_roi_vs_brain_global_r']:.2f}; "
              f"model ROI vs model global r = {g['model_roi_vs_model_global_r']:.2f}")
        v2 = res["pearson_global_removed"][n]; l2 = res["leave_one_out_global_removed"][n]
        print(f"{'':<7} global mean removed: r = {v2['r']:+.2f}, p = {v2['p']:.3f}; leave-one-out "
              f"[{l2['min']:+.2f}, {l2['max']:+.2f}]")
    print(f"whole-map means, model vs brain: r = {r_g:+.2f}, p = {p_g:.3f}")
    for c in focus:
        i = cl_ids.index(c)
        print(f"cluster {c}: model " + ", ".join(f"{n} {M[i, j]:+.2f}" for j, n in enumerate(names))
              + " | brain " + ", ".join(f"{n} {B[i, j]:+.2f}" for j, n in enumerate(names)))

    # --- figure: one scatter per ROI; row 1 raw brain ROI means, row 2 with the global mean removed
    hl = np.array([c in focus for c in cl_ids])
    fcol = ["#d62728" if c == focus[0] else "#2ca02c" for c in np.array(cl_ids)[hl]]

    def scatter(ax, x, y, title, xlabel, ylabel):
        ax.scatter(x[~hl & keep], y[~hl & keep], s=16, color="0.45", zorder=3)
        ax.scatter(x[hl], y[hl], s=30, color=fcol, zorder=4, edgecolor="k", linewidth=0.5)
        if (~keep).any():
            ax.scatter(x[~keep], y[~keep], s=18, facecolor="none", edgecolor="0.45", linewidth=0.8, zorder=3)
        for c, xi, yi in zip(cl_ids, x, y):
            ax.annotate(str(c), (xi, yi), xytext=(3, 2), textcoords="offset points", fontsize=5.5,
                        fontweight="bold" if c in focus else "normal")
        k, b0 = np.polyfit(x[keep], y[keep], 1)
        xs = np.linspace(x[keep].min(), x[keep].max(), 10)
        ax.plot(xs, k * xs + b0, color="k", lw=0.8)
        ax.axhline(0, color="0.8", lw=0.5, zorder=0); ax.axvline(0, color="0.8", lw=0.5, zorder=0)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel(xlabel, fontsize=7)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=7)
        ax.tick_params(labelsize=6)

    fig, axs = plt.subplots(2, len(names), figsize=(3.0 * len(names), 6.6), dpi=200)
    for j, n in enumerate(names):
        v = res["pearson"][n]; v2 = res["pearson_global_removed"][n]
        scatter(axs[0, j], M[:, j], B[:, j], f"{n} (model: {PAIRS[j][2]})\nr = {v['r']:+.2f}, p = {v['p']:.2g}",
                "model: mean cluster t in ROI", "brain: mean group t in ROI" if j == 0 else None)
        scatter(axs[1, j], M[:, j], Bc[:, j], f"{n}, brain global mean removed\nr = {v2['r']:+.2f}, p = {v2['p']:.2g}",
                "model: mean cluster t in ROI",
                "brain: mean t in ROI \u2212 mean t over hemisphere(s)" if j == 0 else None)
    fig.suptitle(f"Mean cluster t inside matched ROIs, model vs brain; model ROIs: top {args.top_k_pct:g}% of the "
                 f"{'whole sheet' if args.model_band == 'whole' else 'vision band'} (each point one of {len(cl_ids)} clusters; "
                 f"Pearson r across clusters; red = cluster {focus[0]}, green = cluster {focus[1]}).\n"
                 "Top: raw brain ROI means. Bottom: each brain ROI mean minus the cluster's mean over the same "
                 "hemisphere(s) (both; left only for VWFA). Model values are the same in both rows."
                 + (f"\nHollow: clusters {', '.join(map(str, excl))}, excluded from every r and fit "
                    f"(n = {int(keep.sum())})." if excl else ""), fontsize=8)
    fig.tight_layout()
    png = os.path.join(out_dir, f"model_brain_roi_mean_t{tag}.png")
    fig.savefig(png, bbox_inches="tight")
    print(f"\n-> {path}\n-> {png}")


if __name__ == "__main__":
    main()
