"""Subject-level version of model_brain_roi_mean_t.py.

For every subject: the mean of each cluster's first-level t-map over each EMFL ROI (both
hemispheres pooled; VWFA left only), raw and with the subject's own mean over the same
hemisphere(s) removed. These 16 cluster values are correlated (Pearson) with the model's
cluster means in the matched model region (same values as the group-level analysis), giving
one r per subject, ROI and version. Inference is across subjects: Fisher z, one-sample t-test
against 0 (two-sided), Holm over the five ROIs within each version. Mean r and its 95% CI are
back-transformed from z.

    python -m src.experiments.spacetop_discovery.model_brain_roi_mean_t_subject \\
        --model_name topo-omni-tiled32-4608 --run_tag disjoint2s_min60 \\
        --fmri_dir results/spacetop_fmri/cluster_contrasts_new16clusters_exclbrokenruns \\
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

from src.experiments.spacetop_discovery.common import add_run_args, resolve_run_args
from src.experiments.spacetop_discovery.brain_roi_subject_test import HEMIS, holm, load_gii
from src.experiments.spacetop_discovery.model_brain_roi_mean_t import PAIRS, model_roi_means

VERSIONS = (("raw", "Raw brain ROI means"),
            ("global_removed", "Brain ROI mean minus the subject's mean over the same hemisphere(s)"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_run_args(ap)
    ap.add_argument("--fmri_dir", required=True, help="holds subject_level/cluster-XX/")
    ap.add_argument("--roi_dir", required=True)
    ap.add_argument("--fwhm", type=float, default=3.0)
    ap.add_argument("--top_k_pct", type=float, default=1.0)
    ap.add_argument("--model_band", choices=("vision", "whole"), default="vision",
                    help="take the model's top-k units within the vision encoder, or over the whole sheet")
    ap.add_argument("--exclude", default="", help="clusters left out of every correlation; outputs get an _excl suffix")
    args = resolve_run_args(ap.parse_args())
    excl = [int(x) for x in args.exclude.split(",") if x]
    tag = ("_excl-" + "-".join(map(str, excl))) if excl else ""
    if args.top_k_pct != 1.0 or args.model_band != "vision":
        tag = f"_top{args.top_k_pct:g}" + ("_whole" if args.model_band == "whole" else "") + tag
    out_dir = os.path.join(args.output_dir, "cluster_maps")

    cl_ids, M, _, _ = model_roi_means(args.model_name, args.output_dir, args.fwhm, args.top_k_pct, args.model_band)
    keep = [i for i, c in enumerate(cl_ids) if c not in excl]
    cl_ids, M = [cl_ids[i] for i in keep], M[keep]
    names = [n for _, n, _ in PAIRS]

    masks = {(k, h): load_gii(os.path.join(args.roi_dir, f"roi-{k}_space-fsaverage6_hemi-{h}_mask.func.gii")) > 0.5
             for k, _, _ in PAIRS for h in HEMIS}
    roi_hemis = [[h for h in HEMIS if masks[(k, h)].any()] for k, _, _ in PAIRS]

    sdir = os.path.join(args.fmri_dir, "subject_level")
    per_cluster = [set(os.listdir(os.path.join(sdir, f"cluster-{c:02d}"))) for c in cl_ids]
    subs = sorted(set.intersection(*per_cluster))
    missing = sorted(set.union(*per_cluster) - set(subs))
    print(f"{len(subs)} subjects with maps for all {len(cl_ids)} clusters"
          + (f" ({len(missing)} dropped: {', '.join(missing)})" if missing else ""))

    # per subject x cluster x ROI: ROI mean, and mean over the ROI's hemisphere(s)
    R = np.zeros((len(subs), len(cl_ids), len(PAIRS))); Gm = np.zeros_like(R)
    for i, s in enumerate(subs):
        for ci, c in enumerate(cl_ids):
            t = {h: load_gii(os.path.join(sdir, f"cluster-{c:02d}", s, f"{s}_hemi-{h}_tstat.func.gii")) for h in HEMIS}
            for j, (k, _, _) in enumerate(PAIRS):
                hs = roi_hemis[j]
                R[i, ci, j] = np.concatenate([t[h][masks[(k, h)]] for h in hs]).mean()
                Gm[i, ci, j] = np.concatenate([t[h] for h in hs]).mean()
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(subs)} subjects loaded", flush=True)
    brain = {"raw": R, "global_removed": R - Gm}

    res = dict(subjects=subs, clusters=cl_ids, excluded_clusters=excl, rois=names, model_mean_t=M.tolist(),
               versions={})
    for v, _ in VERSIONS:
        r = np.array([[stats.pearsonr(M[:, j], brain[v][i, :, j])[0] for j in range(len(names))]
                      for i in range(len(subs))])
        z = np.arctanh(np.clip(r, -0.999999, 0.999999))
        tt = stats.ttest_1samp(z, 0, axis=0)
        se = z.std(0, ddof=1) / np.sqrt(len(subs)); q = stats.t.ppf(0.975, len(subs) - 1)
        p_h = holm(tt.pvalue)
        res["versions"][v] = {n: dict(mean_r=float(np.tanh(z[:, j].mean())),
                                      ci95=[float(np.tanh(z[:, j].mean() - q * se[j])),
                                            float(np.tanh(z[:, j].mean() + q * se[j]))],
                                      t=float(tt.statistic[j]), df=len(subs) - 1, p=float(tt.pvalue[j]),
                                      p_holm=float(p_h[j]), n_positive=int((r[:, j] > 0).sum()),
                                      subject_r=r[:, j].tolist())
                              for j, n in enumerate(names)}
        mu = brain[v].mean(0)
        ci = stats.t.ppf(0.975, len(subs) - 1) * brain[v].std(0, ddof=1) / np.sqrt(len(subs))
        res["versions"][v]["cluster_means"] = dict(mean=mu.tolist(), ci95_halfwidth=ci.tolist())
    path = os.path.join(out_dir, f"model_brain_roi_mean_t_subject{tag}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)

    def stars(p):
        return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."

    for v, label in VERSIONS:
        print(f"\n{label} (n = {len(subs)} subjects, {len(cl_ids)} clusters; Holm over {len(names)} ROIs)")
        for n in names:
            x = res["versions"][v][n]
            print(f"  {n:<7} mean r = {x['mean_r']:+.3f} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]  "
                  f"t({x['df']}) = {x['t']:+.2f}  p = {x['p']:.2g}  p_holm = {x['p_holm']:.2g} {stars(x['p_holm'])}  "
                  f"({x['n_positive']}/{len(subs)} subjects r > 0)")

    # --- figure: subject r per ROI (points), mean r and 95% CI (black), one row per version
    rng = np.random.default_rng(0)
    fig, axs = plt.subplots(2, 1, figsize=(6.4, 6.4), dpi=200, sharex=True)
    for ax, (v, label) in zip(axs, VERSIONS):
        for j, n in enumerate(names):
            x = res["versions"][v][n]
            r = np.array(x["subject_r"])
            ax.scatter(j + rng.uniform(-0.18, 0.18, len(r)), r, s=6, color="0.6", alpha=0.7, zorder=2)
            ax.errorbar(j, x["mean_r"], yerr=[[x["mean_r"] - x["ci95"][0]], [x["ci95"][1] - x["mean_r"]]],
                        fmt="D", color="k", ms=4, capsize=3, lw=1, zorder=4)
            ax.text(j, 1.02, stars(x["p_holm"]), ha="center", va="bottom", fontsize=7)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylim(-1.05, 1.15)
        ax.set_ylabel(f"subject r (model vs brain,\nacross {len(cl_ids)} clusters)", fontsize=7)
        ax.set_title(label, fontsize=8.5)
        ax.tick_params(labelsize=7)
    axs[1].set_xticks(range(len(names)))
    axs[1].set_xticklabels([f"{n}\n(model: {cat})" for _, n, cat in PAIRS], fontsize=7)
    fig.suptitle(f"Model vs brain mean cluster t in matched ROIs, per subject (n = {len(subs)}); "
                 f"model ROIs: top {args.top_k_pct:g}% of the "
                 f"{'whole sheet' if args.model_band == 'whole' else 'vision band'}.\n"
                 "Grey: one subject's Pearson r across clusters. Black: mean r and 95% CI (Fisher z). "
                 "Holm over 5 ROIs per row."
                 + (f"\nClusters {', '.join(map(str, excl))} excluded ({len(cl_ids)} clusters per correlation)."
                    if excl else ""), fontsize=8)
    fig.tight_layout()
    png = os.path.join(out_dir, f"model_brain_roi_mean_t_subject{tag}.png")
    fig.savefig(png, bbox_inches="tight")
    # --- figure: per ROI, cluster scatter of model mean t vs subject-averaged brain value (global removed)
    v = "global_removed"
    mu = np.array(res["versions"][v]["cluster_means"]["mean"])
    ci = np.array(res["versions"][v]["cluster_means"]["ci95_halfwidth"])
    focus = (7, 10)
    fig, axs = plt.subplots(1, len(names), figsize=(3.0 * len(names), 3.5), dpi=200)
    for j, (ax, n) in enumerate(zip(axs, names)):
        x, y, e = M[:, j], mu[:, j], ci[:, j]
        for ci_, c in enumerate(cl_ids):
            col = "#d62728" if c == focus[0] else "#2ca02c" if c == focus[1] else "0.45"
            ax.errorbar(x[ci_], y[ci_], yerr=e[ci_], fmt="o", ms=4.5 if c in focus else 3.5, color=col,
                        mec="k" if c in focus else col, mew=0.5, elinewidth=0.6, capsize=1.5, zorder=4 if c in focus else 3)
            ax.annotate(str(c), (x[ci_], y[ci_]), xytext=(3, 2), textcoords="offset points", fontsize=5.5,
                        fontweight="bold" if c in focus else "normal")
        k, b0 = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 10)
        ax.plot(xs, k * xs + b0, color="k", lw=0.8)
        ax.axhline(0, color="0.8", lw=0.5, zorder=0); ax.axvline(0, color="0.8", lw=0.5, zorder=0)
        rm, pm = stats.pearsonr(x, y)
        sv = res["versions"][v][n]
        ax.set_title(f"{n} (model: {PAIRS[j][2]})\nr of means = {rm:+.2f} (p = {pm:.2g})\n"
                     f"subject mean r = {sv['mean_r']:+.2f} (p_holm = {sv['p_holm']:.2g})", fontsize=7.5)
        ax.set_xlabel("model: mean cluster t in ROI", fontsize=7)
        if j == 0:
            ax.set_ylabel("brain: ROI mean \u2212 subject's mean over hemisphere(s)\n(mean over subjects \u00b1 95% CI)",
                          fontsize=7)
        ax.tick_params(labelsize=6)
    fig.suptitle(f"Model vs brain per cluster, brain values with each subject's own mean removed (n = {len(subs)} "
                 f"subjects; {len(cl_ids)} clusters; model ROIs: top {args.top_k_pct:g}% of the "
                 f"{'whole sheet' if args.model_band == 'whole' else 'vision band'})."
                 + (f" Excluded: clusters {', '.join(map(str, excl))}." if excl else "")
                 + f"\nRed = cluster {focus[0]}, green = cluster {focus[1]}. 'subject mean r': the per-subject "
                 "correlations of the main figure (Fisher z, Holm over 5 ROIs).", fontsize=8)
    fig.tight_layout()
    png2 = os.path.join(out_dir, f"model_brain_roi_mean_t_subject{tag}_scatter.png")
    fig.savefig(png2, bbox_inches="tight")
    print(f"\n-> {path}\n-> {png}\n-> {png2}")


if __name__ == "__main__":
    main()
