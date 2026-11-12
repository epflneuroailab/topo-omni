"""Mean response of discovered SpaceTop clusters inside each EMFL localizer ROI.

For each cluster and subject, the mean of the cluster contrast's first-level t-map over the
vertices of each ROI (per hemisphere). Across subjects: mean, 95% CI, and a one-sample t-test
against 0, Holm-corrected over each cluster's ROIs. The group maps are the one-sample t of
these subject maps (checked in brain_roi_subject_test.py).

Clusters whose contrast is positive over most of cortex give positive ROI means almost
everywhere, so each cluster's mean over the whole hemisphere is shown alongside as a
reference level (not tested).

    python -m src.experiments.spacetop_discovery.brain_roi_mean_t \\
        --model_name topo-omni-tiled32-4608 --run_tag disjoint2s_min60 --clusters 7,10 \\
        --fmri_dir results/spacetop_fmri/cluster_contrasts_new16clusters_exclbrokenruns \\
        --roi_dir results/spacetop_fmri/roi_masks_emfl_fsaverage6
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from src.experiments.spacetop_discovery.common import add_run_args, resolve_run_args
from src.experiments.spacetop_discovery.brain_roi_subject_test import HEMIS, ROIS, holm, load_gii


def summarize(x):
    tt = stats.ttest_1samp(x, 0)
    ci = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return dict(mean=float(x.mean()), ci95=[float(x.mean() - ci), float(x.mean() + ci)], t=float(tt.statistic),
                df=len(x) - 1, p=float(tt.pvalue), cohens_dz=float(x.mean() / x.std(ddof=1)),
                n_positive=int((x > 0).sum()), n_subjects=len(x))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_run_args(ap)
    ap.add_argument("--fmri_dir", required=True, help="holds subject_level/cluster-XX/ and group_level/")
    ap.add_argument("--roi_dir", required=True)
    ap.add_argument("--clusters", default="7,10")
    args = resolve_run_args(ap.parse_args())
    clusters = [int(x) for x in args.clusters.split(",") if x]
    out_dir = os.path.join(args.output_dir, "cluster_maps")
    os.makedirs(out_dir, exist_ok=True)

    masks = {(k, h): load_gii(os.path.join(args.roi_dir, f"roi-{k}_space-fsaverage6_hemi-{h}_mask.func.gii")) > 0.5
             for k, _ in ROIS for h in HEMIS}
    cols = [(k, n, h) for k, n in ROIS for h in HEMIS if masks[(k, h)].any()]
    names = [f"{n} {h}" for _, n, h in cols]

    res = {"clusters": {}}
    for c in clusters:
        sdir = os.path.join(args.fmri_dir, "subject_level", f"cluster-{c:02d}")
        subs = sorted(os.path.basename(d) for d in glob.glob(os.path.join(sdir, "sub-*")))
        S = {h: np.stack([load_gii(os.path.join(sdir, s, f"{s}_hemi-{h}_tstat.func.gii")) for s in subs]) for h in HEMIS}
        G = {h: load_gii(os.path.join(args.fmri_dir, "group_level",
                                      f"group_cluster-{c:02d}_space-fsaverage6_hemi-{h}_tstat.func.gii")) for h in HEMIS}
        rois = {}
        for key, name, h in cols:
            m = masks[(key, h)]
            rois[f"{name} {h}"] = dict(summarize(S[h][:, m].mean(1)), n_vertices=int(m.sum()),
                                       group_map_mean_t=float(G[h][m].mean()))
        for (nm, r), pa in zip(rois.items(), holm([r["p"] for r in rois.values()])):
            r["p_holm"] = float(pa)
        ref = {f"Hemisphere {h}": dict(summarize(S[h].mean(1)), group_map_mean_t=float(G[h].mean())) for h in HEMIS}
        res["clusters"][str(c)] = dict(subjects=subs, rois=rois, reference=ref)
    path = os.path.join(out_dir, "brain_roi_mean_t.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)

    def stars(p):
        return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."

    # --- print
    for c in clusters:
        R = res["clusters"][str(c)]
        print(f"\n=== cluster {c}  (mean subject t in ROI; n = {len(R['subjects'])}; Holm over {len(cols)} ROIs)")
        for nm, r in list(R["rois"].items()) + list(R["reference"].items()):
            ph = f"p_holm = {r['p_holm']:.2g} {stars(r['p_holm']):<4}" if "p_holm" in r else "(reference)          "
            print(f"  {nm:<13} {r['mean']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]  t({r['df']}) = {r['t']:+6.2f}"
                  f"  {ph}  dz = {r['cohens_dz']:+.2f}  ({r['n_positive']}/{r['n_subjects']} > 0)"
                  f"  group-map mean t = {r['group_map_mean_t']:.2f}")

    # --- figure
    colors = dict(zip(clusters, ["#d62728", "#2ca02c", "#9467bd", "#1f77b4"]))
    labels = names + [f"Hemisphere {h}" for h in HEMIS]
    x = np.arange(len(labels)) + np.r_[np.zeros(len(names)), np.full(len(HEMIS), 0.6)]
    w = 0.8 / len(clusters)
    fig, ax = plt.subplots(figsize=(8.2, 3.9), dpi=200)
    for k, c in enumerate(clusters):
        R = res["clusters"][str(c)]
        vals = [R["rois"][n] for n in names] + [R["reference"][f"Hemisphere {h}"] for h in HEMIS]
        mu = np.array([v["mean"] for v in vals])
        lo = mu - np.array([v["ci95"][0] for v in vals]); hi = np.array([v["ci95"][1] for v in vals]) - mu
        xs = x + (k - (len(clusters) - 1) / 2) * w
        is_ref = np.r_[np.zeros(len(names), bool), np.ones(len(HEMIS), bool)]
        for sel, extra in ((~is_ref, {}), (is_ref, dict(hatch="//", edgecolor="white"))):
            ax.bar(xs[sel], mu[sel], w, yerr=[lo[sel], hi[sel]], color=colors[c], alpha=0.85, capsize=2,
                   error_kw=dict(lw=0.8), **extra)
        for xi, v, m_, l_, h_, r in zip(xs, vals, mu, lo, hi, is_ref):
            if not r:
                y = m_ + h_ + 0.01 if m_ >= 0 else m_ - l_ - 0.01
                ax.text(xi, y, stars(v["p_holm"]), ha="center", va="bottom" if m_ >= 0 else "top", fontsize=6)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline((x[len(names) - 1] + x[len(names)]) / 2, color="0.6", lw=0.6, ls="--")
    ax.margins(y=0.12)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("mean t in ROI (subject maps)", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    handles = [Patch(facecolor=colors[c], alpha=0.85) for c in clusters] + \
              [Patch(facecolor="0.7", hatch="//", edgecolor="white")]
    ax.legend(handles, [f"cluster {c}" for c in clusters] + ["whole-hemisphere reference"], fontsize=7, frameon=False)
    n_sub = len(res["clusters"][str(clusters[0])]["subjects"])
    ax.set_title(f"Mean t of each cluster contrast inside the EMFL ROIs (n = {n_sub} subjects; mean ± 95% CI)",
                 fontsize=9)
    fig.text(0.5, -0.02, f"One-sample t-test against 0, Holm-corrected over the {len(cols)} ROIs of each cluster; "
             "* p < .05, ** p < .01, *** p < .001. Hatched: mean over the whole hemisphere (reference, not tested).",
             ha="center", va="top", fontsize=6)
    fig.tight_layout()
    png = os.path.join(out_dir, "brain_roi_mean_t.png")
    fig.savefig(png, bbox_inches="tight")
    print(f"\n-> {path}\n-> {png}")


if __name__ == "__main__":
    main()
