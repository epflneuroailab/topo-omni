"""Subject-level test of where a discovered cluster's fMRI response sits relative to the EMFL ROIs.

brain_fingerprint.py ranks each cluster against the run's other clusters, but on the brain
side only clusters 7 and 10 survive FDR, so that reference set is mostly noise. This test
needs no other clusters. For every subject it takes the cluster contrast's first-level
t-map and computes, per ROI and hemisphere,

  d_hemi = mean over the ROI  -  mean over the rest of the hemisphere
  d_vis  = mean over the ROI  -  mean over the rest of high-level visual cortex
           (the union of the five ROIs in that hemisphere)

and tests the mean of d across subjects against 0 (one-sample t-test, two-sided). A positive
d means the cluster drives the ROI more than its surroundings.

Tests fall into two families, each Holm-corrected on its own:
  primary      the hypotheses named in ``--primary``, each tested against the rest of visual
               cortex: cluster 7 in the left VWFA (the model's word-form prediction) and in
               the face ROIs (its interview footage), cluster 10 in the scene ROIs (its
               landscape content). ``--family_scope joint`` corrects them together;
               ``per_cluster`` corrects each cluster's hypotheses separately. Both are kept
               in the JSON (``p_holm_primary_joint`` / ``p_holm_primary_per_cluster``)
  exploratory  every other ROI x hemisphere x cluster x variant
The Holm correction over all tests together is kept in the JSON as ``p_holm_all``.

The group maps are a one-sample t-test across these subject maps; the script checks that
first, so the subject maps are the group analysis's actual inputs.

    python -m src.experiments.spacetop_discovery.brain_roi_subject_test \\
        --model_name topo-omni-tiled32-4608 --run_tag disjoint2s_min60 --clusters 7,10 \\
        --fmri_dir results/spacetop_fmri/cluster_contrasts_new16clusters_exclbrokenruns \\
        --roi_dir results/spacetop_fmri/roi_masks_emfl_fsaverage6
"""
import argparse
import glob
import json
import os

import numpy as np
import nibabel as nib
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.experiments.spacetop_discovery.common import add_run_args, resolve_run_args

ROIS = [("faces", "Faces"), ("bodies", "Bodies"), ("scenes", "Scenes"), ("loc", "LOC"), ("vwfa", "VWFA")]
HEMIS = ("L", "R")


def load_gii(path):
    return np.asarray(nib.load(path).darrays[0].data, np.float64).ravel()


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_run_args(ap)
    ap.add_argument("--fmri_dir", required=True, help="holds subject_level/cluster-XX/ and group_level/")
    ap.add_argument("--roi_dir", required=True)
    ap.add_argument("--clusters", default="7,10")
    ap.add_argument("--primary", default="7:VWFA L,7:Faces L,7:Faces R,10:Scenes L,10:Scenes R",
                    help="cluster:ROI hemi pairs tested against the rest of visual cortex as the primary family")
    ap.add_argument("--family_scope", choices=("joint", "per_cluster"), default="per_cluster",
                    help="Holm over all primary tests together, or over each cluster's primary tests")
    args = resolve_run_args(ap.parse_args())
    clusters = [int(x) for x in args.clusters.split(",") if x]
    out_dir = os.path.join(args.output_dir, "cluster_maps")
    os.makedirs(out_dir, exist_ok=True)

    masks = {(k, h): load_gii(os.path.join(args.roi_dir, f"roi-{k}_space-fsaverage6_hemi-{h}_mask.func.gii")) > 0.5
             for k, _ in ROIS for h in HEMIS}
    vis = {h: np.any([masks[(k, h)] for k, _ in ROIS], axis=0) for h in HEMIS}
    cols = [(k, n, h) for k, n in ROIS for h in HEMIS if masks[(k, h)].any()]

    res, tests = {"clusters": {}}, []
    for c in clusters:
        sdir = os.path.join(args.fmri_dir, "subject_level", f"cluster-{c:02d}")
        subs = sorted(os.path.basename(d) for d in glob.glob(os.path.join(sdir, "sub-*")))
        S = {h: np.stack([load_gii(os.path.join(sdir, s, f"{s}_hemi-{h}_tstat.func.gii")) for s in subs]) for h in HEMIS}

        # the group map must be the one-sample t of these subject maps
        check = {}
        for h in HEMIS:
            g = load_gii(os.path.join(args.fmri_dir, "group_level",
                                      f"group_cluster-{c:02d}_space-fsaverage6_hemi-{h}_tstat.func.gii"))
            t1 = stats.ttest_1samp(S[h], 0, axis=0).statistic
            check[h] = float(np.corrcoef(g, t1)[0, 1])
            assert np.allclose(g, t1, atol=1e-3), (c, h, "group t is not the one-sample t of the subject maps")
        print(f"cluster {c}: {len(subs)} subjects; group map == one-sample t of subject maps "
              f"(r = {check['L']:.6f} / {check['R']:.6f})")

        rows = {}
        for key, name, h in cols:
            m = masks[(key, h)]
            roi = S[h][:, m].mean(1)
            for variant, rest_mask in (("hemi", ~m), ("visual", vis[h] & ~m)):
                d = roi - S[h][:, rest_mask].mean(1)
                tt = stats.ttest_1samp(d, 0)
                se = d.std(ddof=1) / np.sqrt(len(d))
                ci = stats.t.ppf(0.975, len(d) - 1) * se
                rows[f"{name} {h}|{variant}"] = dict(
                    roi=name, hemi=h, variant=variant, mean_diff=float(d.mean()),
                    ci95=[float(d.mean() - ci), float(d.mean() + ci)], t=float(tt.statistic),
                    df=len(d) - 1, p=float(tt.pvalue), cohens_dz=float(d.mean() / d.std(ddof=1)),
                    n_positive=int((d > 0).sum()), n_subjects=len(d), roi_mean_t=float(roi.mean()))
                tests.append((c, f"{name} {h}|{variant}"))
        res["clusters"][str(c)] = dict(subjects=subs, group_check_r=check, tests=rows)

    primary = {(int(p.split(":")[0]), p.split(":")[1].strip() + "|visual") for p in args.primary.split(",") if p}
    for c, k in primary:
        assert (c, k) in tests, (c, k, "primary test not among the tests run")
    fam = {"primary": [t for t in tests if t in primary], "exploratory": [t for t in tests if t not in primary]}
    P = lambda c, k: res["clusters"][str(c)]["tests"][k]
    for (c, k), pa in zip(fam["exploratory"], holm([P(c, k)["p"] for c, k in fam["exploratory"]])):
        P(c, k).update(family="exploratory", p_holm=float(pa))
    for (c, k), pa in zip(fam["primary"], holm([P(c, k)["p"] for c, k in fam["primary"]])):
        P(c, k).update(family="primary", p_holm_primary_joint=float(pa))
    for cc in clusters:
        mem = [t for t in fam["primary"] if t[0] == cc]
        for (c, k), pa in zip(mem, holm([P(c, k)["p"] for c, k in mem])):
            P(c, k)["p_holm_primary_per_cluster"] = float(pa)
    for c, k in fam["primary"]:
        P(c, k)["p_holm"] = P(c, k)[f"p_holm_primary_{args.family_scope}"]
    res["family_scope"] = args.family_scope
    for (c, k), pa in zip(tests, holm([res["clusters"][str(c)]["tests"][k]["p"] for c, k in tests])):
        res["clusters"][str(c)]["tests"][k]["p_holm_all"] = float(pa)
    res["families"] = {k: [f"cluster {c}: {t}" for c, t in v] for k, v in fam.items()}
    path = os.path.join(out_dir, "brain_roi_subject_test.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)

    def stars(p):
        return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""

    # --- print
    print(f"\nprimary family ({len(fam['primary'])} tests; figure uses Holm {args.family_scope}):")
    for c, k in fam["primary"]:
        r = res["clusters"][str(c)]["tests"][k]
        print(f"  cluster {c:>2} {k.split('|')[0]:<9} diff {r['mean_diff']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]"
              f"  t({r['df']}) = {r['t']:+.2f}  p = {r['p']:.2g}  Holm joint {r['p_holm_primary_joint']:.2g}"
              f" / per cluster {r['p_holm_primary_per_cluster']:.2g} {stars(r['p_holm'])}"
              f"  dz = {r['cohens_dz']:+.2f}  ({r['n_positive']}/{r['n_subjects']} > 0)")
    print(f"exploratory family ({len(fam['exploratory'])} tests), surviving Holm within family:")
    for c, k in fam["exploratory"]:
        r = res["clusters"][str(c)]["tests"][k]
        if r["p_holm"] < .05:
            print(f"  cluster {c:>2} {k:<16} diff {r['mean_diff']:+.3f}  p = {r['p']:.2g}  p_holm = {r['p_holm']:.2g}")

    # --- figure: mean difference +- 95% CI; visual-cortex panel first (primary tests live there)
    colors = dict(zip(clusters, ["#d62728", "#2ca02c", "#9467bd", "#1f77b4"]))
    names = [f"{n} {h}" for _, n, h in cols]
    x = np.arange(len(names))
    n_sub = len(res["clusters"][str(clusters[0])]["subjects"])
    panels = (("visual", "(a) ROI minus rest of high-level visual cortex"),
              ("hemi", "(b) ROI minus rest of hemisphere"))

    def draw(ax, variant, label, legend):
        w = 0.8 / len(clusters)
        for k, c in enumerate(clusters):
            R = [res["clusters"][str(c)]["tests"][f"{n}|{variant}"] for n in names]
            mu = np.array([r["mean_diff"] for r in R])
            lo = mu - np.array([r["ci95"][0] for r in R]); hi = np.array([r["ci95"][1] for r in R]) - mu
            xs = x + (k - (len(clusters) - 1) / 2) * w
            prim = [r["family"] == "primary" for r in R]
            ax.bar(xs, mu, w, yerr=[lo, hi], color=colors[c], alpha=0.85, capsize=2, label=f"cluster {c}",
                   edgecolor=["black" if p else "none" for p in prim], linewidth=[1.4 if p else 0 for p in prim],
                   error_kw=dict(lw=0.8))
            for xi, r, m_, l_, h_, p in zip(xs, R, mu, lo, hi, prim):
                s = stars(r["p_holm"])
                if not s and not p:
                    continue
                txt = s if not p else (s or "n.s.")
                style = dict(ha="center", fontsize=7, fontweight="bold" if p else "normal")
                if m_ >= 0:
                    ax.text(xi, m_ + h_ + 0.005, txt, va="bottom", **style)
                else:
                    ax.text(xi, m_ - l_ - 0.005, txt, va="top", **style)
        ax.axhline(0, color="k", lw=0.6)
        ax.margins(y=0.14)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("mean difference (subject t)", fontsize=8)
        ax.set_title(label, fontsize=9)
        ax.tick_params(axis="y", labelsize=7)
        if legend:
            from matplotlib.patches import Patch
            h_ = [Patch(facecolor=colors[c], alpha=0.85, edgecolor="none") for c in clusters]
            l_ = [f"cluster {c}" for c in clusters]
            h_.append(Patch(facecolor="white", edgecolor="black", linewidth=1.4)); l_.append("primary test")
            ax.legend(h_, l_, fontsize=7, frameon=False)

    scope = (f"Holm over all {len(fam['primary'])}" if args.family_scope == "joint"
             else "Holm within each cluster")
    note = (f"n = {n_sub} subjects; bars: mean \u00b1 95% CI. Outlined: primary tests ({scope}); "
            f"other stars: exploratory (Holm over {len(fam['exploratory'])}). "
            f"* p < .05, ** p < .01, *** p < .001")
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.0), dpi=200)
    for i, (ax, (variant, label)) in enumerate(zip(axs, panels)):
        draw(ax, variant, label, legend=(i == 0))
    fig.suptitle("Subject-level ROI test: " + note, fontsize=8)
    fig.tight_layout()
    outs = [os.path.join(out_dir, "brain_roi_subject_test.png")]
    fig.savefig(outs[0], bbox_inches="tight"); plt.close(fig)
    for variant, label in panels:
        fig, ax = plt.subplots(figsize=(5.8, 3.8), dpi=200)
        draw(ax, variant, label.split(" ", 1)[1], legend=True)
        fig.text(0.5, -0.02, note.replace(" CI. ", " CI.\n").replace("; other stars", ";\nother stars"),
                 ha="center", va="top", fontsize=6)
        fig.tight_layout()
        outs.append(os.path.join(out_dir, f"brain_roi_subject_test_{variant}.png"))
        fig.savefig(outs[-1], bbox_inches="tight"); plt.close(fig)
    print("\n-> " + path + "".join(f"\n-> {o}" for o in outs))


if __name__ == "__main__":
    main()
