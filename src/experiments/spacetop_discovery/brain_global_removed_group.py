"""Group maps of the SpaceTop cluster contrasts with each subject's global shift removed.

The per-subject cluster t-maps carry a large shift of the whole cortex (a subject's mean in
any region tracks its mean over the hemisphere), so the published group test largely
detects that shift. Here each subject's map is first centred on its own mean, over the
whole cortex (``--centre cortex``, keeps hemispheric differences) or per hemisphere
(``--centre hemi``). The group test is then the published one: one-sided one-sample t
across subjects at every fsaverage6 vertex, Benjamini-Hochberg FDR q<0.05 jointly over both
hemispheres. ``--centre none`` reproduces the published maps.

Writes ``<out>/group_cluster-XX_centre-<c>_hemi-{L,R}_tstat.func.gii`` and a summary JSON.

    python -m src.experiments.spacetop_discovery.brain_global_removed_group \\
        --data results/spacetop_fmri/cluster_contrasts_new16clusters_exclbrokenruns --centre cortex
"""
import argparse
import json
import os

import nibabel as nib
import numpy as np
from scipy import stats


def load(path):
    return np.asarray(nib.load(path).darrays[0].data, np.float64)


def bh(p, q=0.05):
    """Benjamini-Hochberg: boolean mask of discoveries and the p threshold."""
    p = np.asarray(p)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    below = np.nonzero(ranked <= q)[0]
    if len(below) == 0:
        return np.zeros(len(p), bool), 0.0
    thr = p[order][below.max()]
    return p <= thr, float(thr)


def save_gii(arr, path):
    nib.save(nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(arr.astype(np.float32))]), path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True, help="dir with subject_level/ and group_level/")
    ap.add_argument("--centre", choices=["cortex", "hemi", "none"], default="cortex")
    ap.add_argument("--clusters", default="", help="comma list; default all in subject_level/")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sub_dir = os.path.join(args.data, "subject_level")
    out = args.out or os.path.join(args.data, f"group_level_centre-{args.centre}")
    os.makedirs(out, exist_ok=True)
    cl_dirs = sorted(d for d in os.listdir(sub_dir) if d.startswith("cluster-"))
    if args.clusters:
        keep = {f"cluster-{int(c):02d}" for c in args.clusters.split(",")}
        cl_dirs = [d for d in cl_dirs if d in keep]

    summary = {}
    for cd in cl_dirs:
        subs = sorted(os.listdir(os.path.join(sub_dir, cd)))
        L = np.stack([load(os.path.join(sub_dir, cd, s, f"{s}_hemi-L_tstat.func.gii")) for s in subs])
        R = np.stack([load(os.path.join(sub_dir, cd, s, f"{s}_hemi-R_tstat.func.gii")) for s in subs])
        nL = L.shape[1]
        X = np.concatenate([L, R], 1)                      # subjects x vertices
        X = np.where(np.isfinite(X), X, np.nan)
        offset = np.nanmean(X, 1)                          # each subject's whole-cortex mean
        if args.centre == "cortex":
            X = X - offset[:, None]
        elif args.centre == "hemi":
            X[:, :nL] -= np.nanmean(X[:, :nL], 1, keepdims=True)
            X[:, nL:] -= np.nanmean(X[:, nL:], 1, keepdims=True)
        t, p2 = stats.ttest_1samp(X, 0.0, axis=0, nan_policy="omit")
        t = np.asarray(t, np.float64)
        p = np.where(t > 0, p2 / 2, 1 - p2 / 2)            # one-sided, t > 0
        sig, pthr = bh(np.nan_to_num(p, nan=1.0))
        tthr = float(t[sig].min()) if sig.any() else float("inf")
        # top 10% of the survivors by t, jointly over hemispheres (the display rule)
        top = np.zeros_like(sig)
        if sig.any():
            k = max(1, int(round(0.10 * sig.sum())))
            idx = np.nonzero(sig)[0]
            top[idx[np.argsort(t[idx])[-k:]]] = True
        cid = int(cd.split("-")[1])
        for h, sl in (("L", slice(0, nL)), ("R", slice(nL, None))):
            save_gii(t[sl], os.path.join(out, f"group_cluster-{cid:02d}_centre-{args.centre}_hemi-{h}_tstat.func.gii"))
            save_gii(np.where(sig[sl], t[sl], 0.0), os.path.join(out, f"group_cluster-{cid:02d}_centre-{args.centre}_hemi-{h}_tfdr.func.gii"))
        summary[cid] = dict(
            n_subjects=len(subs), df=len(subs) - 1, max_t=float(np.nanmax(t)),
            fdr_t=tthr, fdr_p=pthr, n_fdr=int(sig.sum()), n_fdr_L=int(sig[:nL].sum()),
            n_fdr_R=int(sig[nL:].sum()), pct_cortex=float(100 * sig.mean()),
            n_top10=int(top.sum()),
            offset_mean=float(np.nanmean(offset)), offset_sd=float(np.nanstd(offset, ddof=1)),
            n_offset_positive=int((offset > 0).sum()),
            offset_t=float(stats.ttest_1samp(offset, 0.0).statistic),
            offset_p_one_sided=float(stats.ttest_1samp(offset, 0.0, alternative="greater").pvalue),
        )
        s = summary[cid]
        print(f"cluster {cid:>2}: n={s['n_subjects']} max t {s['max_t']:5.2f}  FDR survivors {s['n_fdr']:>6} "
              f"({s['pct_cortex']:5.1f}% of cortex; L {s['n_fdr_L']}, R {s['n_fdr_R']})  "
              f"subject offset {s['offset_mean']:+.2f} +- {s['offset_sd']:.2f}, {s['n_offset_positive']}/{s['n_subjects']} > 0, "
              f"p(one-sided) {s['offset_p_one_sided']:.3g}", flush=True)
    path = os.path.join(out, f"summary_centre-{args.centre}.json")
    with open(path, "w") as f:
        json.dump(dict(centre=args.centre, fdr="BH q<0.05 joint over both hemispheres, one-sided",
                       clusters=summary), f, indent=1)
    print("->", path)


if __name__ == "__main__":
    main()
