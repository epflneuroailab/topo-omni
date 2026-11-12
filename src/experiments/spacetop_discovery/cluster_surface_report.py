"""Locate the significant vertices of one SpaceTop cluster contrast and plot its group t-map.

Rebuilds the group map from the subject-level t-maps exactly as brain_global_removed_group.py
(``--centre cortex``: each subject's map minus its whole-cortex mean; one-sided one-sample t
across subjects; Benjamini-Hochberg FDR q<0.05 jointly over both hemispheres), then:

  * splits the FDR survivors into connected components on the fsaverage6 white-matter mesh;
  * describes each component: vertices, surface area (white surface), peak t, peak position on
    the fsaverage pial surface (approximately MNI305 mm), Desikan-Killiany (aparc) and Destrieux
    (aparc.a2009s) labels at the peak and by share, and the Yeo 7-network composition;
  * reports overlap with the EMFL search spaces and with the medial wall;
  * plots the t-map on the inflated surface (lateral, medial, ventral; both hemispheres),
    thresholded at FDR and unthresholded.

    python cluster_surface_report.py --data <cluster_contrasts dir> --fsavg <fsaverage6 dir> \\
        --rois <roi_masks_emfl_fsaverage6 dir> --cluster 14 --label beatbox --out <dir>
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn import plotting
from scipy import sparse, stats
from scipy.sparse.csgraph import connected_components

HEMIS = ("L", "R")
FS = {"L": "lh", "R": "rh"}


def gii(path):
    return np.asarray(nib.load(path).darrays[0].data, np.float64)


def bh(p, q=0.05):
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    below = np.nonzero(ranked <= q)[0]
    return p <= (p[order][below.max()] if len(below) else -1)


def annot(path):
    labels, ctab, names = nib.freesurfer.read_annot(path)
    names = [n.decode() if isinstance(n, bytes) else n for n in names]
    return np.array([names[i] if i >= 0 else "unknown" for i in labels])


def vertex_area(coords, faces):
    a, b, c = coords[faces[:, 0]], coords[faces[:, 1]], coords[faces[:, 2]]
    tri = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    area = np.zeros(len(coords))
    for k in range(3):
        np.add.at(area, faces[:, k], tri / 3)
    return area


def group_t(X, centre):
    if centre == "cortex":
        X = X - np.nanmean(X, 1, keepdims=True)
    t, p2 = stats.ttest_1samp(X, 0.0, axis=0, nan_policy="omit")
    t = np.asarray(t, np.float64)
    p = np.where(t > 0, p2 / 2, 1 - p2 / 2)
    return t, bh(np.nan_to_num(p, nan=1.0))


def share(names, mask, k=4):
    u, c = np.unique(names[mask], return_counts=True)
    o = np.argsort(c)[::-1][:k]
    return [(str(u[i]), round(100 * c[i] / mask.sum())) for i in o]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--fsavg", required=True)
    ap.add_argument("--rois", required=True)
    ap.add_argument("--cluster", type=int, required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--min_size", type=int, default=10, help="smallest component to list")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cd = os.path.join(args.data, "subject_level", f"cluster-{args.cluster:02d}")
    subs = sorted(os.listdir(cd))
    X = {h: np.stack([gii(os.path.join(cd, s, f"{s}_hemi-{h}_tstat.func.gii")) for s in subs]) for h in HEMIS}
    nL = X["L"].shape[1]
    Xall = np.concatenate([X["L"], X["R"]], 1)
    t_c, sig_c = group_t(Xall, "cortex")
    t_n, sig_n = group_t(Xall, "none")
    tthr = float(t_c[sig_c].min())
    print(f"cluster {args.cluster} ({args.label}): n={len(subs)}; centred FDR survivors {sig_c.sum()} "
          f"(L {sig_c[:nL].sum()}, R {sig_c[nL:].sum()}), t threshold {tthr:.2f}; uncentred survivors {sig_n.sum()}")

    surf, res = {}, dict(cluster=args.cluster, label=args.label, n_subjects=len(subs), centre="cortex",
                         fdr_t_threshold=tthr, n_fdr=int(sig_c.sum()), components=[], hemis={})
    for hi, h in enumerate(HEMIS):
        sl = slice(0, nL) if h == "L" else slice(nL, None)
        t, sig = t_c[sl], sig_c[sl]
        wc, wf = nib.freesurfer.read_geometry(os.path.join(args.fsavg, "surf", f"{FS[h]}.white"))
        pc, _ = nib.freesurfer.read_geometry(os.path.join(args.fsavg, "surf", f"{FS[h]}.pial"))
        surf[h] = wf
        dk = annot(os.path.join(args.fsavg, "label", f"{FS[h]}.aparc.annot"))
        ds = annot(os.path.join(args.fsavg, "label", f"{FS[h]}.aparc.a2009s.annot"))
        yeo = annot(os.path.join(args.fsavg, "label", f"{FS[h]}.Yeo2011_7Networks_N1000.annot"))
        mw = np.zeros(len(t), bool)
        mw[nib.freesurfer.read_label(os.path.join(args.fsavg, "label", f"{FS[h]}.Medial_wall.label"))] = True
        area = vertex_area(wc, wf)
        # mesh graph restricted to significant vertices
        e = np.concatenate([wf[:, [0, 1]], wf[:, [1, 2]], wf[:, [2, 0]]])
        e = e[sig[e[:, 0]] & sig[e[:, 1]]]
        A = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(t), len(t)))
        _, comp = connected_components(A, directed=False)
        res["hemis"][h] = dict(n_fdr=int(sig.sum()), area_mm2=float(area[sig].sum()),
                               in_medial_wall=int((sig & mw).sum()))
        ids, counts = np.unique(comp[sig], return_counts=True)
        for cid in ids[np.argsort(counts)[::-1]]:
            m = sig & (comp == cid)
            if m.sum() < args.min_size:
                continue
            pk = np.nonzero(m)[0][np.argmax(t[m])]
            res["components"].append(dict(
                hemi=h, n_vertices=int(m.sum()), area_mm2=round(float(area[m].sum()), 1),
                peak_t=round(float(t[pk]), 2), peak_xyz=[round(float(v), 1) for v in pc[pk]],
                peak_desikan=str(dk[pk]), peak_destrieux=str(ds[pk]),
                desikan=share(dk, m), destrieux=share(ds, m, 5), yeo7=share(yeo, m, 3),
                medial_wall_pct=round(100 * float((m & mw).sum() / m.sum()))))
        n_small = int(sum(c for c in counts if c < args.min_size))
        res["hemis"][h]["vertices_in_components_below_min"] = n_small

    # EMFL search spaces
    roi = {}
    for f in sorted(os.listdir(args.rois)):
        if f.startswith("roi-") and f.endswith("_mask.func.gii") and "hemi-L" in f:
            name = f.split("_")[0][4:]
            mL = gii(os.path.join(args.rois, f)) > 0
            fr = os.path.join(args.rois, f.replace("hemi-L", "hemi-R"))
            mR = gii(fr) > 0 if os.path.exists(fr) else np.zeros(nL, bool)
            m = np.concatenate([mL, mR])
            roi[name] = dict(roi_vertices=int(m.sum()), fdr_in_roi=int((m & sig_c).sum()),
                             mean_centred_t=round(float(np.nanmean(t_c[m])), 2))
    res["emfl_overlap"] = roi
    with open(os.path.join(args.out, f"cluster-{args.cluster:02d}_surface_report.json"), "w") as f:
        json.dump(res, f, indent=1)

    print(f"{'#':>2} {'hemi':<4} {'verts':>5} {'mm2':>7} {'peak t':>6}  peak xyz (fsaverage)   peak (Desikan / Destrieux)   Desikan share   Yeo-7 share")
    for i, c in enumerate(res["components"], 1):
        print(f"{i:>2} {c['hemi']:<4} {c['n_vertices']:>5} {c['area_mm2']:>7.0f} {c['peak_t']:>6.2f}  "
              f"{str(c['peak_xyz']):<22} {c['peak_desikan']} / {c['peak_destrieux']}   {c['desikan']}   {c['yeo7']}")
    for h, v in res["hemis"].items():
        print(f"hemi {h}: {v}")
    print("EMFL:", roi)

    # ---- figure
    views = [("L", "lateral"), ("L", "medial"), ("L", "ventral"), ("R", "lateral"), ("R", "medial"), ("R", "ventral")]
    vmax = float(np.nanpercentile(np.abs(t_c), 99.9))
    medial = np.concatenate([np.isin(np.arange(nL), nib.freesurfer.read_label(
        os.path.join(args.fsavg, "label", f"{FS[h]}.Medial_wall.label"))) for h in HEMIS])
    # top: the one-sided test's survivors only (t > 0); bottom: the full signed map. Medial wall blanked.
    t_sig = np.where(sig_c & ~medial, t_c, 0.0)
    t_all = np.where(medial, 0.0, t_c)
    fig = plt.figure(figsize=(18, 6.4), dpi=180)
    rows = [(f"FDR q<0.05 survivors (one-sided, t > {tthr:.2f}); each subject's map centred on its cortex mean",
             t_sig, tthr * 0.999, "hot", False),
            ("same centred map, unthresholded (red: above the subject's own mean, blue: below)", t_all, None, "cold_hot", True)]
    for r, (title, tmap, thr, cmap, sym) in enumerate(rows):
        for j, (h, view) in enumerate(views):
            sl = slice(0, nL) if h == "L" else slice(nL, None)
            ax = fig.add_subplot(2, 6, r * 6 + j + 1, projection="3d")
            inf = os.path.join(args.fsavg, "surf", f"{FS[h]}.inflated")
            sulc = nib.freesurfer.read_morph_data(os.path.join(args.fsavg, "surf", f"{FS[h]}.sulc"))
            kw = dict(vmax=vmax, symmetric_cbar=True) if sym else dict(vmin=thr, vmax=vmax, symmetric_cbar=False)
            plotting.plot_surf_stat_map(inf, tmap[sl], hemi="left" if h == "L" else "right", view=view,
                                        bg_map=sulc, threshold=thr, cmap=cmap, colorbar=(j == 5),
                                        axes=ax, figure=fig, engine="matplotlib", **kw)
            ax.set_title(f"{h} {view}", fontsize=9)
        fig.text(0.5, 0.97 - r * 0.49, title, ha="center", fontsize=10)
    fig.suptitle(f"SpaceTop cluster {args.cluster} ({args.label}) vs all other clusters, group t-map (n={len(subs)})",
                 y=1.03, fontsize=12)
    png = os.path.join(args.out, f"cluster-{args.cluster:02d}_tmap.png")
    fig.savefig(png, bbox_inches="tight")
    print("->", png)


if __name__ == "__main__":
    main()
