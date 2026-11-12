"""Functional fingerprint of discovered SpaceTop clusters on the model sheet.

The model sheet has no anatomy to align with cortex, so a cluster's model map and its fMRI
map can only be related through FUNCTION: which localizer-defined regions each one engages.
This scores every cluster's map against every localizer map from the selectivity sweep (the
3 mm-smoothed stats the category maps draw), inside that localizer's own band:

  r      spatial Pearson r between the cluster t-map and the localizer t-map
  dice   overlap of the two maps' top-10% masks (top-10% of the band)

Spatial autocorrelation inflates both, so each cluster is z-scored against the run's OTHER
clusters: a cluster is credited only for overlap beyond what any set of clips shows there.

For the clusters named in ``--islands_for`` it also reports, per highlighted island (from the
``*_island_null.json`` of cluster_null.py, else the largest islands), the share of the island
inside each same-band localizer's top-10% region.

    python -m src.experiments.spacetop_discovery.cluster_fingerprint \\
        --model_name topo-omni-tiled32-4608 --run_tag disjoint2s_min60 --islands_for 7,10
"""
import argparse
import json
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.spatial_utils import DEFAULT_SHEET_LAYOUT as SHEET
from src.experiments.spacetop_discovery import cluster_null as cn
from src.experiments.spacetop_discovery.common import add_run_args, read_json, resolve_run_args

H, W = SHEET.n_row, SHEET.n_col
THK, MID = SHEET.thinker_rows, SHEET.n_col // 2
LOCALIZERS = [("faces", "Faces", "vision"), ("bodies", "Bodies", "vision"),
              ("scenes", "Scenes", "vision"), ("objects", "Objects", "vision"),
              ("vwfa", "Words", "vision"), ("speech", "Speech", "audio"),
              ("vocals", "Voices", "audio"), ("language_text_ALL", "Language", "thinker"),
              ("theory_of_mind_text_ALL", "ToM", "thinker"),
              ("multiple_demand_text_ALL", "MD", "thinker")]


def band_mask(band):
    m = np.zeros((H, W), bool)
    if band == "thinker":
        m[:THK] = True
    elif band == "vision":
        m[THK:, :MID] = True
    else:
        m[THK:, MID:] = True
    return m


def display(t_flat):
    return np.rot90(np.asarray(t_flat, np.float64).reshape(W, H), k=1)


def top_mask(vals, pct=10.0):
    k = int(len(vals) * pct / 100)
    m = np.zeros(len(vals), bool)
    m[np.argpartition(np.nan_to_num(vals, nan=-np.inf), -k)[-k:]] = True
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_run_args(ap)
    ap.add_argument("--fwhm", type=float, default=3.0)
    ap.add_argument("--localizer_dir", default=None,
                    help="Default: $SAVE_DIR/<model_name> (holds <cat>/<cat>_fwhm<f>_...pkl)")
    ap.add_argument("--islands_for", default="", help="clusters to break down per island")
    ap.add_argument("--n_islands", type=int, default=5)
    args = resolve_run_args(ap.parse_args())

    out = args.output_dir
    loc_dir = args.localizer_dir or os.path.join(os.getenv("SAVE_DIR", "results"), args.model_name)
    clips = read_json(os.path.join(out, "clips_used.json"))
    clusters = {c["cluster_id"]: c["video_ids"]
                for c in read_json(os.path.join(out, "clusters_tvals.json"))}
    cache = os.path.join(out, "cluster_maps", f"smoothed_sheets_fwhm{args.fwhm}.npy")
    if not os.path.exists(cache):
        raise SystemExit(f"{cache} missing: run cluster_maps.py or cluster_null.py first")

    # fast Welch t for every cluster, from cluster_null's column-sum machinery
    cn._init(cache)
    X = cn._X
    cn._S1 = np.zeros(X.shape[1]); cn._S2 = np.zeros(X.shape[1])
    for i in range(0, X.shape[0], 256):
        a = np.asarray(X[i:i + 256], np.float64)
        cn._S1 += a.sum(0); cn._S2 += np.einsum("ij,ij->j", a, a)
    pos = {v: k for k, v in enumerate(clips)}
    cl_ids = sorted(clusters)
    cl_maps = {c: display(cn.welch_t(np.array([pos[v] for v in clusters[c]]))) for c in cl_ids}

    loc_maps = {}
    for cat, _, _ in LOCALIZERS:
        with open(os.path.join(loc_dir, cat, f"{cat}_fwhm{args.fwhm}_selectivity_stats.pkl"), "rb") as f:
            loc_maps[cat] = display(pickle.load(f)["t"])

    # cluster x localizer scores, within each localizer's band
    R = np.zeros((len(cl_ids), len(LOCALIZERS)))
    D = np.zeros_like(R)
    for j, (cat, _, band) in enumerate(LOCALIZERS):
        bm = band_mask(band)
        lv = loc_maps[cat][bm]
        lm = top_mask(lv)
        for i, c in enumerate(cl_ids):
            cv = cl_maps[c][bm]
            R[i, j] = np.corrcoef(cv, lv)[0, 1]
            cm = top_mask(cv)
            D[i, j] = 2 * (cm & lm).sum() / (cm.sum() + lm.sum())

    def z_vs_others(M):
        Z = np.zeros_like(M)
        for i in range(len(cl_ids)):
            rest = np.delete(M, i, axis=0)
            Z[i] = (M[i] - rest.mean(0)) / rest.std(0, ddof=1)
        return Z

    ZR, ZD = z_vs_others(R), z_vs_others(D)
    names = [n for _, n, _ in LOCALIZERS]

    res = dict(clusters=cl_ids, localizers=names,
               bands=[b for _, _, b in LOCALIZERS], r=R.tolist(), dice=D.tolist(),
               z_r=ZR.tolist(), z_dice=ZD.tolist(), islands={})

    # per-island breakdown
    for c in [int(x) for x in args.islands_for.split(",") if x]:
        name = f"spacetop_cluster_{c}"
        d = os.path.join(out, "cluster_maps", name)
        with open(os.path.join(d, f"{name}_fwhm{args.fwhm}_selectivity_stats.pkl"), "rb") as f:
            st = pickle.load(f)
        labeled, isl = cn.islands(st["t"])
        null_json = os.path.join(d, f"{name}_island_null.json")
        if os.path.exists(null_json):
            order = [r["id"] for r in read_json(null_json)["islands"]]
            ranking = "depth-corrected mass"
        else:
            order = [r["id"] for r in isl]
            ranking = "size"
        by_id = {r["id"]: r for r in isl}
        rows = []
        for iid in order[:args.n_islands]:
            m = labeled == iid
            r0, c0 = by_id[iid]["centroid"]
            band = "thinker" if r0 < THK else ("vision" if c0 < MID else "audio")
            share = {}
            for cat, nm, b in LOCALIZERS:
                if b != band:
                    continue
                lm = np.zeros((H, W), bool)
                bm = band_mask(b)
                lm[bm] = top_mask(loc_maps[cat][bm])
                share[nm] = float((m & lm).sum() / m.sum())
            rows.append(dict(id=int(iid), size=int(m.sum()), band=band,
                             mean_t=float(by_id[iid]["average_t_values"]), share=share))
        res["islands"][str(c)] = dict(ranking=ranking, islands=rows)

    path = os.path.join(out, "cluster_maps", "cluster_fingerprint.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)

    # --- print
    focus = [int(x) for x in args.islands_for.split(",") if x]
    print(f"{'':>10}" + "".join(f"{n:>9}" for n in names))
    for c in focus:
        i = cl_ids.index(c)
        print(f"cl {c:>3} r  " + "".join(f"{R[i, j]:>9.2f}" for j in range(len(names))))
        print(f"cl {c:>3} zr " + "".join(f"{ZR[i, j]:>9.1f}" for j in range(len(names))))
        print(f"cl {c:>3} dice" + "".join(f"{D[i, j]:>9.2f}" for j in range(len(names))))
        print(f"cl {c:>3} zd " + "".join(f"{ZD[i, j]:>9.1f}" for j in range(len(names))))
    for c, v in res["islands"].items():
        print(f"\ncluster {c} islands (ranked by {v['ranking']}):")
        for r in v["islands"]:
            sh = ", ".join(f"{k} {100 * s:.0f}%" for k, s in sorted(r["share"].items(),
                                                                    key=lambda kv: -kv[1]))
            print(f"  id {r['id']:>4}  {r['band']:<7}  n={r['size']:>5}  mean t {r['mean_t']:5.1f}"
                  f"  inside: {sh}")

    # --- figure: z-scored spatial r, clusters x localizers
    fig, ax = plt.subplots(figsize=(0.62 * len(names) + 2, 0.34 * len(cl_ids) + 1.6), dpi=200)
    lim = np.nanmax(np.abs(ZR))
    im = ax.imshow(ZR, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cl_ids)))
    ax.set_yticklabels([f"cluster {c}" for c in cl_ids], fontsize=7.5)
    for c in focus:
        ax.get_yticklabels()[cl_ids.index(c)].set_fontweight("bold")
    for i in range(len(cl_ids)):
        for j in range(len(names)):
            ax.text(j, i, f"{R[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                    color="white" if abs(ZR[i, j]) > 0.6 * lim else "black")
    fig.colorbar(im, ax=ax, label="z vs other clusters (spatial r)", fraction=0.04)
    ax.set_title("Cluster maps vs localizer maps, within each localizer's band\n"
                 "(cell text: spatial r; colour: z against the other clusters)", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "cluster_maps", "cluster_fingerprint.png"), bbox_inches="tight")
    print(f"\n-> {path}\n-> {os.path.join(out, 'cluster_maps', 'cluster_fingerprint.png')}")


if __name__ == "__main__":
    main()
