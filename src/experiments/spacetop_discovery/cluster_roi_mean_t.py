"""Mean cluster t in each localizer region of the model sheet (the fROI version of the fingerprint).

A simpler alternative to the spatial-correlation fingerprint of cluster_fingerprint.py, in the
form neuroscientists use for fROIs. Each localizer region is the top 10% of units by that
localizer's t over the whole sheet (``--band whole``, the default), or within the part of the
sheet the localizer is restricted to (``--band own``: vision encoder for the visual
localizers, audio encoder for the auditory ones, language/cognitive module for the
text-based ones). The score of a cluster in that region is the mean of the cluster-vs-rest
Welch t over the region's units.

The model's Welch t is computed per unit, so there is no shared per-cluster offset of the
kind the fMRI contrasts carry. The JSON also stores each cluster's mean t over the band as a
check.

    python -m src.experiments.spacetop_discovery.cluster_roi_mean_t \\
        --model_name topo-omni-tiled32-4608 --run_tag disjoint2s_min60 --focus 7,10

Re-plot from a saved JSON without recomputing (needs only numpy and matplotlib):

    python cluster_roi_mean_t.py --replot <stem>.json [--focus 7,10]
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(res, png, focus):
    """Heatmap of res["mean_t"] (clusters x localizers); cell text gives the value."""
    M = np.asarray(res["mean_t"]); cl_ids = res["clusters"]; names = res["localizers"]
    fig, ax = plt.subplots(figsize=(0.95 * len(names) + 2.8, 0.5 * len(cl_ids) + 2.0), dpi=200)
    # colour saturates at the 95th percentile of |t| so one extreme cell does not wash out the rest
    lim = np.nanpercentile(np.abs(M), 95)
    im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=15)
    ax.set_yticks(range(len(cl_ids)))
    ax.set_yticklabels([f"cluster {c}" for c in cl_ids], fontsize=14)
    for c in focus:
        ax.get_yticklabels()[cl_ids.index(c)].set_fontweight("bold")
    for i in range(len(cl_ids)):
        for j in range(len(names)):
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=11.5,
                    color="white" if abs(M[i, j]) > 0.6 * lim else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, extend="both")
    cb.set_label("mean cluster t in region", fontsize=14); cb.ax.tick_params(labelsize=12)
    ax.set_title(f"Mean cluster t in each localizer region of the model sheet\n(region: {res['roi']})",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(png, bbox_inches="tight")
    return png


def main():
    from src.experiments.spacetop_discovery import cluster_null as cn
    from src.experiments.spacetop_discovery.cluster_fingerprint import LOCALIZERS, band_mask, display, top_mask
    from src.experiments.spacetop_discovery.common import add_run_args, read_json, resolve_run_args

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_run_args(ap)
    ap.add_argument("--fwhm", type=float, default=3.0)
    ap.add_argument("--localizer_dir", default=None,
                    help="Default: $SAVE_DIR/<model_name> (holds <cat>/<cat>_fwhm<f>_...pkl)")
    ap.add_argument("--top_pct", type=float, default=10.0, help="region size, %% of the band")
    ap.add_argument("--band", choices=["whole", "own"], default="whole",
                    help="take the top units over the whole sheet or within the localizer's own band")
    ap.add_argument("--focus", default="7,10", help="clusters to print and set in bold")
    args = resolve_run_args(ap.parse_args())

    out = args.output_dir
    loc_dir = args.localizer_dir or os.path.join(os.getenv("SAVE_DIR", "results"), args.model_name)
    clips = read_json(os.path.join(out, "clips_used.json"))
    clusters = {c["cluster_id"]: c["video_ids"]
                for c in read_json(os.path.join(out, "clusters_tvals.json"))}
    cache = os.path.join(out, "cluster_maps", f"smoothed_sheets_fwhm{args.fwhm}.npy")
    if not os.path.exists(cache):
        raise SystemExit(f"{cache} missing: run cluster_maps.py or cluster_null.py first")

    # fast Welch t for every cluster, as in cluster_fingerprint.py
    cn._init(cache)
    X = cn._X
    cn._S1 = np.zeros(X.shape[1]); cn._S2 = np.zeros(X.shape[1])
    for i in range(0, X.shape[0], 256):
        a = np.asarray(X[i:i + 256], np.float64)
        cn._S1 += a.sum(0); cn._S2 += np.einsum("ij,ij->j", a, a)
    pos = {v: k for k, v in enumerate(clips)}
    cl_ids = sorted(clusters)
    cl_maps = {c: display(cn.welch_t(np.array([pos[v] for v in clusters[c]]))) for c in cl_ids}

    names = [n for _, n, _ in LOCALIZERS]
    M = np.zeros((len(cl_ids), len(LOCALIZERS)))
    B = np.zeros_like(M)
    roi_size = {}
    for j, (cat, nm, band) in enumerate(LOCALIZERS):
        with open(os.path.join(loc_dir, cat, f"{cat}_fwhm{args.fwhm}_selectivity_stats.pkl"), "rb") as f:
            lv = display(pickle.load(f)["t"])
        bm = np.ones_like(lv, bool) if args.band == "whole" else band_mask(band)
        roi = top_mask(lv[bm], args.top_pct)
        roi_size[nm] = int(roi.sum())
        for i, c in enumerate(cl_ids):
            cv = cl_maps[c][bm]
            M[i, j] = np.nanmean(cv[roi])
            B[i, j] = np.nanmean(cv)
    # rank 1 = highest mean t of the 16 clusters in that region
    rank = (-M).argsort(0).argsort(0) + 1

    res = dict(clusters=cl_ids, localizers=names, bands=[b for _, _, b in LOCALIZERS],
               roi=f"top {args.top_pct:g}% of " + ("the whole sheet" if args.band == "whole" else "the localizer's band")
               + " by localizer t", roi_size=roi_size,
               mean_t=M.tolist(), rank=rank.tolist(), band_mean_t=B.tolist())
    stem = os.path.join(out, "cluster_maps", "cluster_roi_mean_t" + ("" if args.band == "whole" else "_ownband"))
    if args.band == "whole":
        # where each region lies on the sheet
        res["roi_band_share"] = {}
        for cat, nm, _ in LOCALIZERS:
            with open(os.path.join(loc_dir, cat, f"{cat}_fwhm{args.fwhm}_selectivity_stats.pkl"), "rb") as f:
                lv = display(pickle.load(f)["t"])
            roi = top_mask(lv.ravel(), args.top_pct).reshape(lv.shape)
            res["roi_band_share"][nm] = {b: float((roi & band_mask(b)).sum() / roi.sum())
                                         for b in ("vision", "audio", "thinker")}
    path = stem + ".json"
    with open(path, "w") as f:
        json.dump(res, f, indent=1)

    focus = [int(x) for x in args.focus.split(",") if x]
    print(f"{'':>12}" + "".join(f"{n:>9}" for n in names))
    for c in focus:
        i = cl_ids.index(c)
        print(f"cl {c:>3} t    " + "".join(f"{M[i, j]:>9.2f}" for j in range(len(names))))
        print(f"cl {c:>3} rank " + "".join(f"{rank[i, j]:>9d}" for j in range(len(names))))
        print(f"cl {c:>3} band " + "".join(f"{B[i, j]:>9.2f}" for j in range(len(names))))
    print(f"max |band mean t| over all clusters: {np.abs(B).max():.2f}")
    for nm, sh in res.get("roi_band_share", {}).items():
        print(f"  {nm:<9} region: " + ", ".join(f"{b} {100 * v:.0f}%" for b, v in sh.items()))

    png = plot(res, stem + ".png", focus)
    print(f"\n-> {path}\n-> {png}")


if __name__ == "__main__":
    if "--replot" in sys.argv:
        ap = argparse.ArgumentParser()
        ap.add_argument("--replot", required=True, help="a cluster_roi_mean_t*.json written by this script")
        ap.add_argument("--focus", default="7,10")
        a = ap.parse_args()
        focus = [int(x) for x in a.focus.split(",") if x]
        print("->", plot(json.load(open(a.replot)), a.replot[:-len(".json")] + ".png", focus))
    else:
        main()
