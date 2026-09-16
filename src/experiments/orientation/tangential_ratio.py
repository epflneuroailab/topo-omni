"""Within-layer tangential correlation decay -- the Lambda-free readout of topography.

WHY THIS METRIC. Smoothness and pinwheel density both depend on Lambda and on the fwhm
applied before measuring, and both were shown to invert or move by 2x under those choices.
This one needs neither: it restricts to SAME-LAYER pairs (removing the depth axis, which
carries a large pre-existing residual-stream gradient) and asks whether correlation falls
with tangential distance. near/far > 1 means it does, i.e. a map.

Reference values measured on the 309x512 strip layout, vision band, all slots:
    random init (untrained)          0.995
    pretrained + identity adaptors   0.998
    topo-omni-frontend-invr          1.046   <- best any real run achieved
    strip-chunk5 ckpt276             1.009
    frozen one-clip ckpt148          1.0065  (deterministic gradient, still no structure)
So the floor is ~0.995-1.00 and nothing has beaten 1.05. Read a run as having produced
tangential topography only if it clears that materially.

Usage (from the repo root, PYTHONPATH=.; set TOPO_SHEET_LAYOUT to match the checkpoint):
    python -m src.experiments.orientation.tangential_ratio \
        --runs topo-omni-final-nosmooth random-init-qwen-nosmooth
    python -m src.experiments.orientation.tangential_ratio --runs my-run --slots 0-16
"""
import argparse
import numpy as np

from src.models.spatial_utils import DEFAULT_SHEET_LAYOUT as S, build_sheet_coordinates

BINS = [(0, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 260)]


def slot_units(coords, layout):
    """(units, slot_index) for the vision half of the encoder band."""
    idx = np.arange(layout.num_units)
    row, col = idx // layout.n_col, idx % layout.n_col
    half = layout.n_col // 2
    vis = (row < layout.encoder_rows) & (col < half)
    u = idx[vis]
    # depth_slot knows how depth is embedded for BOTH strip and tiled layouts
    slot = layout.depth_slot(coords[u]).numpy() if hasattr(layout, "depth_slot") else None
    return u, slot


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", nargs="+", required=True,
                    help="result dir names under results/ (each needs "
                         "orientation_unbiased/cortical_sheets.npy)")
    ap.add_argument("--slots", default="all",
                    help="'all' or 'lo-hi' -- encoder slot range, e.g. 0-16 for the early "
                         "half. Slot 0 is the conv front-end.")
    ap.add_argument("--per-slot", type=int, default=400,
                    help="units sampled per slot (keeps the pairwise matrices small)")
    ap.add_argument("--stride", type=int, default=2, help="use every Nth slot")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bins", default=None,
                    help="comma-separated bin EDGES in px, e.g. 0,4,8,16,32,64,91. "
                         "Default: the strip bins in BINS. A layer's tangential extent is "
                         "layout-specific: on the 64x64 continuous tile the strip far bins "
                         "(>=64 px) are empty, so near/far there is undefined, not 1.0.")
    ap.add_argument("--axis", choices=("col", "2d"), default="col",
                    help="tangential distance. 'col' = |dcol| only, correct for strip, "
                         "where a layer is 5 rows deep and the row axis IS depth. "
                         "'2d' = hypot(drow, dcol), correct for tiled, where both tile "
                         "axes are tangential and 'col' would discard half the extent.")
    args = ap.parse_args()

    global BINS
    if args.bins:
        e = [float(x) for x in args.bins.split(",")]
        BINS = [(e[i], e[i + 1]) for i in range(len(e) - 1)]

    import torch
    coords_t = build_sheet_coordinates(S, seed=42)
    coords = coords_t.numpy().astype(int)
    print(f"layout {S.mode} {S.n_row}x{S.n_col}, {S.num_units:,} units, "
          f"units/layer {S.units_per_layer}")

    u_all, slot_all = slot_units(coords_t, S)
    if args.slots == "all":
        keep = np.arange(S.num_encoder_layers)
    else:
        lo, hi = (int(x) for x in args.slots.split("-"))
        keep = np.arange(lo, hi + 1)
    keep = keep[::args.stride]
    print(f"slots: {args.slots} -> {len(keep)} sampled ({keep.min()}..{keep.max()})\n")

    hdr = " ".join(f"{lo:>3g}-{hi:<3g}" for lo, hi in BINS)
    print(f"{'run':<34} {hdr}   near/far")
    for rt in args.runs:
        X = np.load(f"results/{rt}/orientation_unbiased/cortical_sheets.npy", mmap_mode="r")
        rng = np.random.default_rng(args.seed)
        acc = {b: [] for b in BINS}
        for s in keep:
            u = u_all[slot_all == s]
            if len(u) < 40:
                continue
            if len(u) > args.per_slot:
                u = np.sort(rng.choice(u, args.per_slot, replace=False))
            A = np.asarray(X[:, u], dtype=np.float64)
            A -= A.mean(0, keepdims=True)
            sd = A.std(0, keepdims=True)
            sd[sd == 0] = 1.0
            C = np.abs((A / sd).T @ (A / sd) / A.shape[0])
            dc = coords[u][:, 1][:, None] - coords[u][:, 1][None, :]
            if args.axis == "2d":
                dr = coords[u][:, 0][:, None] - coords[u][:, 0][None, :]
                d = np.hypot(dr, dc)
            else:
                d = np.abs(dc)
            iu = np.triu_indices(len(u), k=1)
            dv, cv = d[iu], C[iu]
            for lo, hi in BINS:
                sel = (dv >= lo) & (dv < hi)
                if sel.any():
                    acc[(lo, hi)].append(cv[sel].mean())
        m = [np.mean(acc[b]) if acc[b] else np.nan for b in BINS]
        ratio = (m[0] / m[-1] if np.isfinite(m[0]) and np.isfinite(m[-1])
                 else float("nan"))
        print(f"{rt:<34} " + " ".join(f"{v:7.4f}" for v in m) + f"   {ratio:.4f}")
    print("\nnear/far > 1 = correlation falls with tangential distance (a map).")
    print("floor is ~0.995-1.00 (untrained); best real run to date 1.046.")


if __name__ == "__main__":
    main()
