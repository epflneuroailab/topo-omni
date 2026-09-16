"""Nearest-neighbour |delta theta| within a layer, real vs position-shuffled.

Measured on the RAW map: smoothing manufactures local continuity, so any fwhm > 0 makes
this number look good on a structureless sheet (measured: fwhm=2 takes a random sheet from
45 to 17.5 deg). Run it on a '-nosmooth' result.

Orientation is axial, so |dtheta| wraps at 180 and an unstructured pair averages 45 deg.
The shuffle permutes preferred orientation among the valid units OF THE SAME LAYER, so it
controls for the layer's own tuning distribution and tests only tangential arrangement.

Works for both layouts: under 'tiled' a layer is a 64x64 tile, under 'strip' it is
rows_per_encoder_layer x (n_col // 2). Both axes inside a layer are tangential in either
case -- depth is what separates layers, not what runs across one.

    python -m src.experiments.orientation.nn_delta --runs A B C
"""
import argparse
import numpy as np

from src.models.spatial_utils import tile_grid_pos

from src.models.spatial_utils import DEFAULT_SHEET_LAYOUT as S


def slot_tile(flat, layout, slot):
    """One vision-encoder slot as a 2-D array, for either layout.

    The smoother rasterises by (y=coords[:,1], x=coords[:,0]), so a per-unit vector
    reshapes to (n_col, n_row) and sheet position (row, col) lives at [col, row].
    """
    grid = np.asarray(flat, dtype=float).reshape(layout.n_col, layout.n_row)
    if layout.mode == "tiled":
        eth, etw, _, enc_n = layout.enc_tile
        # tile_grid_pos, NOT divmod: under `tile_order="serpentine"` odd tile-rows run
        # right-to-left, so raster arithmetic silently returns a DIFFERENT layer's tile.
        tr, tc = tile_grid_pos(slot, enc_n, layout.tile_order == "serpentine")
        r0, c0 = tr * eth, tc * etw
        return grid[c0:c0 + etw, r0:r0 + eth].T
    rpl, half = layout.rows_per_encoder_layer, layout.n_col // 2
    return grid[0:half, slot * rpl:(slot + 1) * rpl].T


def slot_window(flat, layout, slot, window=1):
    """`window` CONSECUTIVE slots as one 2-D array, sliding by one slot.

    Why a window rather than a lone slot. Under strip-4096 a slot is 16 x 256 px = 1.6 x
    25.6 Lambda at the nominal Lambda=10, and a pinwheel needs 2-D room -- one slot cannot
    contain one, whatever the model learned. Two consecutive slots give 32 x 256 = 3.2
    Lambda on the short axis.

    It is also closer to what the loss optimises: neighbourhoods are radius-5 discs and a
    layer is only 16 rows deep, so a neighbourhood centred within 5 rows of a boundary
    ALREADY straddles two layers. Analysing a single slot asks a question the objective
    was never posed.

    The cost, stated plainly: the window crosses a depth discontinuity. Adjacent
    transformer layers are similar -- one block's residual update apart -- so it is a far
    milder discontinuity than slot 3 against slot 30, but it is not zero, and a winding
    contour that closes across the seam is not measuring tangential structure alone. The
    shuffle control is applied inside the SAME window, so it inherits the same seam.
    """
    if window == 1:
        return slot_tile(flat, layout, slot)
    if layout.mode != "strip":
        # tiled slots wrap: with 3 tiles across, slots 2 and 3 are not neighbours
        eth, etw, _, enc_n = layout.enc_tile
        serp = layout.tile_order == "serpentine"
        # The window must be `window` CONSECUTIVE layers that are also contiguous on the
        # sheet. Under serpentine an odd tile-row runs right-to-left, so the run of tiles
        # starting at `slot` extends LEFTWARDS there; take the leftmost of the two ends as
        # the slice origin. Either way the window may not cross a tile-row boundary.
        tr, tc = tile_grid_pos(slot, enc_n, serp)
        tr_end, tc_end = tile_grid_pos(slot + window - 1, enc_n, serp)
        if slot + window > layout.num_encoder_layers or tr_end != tr:
            return None
        grid = np.asarray(flat, dtype=float).reshape(layout.n_col, layout.n_row)
        r0, c0 = tr * eth, min(tc, tc_end) * etw
        return grid[c0:c0 + window * etw, r0:r0 + eth].T
    if slot + window > layout.num_encoder_layers:
        return None
    grid = np.asarray(flat, dtype=float).reshape(layout.n_col, layout.n_row)
    rpl, half = layout.rows_per_encoder_layer, layout.n_col // 2
    return grid[0:half, slot * rpl:(slot + window) * rpl].T


def nn_delta(th, ok):
    out = []
    for a, b, m in ((th[:, :-1], th[:, 1:], ok[:, :-1] & ok[:, 1:]),
                    (th[:-1, :], th[1:, :], ok[:-1, :] & ok[1:, :])):
        if m.any():
            d = np.abs(a[m] - b[m]) % 180.0
            out.append(np.minimum(d, 180.0 - d))
    return float(np.concatenate(out).mean()) if out else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-valid", type=int, default=200)
    ap.add_argument("--slots", default="all",
                    help="'all', 'lo-hi' (e.g. 21-32), or 'pXX-100' as a percentage of "
                         "encoder depth where 100 is the DEEPEST slot (e.g. p65-100).")
    ap.add_argument("--window", type=int, default=1,
                    help="consecutive slots per analysis unit, sliding by 1. See "
                         "slot_window: 2 doubles a strip slot's short axis and matches "
                         "the radius-5 neighbourhoods that already cross layer seams.")
    ap.add_argument("--per-slot", action="store_true",
                    help="print the depth profile instead of just the mean")
    args = ap.parse_args()

    n_enc = S.num_encoder_layers
    if args.slots == "all":
        keep = list(range(n_enc))
    elif args.slots.startswith("p"):
        lo_pct, hi_pct = (float(x) for x in args.slots[1:].split("-"))
        lo = int(round(lo_pct / 100.0 * (n_enc - 1)))
        hi = int(round(hi_pct / 100.0 * (n_enc - 1)))
        keep = list(range(lo, hi + 1))
    else:
        lo, hi = (int(x) for x in args.slots.split("-"))
        keep = list(range(lo, hi + 1))
    print(f"layout {S.mode} {S.n_row}x{S.n_col}, a vision slot is "
          f"{slot_tile(np.zeros(S.num_units), S, 0).shape}")
    print(f"slots: {args.slots} -> {keep[0]}..{keep[-1]} ({len(keep)} of {n_enc}), "
          f"window={args.window} slot(s), stride 1")
    print(f"\n{'run':<40} {'real':>7} {'shuf':>7} {'diff':>7} {'slots<':>7}")
    for run in args.runs:
        d = np.load(f"results/{run}/orientation_unbiased/orientation_stats.npz")
        pref = d["pref_ori"]
        gate = d["gate"] & d["vision_band"]
        rng = np.random.default_rng(args.seed)
        R, H, prof = [], [], []
        for s in keep:
            th = slot_window(pref, S, s, args.window)
            if th is None:
                continue
            ok = (slot_window(gate.astype(float), S, s, args.window) > 0.5) & np.isfinite(th)
            if ok.sum() < args.min_valid:
                continue
            sh = th.copy()
            sh[ok] = rng.permutation(th[ok])
            r_s, h_s = nn_delta(th, ok), nn_delta(sh, ok)
            R.append(r_s); H.append(h_s)
            if args.per_slot:
                prof.append((s, r_s, h_s))
        R, H = np.array(R), np.array(H)
        print(f"{run:<40} {R.mean():>7.2f} {H.mean():>7.2f} "
              f"{R.mean() - H.mean():>+7.2f} {(R < H).sum():>4}/{len(R)}")
        if args.per_slot:
            for s_, r_, h_ in prof:
                print(f"    slot {s_:>2} ({100 * s_ / (n_enc - 1):>5.1f}% depth)  "
                      f"real {r_:>6.2f}  shuf {h_:>6.2f}  {r_ - h_:>+6.2f}")
    print("\n45 deg = no tangential clustering. 'slots<' = slots where real beats shuffle.")


if __name__ == "__main__":
    main()
