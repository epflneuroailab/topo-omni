"""Measure Lambda (orientation-column spacing, px) per vision-encoder tile.

Only meaningful under a TILED layout, where a layer occupies a compact 2-D tile and
therefore has enough extent in BOTH axes for a 2-D power spectrum to mean anything.
On the strip layout a layer is 5x256, so this would measure the column axis only.

The smoother rasterises by (y=coords[:,1], x=coords[:,0]), so the saved per-unit
vectors reshape to (n_col, n_row) with sheet position (row, col) at [col, row] --
the same transform `run_orientation.vision_band` applies, spelled for the tiled case.

Usage (repo root, PYTHONPATH=., TOPO_SHEET_LAYOUT=continuous):
    python -m src.experiments.orientation.measure_lambda --run continuous-4096-ckpt69-nosmooth
"""
import argparse
import numpy as np

from src.models.spatial_utils import (DEFAULT_SHEET_LAYOUT as S, MACAQUE_LAMBDA_MM,
                                      tile_grid_pos)
from src.experiments.orientation.metrics import column_spacing


def slot_tile(flat, layout, slot):
    """(tile_rows, tile_cols) view of a vision-encoder slot from a smoothed-sheet vector."""
    eth, etw, enr, enc_n = layout.enc_tile
    grid = np.asarray(flat, dtype=float).reshape(layout.n_col, layout.n_row)
    # tile_grid_pos, NOT divmod -- see nn_delta.slot_tile: serpentine tile order reverses
    # odd tile-rows, and raster arithmetic would read the wrong layer's tile.
    tr, tc = tile_grid_pos(slot, enc_n, layout.tile_order == "serpentine")
    r0, c0 = tr * eth, tc * etw
    return grid[c0:c0 + etw, r0:r0 + eth].T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--min-valid", type=int, default=200,
                    help="skip a tile with fewer than this many gated, tuned units")
    ap.add_argument("--shuffle", type=int, default=0,
                    help="CONTROL: permute preferred orientation among the valid units of "
                         "each tile with this seed, destroying all spatial structure. A "
                         "64x64 tile leaves only ~5 usable radial bins, so the spectrum "
                         "peak has somewhere to land even in noise -- if the shuffled "
                         "Lambda matches the real one, the real one is measuring the "
                         "window and the bin grid, not columns.")
    args = ap.parse_args()

    if S.mode != "tiled":
        raise SystemExit(f"layout is {S.mode!r}; set TOPO_SHEET_LAYOUT=continuous")

    d = np.load(f"results/{args.run}/orientation_unbiased/orientation_stats.npz")
    pref, gate = d["pref_ori"], d["gate"]
    print(f"layout {S.n_row}x{S.n_col} tiled, enc_tile={S.enc_tile}, "
          f"mm_per_px currently {S.mm_per_px}")

    lams, rows = [], []
    for s in range(S.num_encoder_layers):
        theta = slot_tile(pref, S, s)
        valid = slot_tile(gate.astype(float), S, s) > 0.5
        valid &= np.isfinite(theta)
        n = int(valid.sum())
        if n < args.min_valid:
            rows.append((s, n, float("nan")))
            continue
        if args.shuffle:
            rng = np.random.default_rng(args.shuffle + s)
            vals = theta[valid]
            theta = theta.copy()
            theta[valid] = rng.permutation(vals)
        lam, _, _ = column_spacing(theta, valid)
        rows.append((s, n, lam))
        lams.append(lam)

    print("\nslot   n_valid  Lambda_px")
    for s, n, lam in rows:
        print(f"{s:>4} {n:>8} {lam:>10.2f}")

    lams = np.array(lams)
    if not len(lams):
        raise SystemExit("no tile had enough valid units")
    med = float(np.median(lams))
    print(f"\ntiles measured : {len(lams)}/{S.num_encoder_layers}")
    print(f"Lambda median  : {med:.2f} px   (IQR "
          f"{np.percentile(lams, 25):.2f}-{np.percentile(lams, 75):.2f})")
    print(f"mm_per_px      : {MACAQUE_LAMBDA_MM} / {med:.2f} = "
          f"{MACAQUE_LAMBDA_MM / med:.5f}   (currently {S.mm_per_px})")
    print(f"layer extent   : {S.enc_tile[0] * MACAQUE_LAMBDA_MM / med:.2f} mm "
          f"({S.enc_tile[0] / med:.2f} Lambda across)")


if __name__ == "__main__":
    main()
