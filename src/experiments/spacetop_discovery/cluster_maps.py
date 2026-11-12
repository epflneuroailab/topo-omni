"""Cortical-sheet map of a discovered SpaceTop cluster: its clips vs every other clip.

The contrast the discovery walk scores (``calculate_selectivity``: Welch t of a cluster's
sheets against all other clips'), turned into a map in the style of the category maps
(``selectivity/plot_category.py``). It reads the per-clip sheets the walk already cached,
so no model is loaded and no clip is re-extracted.

Sheets are smoothed exactly as the category maps' sweep was: each cached sheet is the
model's unified sheet in its INTERNAL unit order (encoder band first, thinker second), so
it is flattened to a per-unit vector and placed + smoothed through the training
``coords.npy`` by ``NeuronSmoothingConv``. The t-test then runs on the smoothed grid, and
the stats are written in the raster order ``plot_category`` expects (reshape(n_col, n_row)
+ rot90). Draw them with::

    python -m src.experiments.selectivity.plot_category \\
        --results-dir <output_dir>/cluster_maps --category spacetop_cluster_10 \\
        --modality all --stats-suffix _fwhm3.0 --top-k-pct 10 --n-islands 3

``--modality all`` because these clips are audiovisual: the contrast can land anywhere on
the sheet, so top-k is ranked over all of it rather than one modality's band.

    python -m src.experiments.spacetop_discovery.cluster_maps \\
        --model_name topo-omni-tiled32-4608 --run_tag disjoint2s_min60 --clusters 10,7 \\
        --coords <coords.npy the run was trained with>
"""
import argparse
import os
import pickle

import numpy as np
import scipy.stats as stats
from tqdm import tqdm

from src.models.spatial_utils import DEFAULT_SHEET_LAYOUT as SHEET
from src.utils.smoothing import NeuronSmoothingConv
from src.experiments.spacetop_discovery.common import (
    add_run_args, read_json, resolve_run_args,
)


def neighbour_r(m):
    """Horizontal / vertical nearest-neighbour correlation of a map (placement check)."""
    m = np.asarray(m, float)
    ok_h = np.isfinite(m[:, 1:]) & np.isfinite(m[:, :-1])
    ok_v = np.isfinite(m[1:, :]) & np.isfinite(m[:-1, :])
    rh = np.corrcoef(m[:, 1:][ok_h], m[:, :-1][ok_h])[0, 1]
    rv = np.corrcoef(m[1:, :][ok_v], m[:-1, :][ok_v])[0, 1]
    return rh, rv


def smoothed_grid(out, clips, coords, fwhm, chunk=256, cache=None, keep_raw=False):
    """Every clip's sheet placed + smoothed through coords, in stats raster order.

    Returns ``(grid, raw)``: ``grid`` is (n_clips, n_col * n_row) -- the order plot_category
    reshapes to (n_col, n_row) and rotates -- and ``raw`` the unsmoothed per-unit sheets when
    ``keep_raw``. With ``cache``, the smoothed grid is written there once and memory-mapped
    on later calls, so the permutation null does not redo the ~5 min of smoothing.
    """
    if cache and os.path.exists(cache) and not keep_raw:
        grid = np.load(cache, mmap_mode="r")
        assert grid.shape[0] == len(clips), (grid.shape, len(clips))
        print(f"> smoothed sheets from cache {cache} {grid.shape}")
        return grid, None
    sheets_dir = os.path.join(out, "cortical_sheets")
    n_units = SHEET.n_row * SHEET.n_col
    smoother = NeuronSmoothingConv(fwhm_mm=fwhm, resolution_mm=1.0)
    grid = None
    raw = np.empty((len(clips), n_units), np.float32) if keep_raw else None
    for i in tqdm(range(0, len(clips), chunk), desc=f"smoothing (fwhm {fwhm})"):
        batch = np.stack([np.load(os.path.join(sheets_dir, os.path.basename(v) + ".npy"))
                          .reshape(-1) for v in clips[i:i + chunk]]).astype(np.float32)
        assert batch.shape[1] == n_units, batch.shape
        if keep_raw:
            raw[i:i + len(batch)] = batch
        sm = smoother(coords, batch).astype(np.float32)
        if grid is None:
            grid = np.empty((len(clips), sm.shape[1]), np.float32)
        grid[i:i + len(batch)] = sm
    # (n_col, n_row): the smoother's grid is y-by-x, see topo-omni stats placement
    assert (smoother.height, smoother.width) == (SHEET.n_col, SHEET.n_row)
    if cache:
        np.save(cache, grid)
        print(f"> cached smoothed sheets -> {cache}")
    return grid, raw


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_run_args(ap)
    ap.add_argument("--clusters", required=True, help="comma-separated cluster ids, e.g. 10,7")
    ap.add_argument("--coords", required=True,
                    help="coords.npy the run was TRAINED with (per-unit sheet positions).")
    ap.add_argument("--fwhm", type=float, default=3.0,
                    help="smoothing FWHM in mm; 3.0 matches the category maps.")
    ap.add_argument("--clusters_json", default=None,
                    help="Default: <output_dir>/clusters_tvals.json")
    ap.add_argument("--chunk", type=int, default=256, help="sheets smoothed per batch")
    args = resolve_run_args(ap.parse_args())

    out = args.output_dir
    clusters = {c["cluster_id"]: c["video_ids"]
                for c in read_json(args.clusters_json or os.path.join(out, "clusters_tvals.json"))}
    want = [int(c) for c in args.clusters.split(",")]
    missing = [c for c in want if c not in clusters]
    if missing:
        raise SystemExit(f"clusters not in this run: {missing}")

    # The walk's clip set, in the order its sheets were stacked.
    clips = read_json(os.path.join(out, "clips_used.json"))
    coords = np.load(args.coords)
    n_units = SHEET.n_row * SHEET.n_col
    assert coords.shape == (n_units, 2), (coords.shape, n_units)

    # raw is kept for the placement check below; the smoothed grid is cached for
    # cluster_null.py
    cache = os.path.join(out, "cluster_maps", f"smoothed_sheets_fwhm{args.fwhm}.npy")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    grid, raw = smoothed_grid(out, clips, coords, args.fwhm, args.chunk, cache=cache,
                              keep_raw=True)
    H, W = SHEET.n_col, SHEET.n_row                  # (n_col, n_row): y-by-x
    print(f"> {len(clips)} sheets on a {H}x{W} grid")

    pos = {v: k for k, v in enumerate(clips)}
    for cid in want:
        idx = np.array([pos[v] for v in clusters[cid] if v in pos])
        if len(idx) != len(clusters[cid]):
            print(f"  ! cluster {cid}: {len(clusters[cid]) - len(idx)} clips have no sheet")
        mask = np.zeros(len(clips), bool)
        mask[idx] = True
        # One-sided, as in calculate_selectivity and run_selectivity: q marks units that
        # respond MORE to the cluster. t itself is the same either way.
        t, p = stats.ttest_ind(grid[mask], grid[~mask], axis=0, equal_var=False,
                               alternative="greater")
        q = stats.false_discovery_control(np.nan_to_num(p, nan=1.0))
        name = f"spacetop_cluster_{cid}"
        d = os.path.join(out, "cluster_maps", name)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{name}_fwhm{args.fwhm}_selectivity_stats.pkl")
        with open(path, "wb") as f:
            pickle.dump(dict(t=t.astype(np.float32), p=p, q=q, n_target=int(mask.sum()),
                             n_baseline=int((~mask).sum()), video_ids=clusters[cid],
                             fwhm_mm=args.fwhm, coords=args.coords), f)
        rh, rv = neighbour_r(np.rot90(t.reshape(H, W), k=1))
        # Smoothing makes ANY map spatially correlated, so it cannot show the placement is
        # right. The unsmoothed per-unit t scattered through coords can: positive neighbour
        # r if coords matches these sheets, ~0 if it does not.
        t_raw, _ = stats.ttest_ind(raw[mask], raw[~mask], axis=0, equal_var=False)
        ij = coords.astype(np.int64)
        placed = np.full((SHEET.n_col, SHEET.n_row), np.nan, np.float32)
        placed[ij[:, 1], ij[:, 0]] = t_raw
        rh0, rv0 = neighbour_r(np.rot90(placed, k=1))
        print(f"> cluster {cid}: n={mask.sum()} vs {(~mask).sum()} | t max {np.nanmax(t):.1f} "
              f"| q<0.05 & t>0: {int(((q < 0.05) & (t > 0)).sum()):,} units "
              f"| neighbour r smoothed h={rh:+.3f} v={rv:+.3f}, unsmoothed-by-coords "
              f"h={rh0:+.3f} v={rv0:+.3f} | {path}")


if __name__ == "__main__":
    main()
