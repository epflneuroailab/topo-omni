"""Depth-corrected island scores for the localizer category maps, from a label-permutation null.

The coarse spatial-loss radius grows 4 -> 32 across the encoder layers and stays at 32 in the
thinker, so islands in deep tiles can be large for reasons unrelated to the contrast. This
builds the chance distribution of island size PER TILE (one tile = one layer) and rescales
each observed island by it, the same correction ``spacetop_discovery/cluster_null.py`` applies
to the SpaceTop cluster maps -- here the null shuffles the localizer's own ON/OFF labels.

Per category:
  1. the cached per-unit ON/OFF features of the selectivity sweep are placed + smoothed
     through the training coords, and the Welch t is checked against the saved stats map;
  2. ``--n_perm`` label shuffles (ON/OFF group sizes kept) go through the category map's
     island pipeline exactly: top-k% of the modality band, 3x3 opening, >= 20 units,
     8-connected islands;
  3. each observed island gets
         corrected mass = (size / median chance-island size in its tile) x mean t
         p_FWE          = share of shuffles whose LARGEST corrected mass reaches it

Output ``<results-dir>/<cat>/<cat><suffix>_island_null.json``, read by
``plot_category.py --island-rank corrected`` and ``plot.py --island-rank corrected``.

    python -m src.experiments.selectivity.island_null \\
        --results-dir results/topo-omni-tiled32-4608 --stats-suffix _fwhm3.0 \\
        --coords <coords.npy the run was trained with>
"""
import argparse
import collections
import json
import os
import pickle
import zlib
from multiprocessing import Pool

import numpy as np
from scipy.ndimage import binary_opening
from tqdm import tqdm

from src.models.spatial_utils import DEFAULT_SHEET_LAYOUT as SHEET
from src.utils.connected_components import label_islands, island_stats
from src.utils.smoothing import NeuronSmoothingConv
from src.experiments.selectivity.plot_category import remove_small_components

H, W = SHEET.n_row, SHEET.n_col
THK, MID = SHEET.thinker_rows, SHEET.n_col // 2
# band of each localizer, as plot_category's MODALITY assigns it
BAND = {"faces": "vision", "bodies": "vision", "scenes": "vision", "objects": "vision",
        "vwfa": "vision", "speech": "audio", "vocals": "audio",
        "language_text_ALL": "thinker", "theory_of_mind_text_ALL": "thinker",
        "multiple_demand_text_ALL": "thinker",
        # Moseley verb/noun negative control: single words, text input -> thinker band
        "moseley_concrete_verbs": "thinker", "moseley_concrete_nouns": "thinker",
        "moseley_abstract_verbs": "thinker", "moseley_abstract_nouns": "thinker"}


def band_mask(band):
    m = np.zeros((H, W), bool)
    if band == "thinker":
        m[:THK] = True
    elif band == "vision":
        m[THK:, :MID] = True
    elif band == "audio":
        m[THK:, MID:] = True
    else:
        m[:] = True
    return m


def tile_of(r, c):
    """(band, tile_row, tile_col) of a display-space position; each tile is one layer."""
    if r < THK:
        th, tw = SHEET.thk_tile[:2]
        return ("thinker", int(r // th), int(c // tw))
    eh, ew = SHEET.enc_tile[:2]
    band = "vision" if c < MID else "audio"
    return (band, int((r - THK) // eh), int((c - (0 if c < MID else MID)) // ew))


def band_islands(t_flat, band, top_k_pct):
    """Islands exactly as plot_category / plot.py draw one category (anatomical band on)."""
    t = np.rot90(np.asarray(t_flat, np.float64).reshape(W, H), k=1)
    t = np.nan_to_num(t, nan=-np.inf)
    bm = band_mask(band)
    t[~bm] = -np.inf
    k = int((top_k_pct / 100) * bm.sum())
    flat = t.ravel()
    mask = np.zeros(flat.size, bool)
    mask[np.argpartition(flat, -k)[-k:]] = True
    clean = binary_opening(mask.reshape(H, W), structure=np.ones((3, 3)))
    clean = remove_small_components(clean, min_size=20)
    labeled, n = label_islands(clean.astype(np.uint8), connectivity=8)
    return labeled, island_stats(labeled, n, t)


# ---- workers: the smoothed stimuli live in a module global so fork shares them ----------
_X = _S1 = _S2 = None


def welch_t(on_idx):
    a = _X[on_idx].astype(np.float64)
    n1, n2 = len(on_idx), _X.shape[0] - len(on_idx)
    s1, q1 = a.sum(0), np.einsum("ij,ij->j", a, a)
    s2, q2 = _S1 - s1, _S2 - q1
    m1, m2 = s1 / n1, s2 / n2
    v1 = (q1 - s1 * m1) / (n1 - 1)
    v2 = (q2 - s2 * m2) / (n2 - 1)
    return (m1 - m2) / np.sqrt(v1 / n1 + v2 / n2)


def _one_perm(job):
    n_on, seed, band, top_k = job
    rng = np.random.default_rng(seed)
    on = np.sort(rng.choice(_X.shape[0], size=n_on, replace=False))
    _, st = band_islands(welch_t(on), band, top_k)
    return [(d["size"], float(d["average_t_values"]), tile_of(*d["centroid"])) for d in st]


def main():
    global _X, _S1, _S2
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--categories", default=",".join(BAND))
    ap.add_argument("--stats-suffix", default="_fwhm3.0")
    ap.add_argument("--coords", required=True)
    ap.add_argument("--fwhm", type=float, default=3.0)
    ap.add_argument("--top-k-pct", type=float, default=10.0)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--min-null", type=int, default=50)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    coords = np.load(args.coords).astype(int)      # as maybe_hrf_project passes them
    for cat in [c.strip() for c in args.categories.split(",") if c.strip()]:
        band = BAND[cat]
        d = os.path.join(args.results_dir, cat)
        with open(os.path.join(d, "cache", f"selectivity_{cat}_features.pkl"), "rb") as f:
            feats = pickle.load(f)
        with open(os.path.join(d, f"{cat}{args.stats_suffix}_selectivity_stats.pkl"), "rb") as f:
            st = pickle.load(f)
        smoother = NeuronSmoothingConv(fwhm_mm=args.fwhm, resolution_mm=1.0)
        on = smoother(coords, np.asarray(feats["on_sheet"], np.float32)).astype(np.float32)
        off = smoother(coords, np.asarray(feats["off_sheet"], np.float32)).astype(np.float32)
        assert (smoother.height, smoother.width) == (W, H)
        _X = np.concatenate([on, off])
        _S1 = _X.sum(0, dtype=np.float64)
        _S2 = np.einsum("ij,ij->j", _X.astype(np.float64), _X.astype(np.float64))
        n_on = len(on)

        t_obs = welch_t(np.arange(n_on))
        rel = np.nanmax(np.abs(t_obs - st["t"])) / np.nanmax(np.abs(st["t"]))
        print(f"> {cat}: {n_on} ON vs {len(off)} OFF stimuli, band {band}; "
              f"t vs saved map, max rel diff {rel:.1e}", flush=True)
        assert rel < 1e-3, f"{cat}: cached features do not reproduce the saved map ({rel})"

        # zlib.crc32, not hash(): str hashes are salted per process, so seeds would not repeat
        jobs = [(n_on, args.seed * 1_000_003 + zlib.crc32(cat.encode()) % 100_000 * 1000 + p, band,
                 args.top_k_pct) for p in range(args.n_perm)]
        with Pool(args.workers) as pool:
            perms = list(tqdm(pool.imap(_one_perm, jobs, chunksize=4), total=len(jobs),
                              desc=f"null {cat}", mininterval=30))

        by = [collections.defaultdict(list) for _ in range(3)]    # tile, depth row, band
        for isl in perms:
            for size, _, tile in isl:
                by[0][tile].append(size); by[1][tile[:2]].append(size); by[2][tile[0]].append(size)
        ref_cache = {}

        def ref(tile):
            if tile not in ref_cache:
                for pool_, key, lvl in ((by[0], tile, "tile"), (by[1], tile[:2], "row"),
                                        (by[2], tile[0], "band")):
                    if len(pool_.get(key, ())) >= args.min_null:
                        ref_cache[tile] = (float(np.median(pool_[key])), lvl, len(pool_[key]))
                        break
                else:
                    allv = [s for v in by[2].values() for s in v]
                    ref_cache[tile] = (float(np.median(allv)) if allv else 20.0, "sheet", len(allv))
            return ref_cache[tile]

        max_cm = np.array([max((s / ref(t)[0]) * mt for s, mt, t in isl) if isl else 0.0
                           for isl in perms])
        # the observed islands, labelled from the SAVED map so ids match the plotters
        _, obs = band_islands(st["t"], band, args.top_k_pct)
        rows = []
        for o in obs:
            tile = tile_of(*o["centroid"])
            r_size, lvl, n_ref = ref(tile)
            cm = (o["size"] / r_size) * float(o["average_t_values"])
            rows.append(dict(id=int(o["id"]), size=int(o["size"]),
                             mean_t=float(o["average_t_values"]),
                             mass=float(o["size"] * o["average_t_values"]),
                             centroid=[float(x) for x in o["centroid"]], tile=list(tile),
                             null_median_size=r_size, null_level=lvl, null_n=n_ref,
                             corrected_mass=cm,
                             p_fwe=float((1 + (max_cm >= cm).sum()) / (1 + len(max_cm)))))
        for r, row in enumerate(sorted(rows, key=lambda r: -r["size"]), 1):
            row["rank_size"] = r
        rows.sort(key=lambda r: -r["corrected_mass"])
        for r, row in enumerate(rows, 1):
            row["rank_corrected"] = r
        out = os.path.join(d, f"{cat}{args.stats_suffix}_island_null.json")
        with open(out, "w") as f:
            json.dump(dict(category=cat, band=band, n_on=n_on, n_off=int(len(off)),
                           n_perm=args.n_perm, top_k_pct=args.top_k_pct,
                           null_islands_per_perm=float(np.mean([len(p) for p in perms])),
                           null_median_size_by_depth_row={f"{b} row {k}": float(np.median(v))
                                                          for (b, k), v in sorted(by[1].items())},
                           islands=rows), f, indent=1)
        print(f"  null: {np.mean([len(p) for p in perms]):.0f} islands/shuffle | "
              f"chance-island size by depth row: " + ", ".join(
                  f"r{k}:{np.median(v):.0f}" for (b, k), v in sorted(by[1].items())))
        for row in rows[:5]:
            print(f"   #{row['rank_corrected']} (size rank {row['rank_size']}): id {row['id']} "
                  f"size {row['size']} mean t {row['mean_t']:.1f} tile {tuple(row['tile'])} "
                  f"chance {row['null_median_size']:.0f} -> corrected {row['corrected_mass']:.1f} "
                  f"p_FWE {row['p_fwe']:.3f}")
        print(f"  -> {out}", flush=True)


if __name__ == "__main__":
    main()
