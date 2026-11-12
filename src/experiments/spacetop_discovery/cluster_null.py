"""Depth-corrected island scores for a SpaceTop cluster map, from a permutation null.

Why. The coarse spatial-loss radius grows 4 -> 32 across the 32 encoder layers and stays at
32 in the thinker, and the map's wavelength follows it (~2.4 r, see the run config). Chance
islands are therefore far larger in the thinker and late encoder than in early encoder
layers, and ranking a cluster map's islands by raw mass (size x mean t) favours deep
regions whatever the contrast.

What. Random clip sets the size of the cluster go through the IDENTICAL map pipeline --
Welch t on the 3 mm-smoothed sheets, top-10% of the whole sheet, 3x3 opening, >= 20 units,
8-connected islands -- and every chance island's size, mean t and tile is recorded. Each
tile holds one layer, so a tile's chance islands give the island size expected at that
depth, whatever produced it (loss radius, smoothing, the fine term).

Sets are drawn as blocks of consecutive clips (``--block``, default 10 = 20 s): adjacent 2 s
clips of a movie are strongly correlated, and single-clip permutation would make the null
too narrow and the p-values too small.

Scores, per observed island:
    corrected mass = (size / median chance-island size in its tile) x mean t
    p_FWE          = share of permutations whose LARGEST corrected mass reaches it
Tiles with fewer than ``--min_null`` chance islands borrow from their depth row (same band,
same tile row), then from the whole band.

    python -m src.experiments.spacetop_discovery.cluster_null \\
        --model_name topo-omni-tiled32-4608 --run_tag disjoint2s_min60 --clusters 10,7 \\
        --coords <coords.npy the run was trained with>

Writes ``<output_dir>/cluster_maps/spacetop_cluster_<id>/spacetop_cluster_<id>_island_null.json``,
which ``selectivity/plot_category.py --island-rank corrected --island-scores <json>`` reads.
"""
import argparse
import collections
import json
import os
import pickle
import re
from multiprocessing import Pool

import numpy as np
from scipy.ndimage import binary_opening
from tqdm import tqdm

from src.models.spatial_utils import DEFAULT_SHEET_LAYOUT as SHEET
from src.utils.connected_components import label_islands, island_stats
from src.experiments.selectivity.plot_category import remove_small_components
from src.experiments.spacetop_discovery.cluster_maps import smoothed_grid
from src.experiments.spacetop_discovery.common import add_run_args, read_json, resolve_run_args

CHUNK = re.compile(r"^(?P<movie>.*)_chunk_(?P<n>\d+)\.mp4$")
H, W = SHEET.n_row, SHEET.n_col
THK, MID = SHEET.thinker_rows, SHEET.n_col // 2


# ----------------------------------------------------------------------------- geometry
def tile_of(r, c):
    """(band, tile_row, tile_col) of a display-space position; each tile is one layer."""
    if r < THK:
        th, tw = SHEET.thk_tile[:2]
        return ("thinker", int(r // th), int(c // tw))
    eh, ew = SHEET.enc_tile[:2]
    band = "vision" if c < MID else "audio"
    return (band, int((r - THK) // eh), int((c - (0 if c < MID else MID)) // ew))


# ------------------------------------------------------------- the map pipeline, shared
def islands(t_flat, top_k_pct=10.0):
    """Islands of a t-map exactly as plot_category draws them (modality 'all').

    ``t_flat`` is in stats raster order (reshape(n_col, n_row) + rot90). Returns
    ``(labeled, stats)`` with stats from ``island_stats`` (id, size, centroid,
    average_t_values), so ids match what plot_category labels on the same map.
    """
    t = np.rot90(np.asarray(t_flat, np.float64).reshape(W, H), k=1)
    t = np.nan_to_num(t, nan=-np.inf)
    k = int((top_k_pct / 100) * H * W)
    flat = t.ravel()
    mask = np.zeros(flat.size, bool)
    mask[np.argpartition(flat, -k)[-k:]] = True
    clean = binary_opening(mask.reshape(H, W), structure=np.ones((3, 3)))
    clean = remove_small_components(clean, min_size=20)
    labeled, n = label_islands(clean.astype(np.uint8), connectivity=8)
    return labeled, island_stats(labeled, n, t)


# ----------------------------------------------------------------- fast Welch t for sets
_X = _S1 = _S2 = None


def _init(path):
    global _X
    _X = np.load(path, mmap_mode="r")


def welch_t(idx):
    """Welch t of rows ``idx`` against all other rows, from the precomputed column sums."""
    a = np.asarray(_X[np.sort(idx)], np.float64)
    n1, n2 = len(idx), _X.shape[0] - len(idx)
    s1, q1 = a.sum(0), np.einsum("ij,ij->j", a, a)
    s2, q2 = _S1 - s1, _S2 - q1
    m1, m2 = s1 / n1, s2 / n2
    v1 = (q1 - s1 * m1) / (n1 - 1)
    v2 = (q2 - s2 * m2) / (n2 - 1)
    return (m1 - m2) / np.sqrt(v1 / n1 + v2 / n2)


def _one_perm(job):
    n, seed, blocks_of_movie, block = job
    rng = np.random.default_rng(seed)
    # cut each movie into consecutive blocks with a random phase, shuffle, take to n
    pieces = []
    for rows in blocks_of_movie:
        off = rng.integers(block)
        cuts = list(range(0, len(rows), block)) if off == 0 else \
            [0] + list(range(off, len(rows), block))
        pieces += [rows[a:b] for a, b in zip(cuts, cuts[1:] + [len(rows)]) if b > a]
    rng.shuffle(pieces)
    idx, got = [], 0
    for p in pieces:
        take = p[:n - got]
        idx += list(take)
        got += len(take)
        if got >= n:
            break
    _, st = islands(welch_t(np.array(idx)))
    return [(d["size"], float(d["average_t_values"]), tile_of(*d["centroid"])) for d in st]


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_run_args(ap)
    ap.add_argument("--clusters", required=True)
    ap.add_argument("--coords", required=True)
    ap.add_argument("--fwhm", type=float, default=3.0)
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--block", type=int, default=10, help="clips per permutation block")
    ap.add_argument("--min_null", type=int, default=50,
                    help="chance islands a tile needs before it stands on its own")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = resolve_run_args(ap.parse_args())

    global _S1, _S2
    out = args.output_dir
    clips = read_json(os.path.join(out, "clips_used.json"))
    clusters = {c["cluster_id"]: c["video_ids"]
                for c in read_json(os.path.join(out, "clusters_tvals.json"))}
    cache = os.path.join(out, "cluster_maps", f"smoothed_sheets_fwhm{args.fwhm}.npy")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    if not os.path.exists(cache):
        smoothed_grid(out, clips, np.load(args.coords), args.fwhm, cache=cache)
    X = np.load(cache, mmap_mode="r")

    _S1 = np.zeros(X.shape[1]); _S2 = np.zeros(X.shape[1])
    for i in tqdm(range(0, X.shape[0], 256), desc="column sums"):
        a = np.asarray(X[i:i + 256], np.float64)
        _S1 += a.sum(0); _S2 += np.einsum("ij,ij->j", a, a)

    # rows of each movie in clip order, for block sampling
    per_movie = collections.defaultdict(list)
    for row, v in enumerate(clips):
        m = CHUNK.match(os.path.basename(v))
        per_movie[m.group("movie")].append((int(m.group("n")), row))
    blocks_of_movie = [np.array([r for _, r in sorted(v)]) for v in per_movie.values()]

    _init(cache)
    for cid in [int(c) for c in args.clusters.split(",")]:
        name = f"spacetop_cluster_{cid}"
        d = os.path.join(out, "cluster_maps", name)
        with open(os.path.join(d, f"{name}_fwhm{args.fwhm}_selectivity_stats.pkl"), "rb") as f:
            st = pickle.load(f)
        pos = {v: k for k, v in enumerate(clips)}
        idx = np.array([pos[v] for v in clusters[cid]])

        # The fast t must reproduce the saved map before it is trusted for the null. The
        # saved map came from scipy on float32 sheets, so it carries float32 rounding
        # (~1e-4 relative); this one accumulates in float64. Compare relatively.
        dt = np.nanmax(np.abs(welch_t(idx) - st["t"]))
        rel = dt / np.nanmax(np.abs(st["t"]))
        print(f"> cluster {cid}: fast Welch t vs saved map, max |diff| = {dt:.2e} "
              f"({rel:.1e} of max |t|)")
        assert rel < 1e-3, rel

        jobs = [(len(idx), args.seed * 100003 + cid * 7919 + p, blocks_of_movie, args.block)
                for p in range(args.n_perm)]
        with Pool(args.workers, initializer=_init, initargs=(cache,)) as pool:
            perms = list(tqdm(pool.imap(_one_perm, jobs, chunksize=4), total=len(jobs),
                              desc=f"null, cluster {cid}"))

        # chance-island size per tile, with depth-row and band fallbacks
        by_tile, by_row, by_band = (collections.defaultdict(list) for _ in range(3))
        for isl in perms:
            for size, _, tile in isl:
                by_tile[tile].append(size); by_row[tile[:2]].append(size)
                by_band[tile[0]].append(size)

        _ref = {}

        def ref_size(tile):
            if tile not in _ref:
                for pool_, key, lvl in ((by_tile, tile, "tile"), (by_row, tile[:2], "row"),
                                        (by_band, tile[0], "band")):
                    if len(pool_.get(key, ())) >= args.min_null:
                        _ref[tile] = (float(np.median(pool_[key])), lvl, len(pool_[key]))
                        break
                else:
                    _ref[tile] = (float(np.median(sum(by_band.values(), []))), "sheet", 0)
            return _ref[tile]

        # null of the per-permutation MAXIMUM corrected mass, for family-wise p
        max_cm = np.array([max((s / ref_size(t)[0]) * mt for s, mt, t in isl) if isl else 0.0
                           for isl in perms])

        labeled, obs = islands(st["t"])
        rows = []
        for o in obs:
            tile = tile_of(*o["centroid"])
            ref, lvl, n_ref = ref_size(tile)
            cm = (o["size"] / ref) * float(o["average_t_values"])
            rows.append(dict(id=int(o["id"]), size=int(o["size"]),
                             mean_t=float(o["average_t_values"]),
                             mass=float(o["size"] * o["average_t_values"]),
                             centroid=[float(x) for x in o["centroid"]],
                             tile=list(tile), null_median_size=ref, null_level=lvl,
                             null_n=n_ref, corrected_mass=cm,
                             p_fwe=float((1 + (max_cm >= cm).sum()) / (1 + len(max_cm)))))
        rows.sort(key=lambda r: -r["corrected_mass"])
        for r, row in enumerate(rows, 1):
            row["rank_corrected"] = r
        for r, row in enumerate(sorted(rows, key=lambda r: -r["mass"]), 1):
            row["rank_mass"] = r

        # the depth gradient the correction removes, for the record
        gradient = {f"{b} row {k}": float(np.median(v))
                    for (b, k), v in sorted(by_row.items())}
        out_json = os.path.join(d, f"{name}_island_null.json")
        with open(out_json, "w") as f:
            json.dump(dict(cluster=cid, n_clips=int(len(idx)), n_perm=args.n_perm,
                           block=args.block, min_null=args.min_null,
                           null_islands_per_perm=float(np.mean([len(p) for p in perms])),
                           null_median_size_by_depth_row=gradient, islands=rows), f, indent=1)
        print(f"  null: {np.mean([len(p) for p in perms]):.0f} islands/permutation; "
              f"median chance-island size thinker {np.median(by_band['thinker']):.0f} px, "
              f"vision {np.median(by_band['vision']):.0f}, audio {np.median(by_band['audio']):.0f}")
        print(f"  {'rank':>4} {'mass#':>5} {'size':>5} {'mean t':>6} {'tile':<22} "
              f"{'null med':>8} {'corr. mass':>10} {'p_FWE':>7}")
        for row in rows[:8]:
            print(f"  {row['rank_corrected']:>4} {row['rank_mass']:>5} {row['size']:>5} "
                  f"{row['mean_t']:>6.1f} {str(tuple(row['tile'])):<22} "
                  f"{row['null_median_size']:>8.0f} {row['corrected_mass']:>10.1f} "
                  f"{row['p_fwe']:>7.3f}")
        print(f"  -> {out_json}")


if __name__ == "__main__":
    main()
