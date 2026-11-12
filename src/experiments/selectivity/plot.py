import os
import json
import argparse
import numpy as np

from src.models.spatial_utils import DEFAULT_SHEET_LAYOUT as SHEET
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from scipy.ndimage import binary_opening

from skimage import measure
from src.utils.connected_components import (label_islands, island_stats, print_stats,
                                            keep_only_id, keep_only_ids)
from src.utils.island_morans_I import island_morans_I
from src.core.model_loading import MODEL_TITLE

def remove_small_components(mask, min_size):
    labeled = measure.label(mask)
    cleaned = np.zeros_like(mask)
    for region in measure.regionprops(labeled):
        if region.area >= min_size:
            cleaned[labeled == region.label] = 1
    return cleaned

def _guess_stats_path(dirpath, category):
    """Legacy stats-name guess, used only when --stats-suffix is not given.

    run_selectivity writes "<cat>_selectivity_stats.pkl"; the "_all" variant only appears
    when category_name picked up a suffix. Accept either -- but prefer --stats-suffix,
    because this guess cannot tell a current file from a stale one left by an earlier run.
    """
    filepath = f"{dirpath}/{category}/{category}_all_selectivity_stats.pkl"
    if os.path.exists(filepath):
        return filepath
    alt = f"{dirpath}/{category}/{category}_selectivity_stats.pkl"
    if os.path.exists(alt):
        return alt
    raise SystemExit(f"no stats for {category}: tried\n  {filepath}\n  {alt}")


def read_pickle(filepath):
    with open(filepath, "rb") as f:
        data = pkl.load(f)
    return data

def dump_pickle(data, filepath):
    with open(filepath, "wb") as f:
        pkl.dump(data, f)

def write_json(data, filepath):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

conditions_labels = [
    "Faces",
    "Bodies",
    "Scenes",
    "Objects",
    "Words",
    "False Belief",
    "False Photo",
    "Nonwords",
    "Quilted Speech",
    "Math"
]

# -----------------------------
# Color palettes
# -----------------------------
# The paper's original muted two-family palette -- warm = the 5 VISION categories, cool =
# the 5 audio/text ones -- re-spaced so every pair that actually meets can be told apart.
# The original was #E64B35 #F39B2F #F1C232 #EFC94C #D4AC0D | #5AB4AC #4C9F9B #3F8F9C
# #3B5B92 #2F3E75, and the published bars went through sns.barplot's saturation=0.75.
# As rendered, Scenes/Objects was dE2000 = 2.8 (indistinguishable), FB/FP 6.4, FP/NW 7.1,
# Scenes/Words 7.5 and Speech/MD 8.7; below ~10 two fills read as one colour.
#
# Each colour keeps its hue (terracotta, ochre, mustard, khaki, olive | sage, teal, steel,
# slate, navy) and chroma stays roughly uniform within a family, so the palette still
# reads as the original; the separation comes from LIGHTNESS. Over the pairs that meet --
# all 5 vision categories (they overlap at the top of the vision encoder), speech/vocals,
# language/ToM/MD in the thinker, neighbouring profile bars and the two-bar cognitive
# contrasts -- the worst pair is now dE2000 = 11.1 (FB/FP).
#
# These are the final fills: nothing downstream desaturates them (the old palette relied
# on sns.barplot doing so), and the same ten values in the same order are the condition
# colours in response_profiles/plot.py and plot_cognitive.py. Slot order is
# conditions_labels order; `selectivity` below maps slots to categories: "False Belief" ->
# theory_of_mind_text, "False Photo" -> vocals, "Nonwords" -> language_text,
# "Quilted Speech" -> speech, "Math" -> multiple_demand_text.
colors = ["#A04D3B", "#BF8649", "#CDB34C", "#E7DFA1", "#877C26",
     "#92CBC3", "#61A6A4", "#478190", "#485682", "#303557"]

CLUSTER_NAMES = {
    "faces": "Faces", "bodies": "Bodies", "scenes": "Scenes", "objects": "Objects",
    "vwfa": "VWFA", "speech": "Speech", "vocals": "Vocals",
    "language_text": "Language", "language_text_ALL": "Language",
    "theory_of_mind_text": "Theory of Mind", "theory_of_mind_text_ALL": "Theory of Mind",
    "multiple_demand_text": "Multiple Demand", "multiple_demand_text_ALL": "Multiple Demand",
    "fedorenko_words_nonwords": "Words vs Nonwords",
}

cmaps = {label: mcolors.LinearSegmentedColormap.from_list("custom", [color, "#E7E7E7"]) for color, label in zip(colors, conditions_labels)}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Render the unified in-silico cortical sheet with all category localizers overlaid (Fig 2b / 3-5 maps).")
    parser.add_argument("--results-dir", default=os.path.join(os.getenv("SAVE_DIR", "results"), MODEL_TITLE),
                        help="Directory with per-category selectivity stats "
                             "(<dir>/<category>/<category>_all_selectivity_stats.pkl).")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory for the figure (default: <results-dir>/unified_map).")
    parser.add_argument("--top-k-pct", type=float, default=10.0, help="Top-k%% most selective units to keep.")
    parser.add_argument("--p-threshold", type=float, default=0.001, help="FDR q-value threshold.")
    parser.add_argument("--no-anatomical", action="store_true", help="Disable the anatomical constraint.")
    parser.add_argument("--filter-nonsig", action="store_true", help="Filter out non-significant units.")
    parser.add_argument("--largest-island-only", action="store_true",
                        help="reduce the drawn mask to the single largest connected "
                             "component. OFF by default: at top-10%% the selection is "
                             "41,779 units across >5,600 components, and keeping only the "
                             "largest (193 units) renders an almost blank sheet that "
                             "looks like there is no selectivity at all.")
    parser.add_argument("--n-islands", type=int, default=None,
                        help="draw the N largest connected components per category "
                             "instead of just one. --largest-island-only is the N=1 case "
                             "and still works; this generalises it. A category with fewer "
                             "than N surviving islands simply draws all it has.")
    parser.add_argument("--island-rank", default="size", choices=["size", "corrected"],
                        help="which islands --n-islands keeps per category. 'size' "
                             "(default): the N largest. 'corrected': the N with the largest "
                             "depth-corrected mass, (size / chance-island size at that "
                             "depth) x mean t, read from <results-dir>/<cat>/<cat>"
                             "<stats-suffix>_island_null.json (selectivity/island_null.py). "
                             "The deep layers' larger spatial-loss radius makes their islands "
                             "big by construction; this ranks against that.")
    parser.add_argument("--morph-clean", action="store_true",
                        help="apply binary_opening(3x3) + remove_small_components(20) to "
                             "the selection before labelling islands. OFF by default: on "
                             "a sheet whose coordinates are shuffled within each layer, "
                             "the opening requires solid 3x3 blocks and erased the entire "
                             "top-1%% selection (4,177 units over 417,792 positions), so "
                             "it was deleting the data rather than cleaning it.")
    parser.add_argument("--categories", default=None,
                        help="comma-separated subset to draw, e.g. 'faces'. Default: every "
                             "category in the hardcoded list, which throws if any of their "
                             "stats files are missing.")
    parser.add_argument("--coords", default=None,
                        help="coords.npy the run was TRAINED with. Required: stats are in "
                             "per-unit order and build_sheet_coordinates shuffles every "
                             "unit within its layer block, so a plain reshape scatters "
                             "each layer at random. Without this the map is meaningless.")
    parser.add_argument("--stats-suffix", default=None,
                        help="name the stats file explicitly: <cat><suffix>_selectivity_"
                             "stats.pkl, e.g. '_fwhm3.0'. run_selectivity appends "
                             "'_fwhm<f>' to category_name when smoothing is on, so a "
                             "smoothed sweep writes a name the legacy guess does not "
                             "match -- and if an older unsmoothed sweep left a stale "
                             "'<cat>_all_...pkl' behind, the guess silently picks THAT up "
                             "and draws one category from a different run.")
    parser.add_argument("--pre-gridded", action="store_true",
                        help="the stats vector is ALREADY in grid raster order, so place "
                             "it by reshape rather than by scattering through coords.npy. "
                             "Identical to omitting --coords; it exists to state the "
                             "intent and to make passing --coords too a hard error rather "
                             "than a silent scramble. Every SMOOTHED sweep needs this: "
                             "run_selectivity calls maybe_hrf_project BEFORE run_stats, so "
                             "the t-test runs on grid-projected data and stats['t'] is "
                             "length H*W in raster order -- which is why run_selectivity "
                             "then does a plain .reshape(H_lm, W_lm) with no coords. "
                             "Scattering such a vector permutes it a SECOND time: measured "
                             "on tiled32 faces, nearest-neighbour r falls from +0.983 to "
                             "+0.002. For an UNSMOOTHED sweep the opposite holds (+0.0004 "
                             "by reshape vs +0.517 by coords) -- pass --coords there.")
    parser.add_argument("--faded-background", action="store_true",
                        help="draw the full sheet as one faint grey backdrop UNDER the "
                             "categories, so a small top-k selection reads as figure "
                             "against ground instead of floating on white.")
    parser.add_argument("--label-clusters", action="store_true",
                        help="write each cluster's name in bold below it")
    parser.add_argument("--label-size", type=float, default=8.0)
    parser.add_argument("--label-pad", type=float, default=14.0,
                        help="gap in sheet px between the cluster and its label")
    parser.add_argument("--no-svg", action="store_true",
                        help="write only the PNG. The SVG embeds the heatmap as vector "
                             "cells -- one path per sheet position per category -- so it "
                             "is slow to write and very large at 800x576 x N categories.")
    args = parser.parse_args()

    selectivity = [
        ("faces", "vision", "Faces"),
        ("bodies", "vision", "Bodies"),
        ("scenes", "vision", "Scenes"),
        ("objects", "vision", "Objects"),
        ("vwfa", "vision", "Words"),
        ("speech", "audio", "Quilted Speech"),
        ("vocals", "audio", "False Photo"),
        ("language_text_ALL", "language", "Nonwords"),
        ("theory_of_mind_text_ALL", "cognitive", "False Belief"),
        ("multiple_demand_text_ALL", "cognitive", "Math"),

        # ("fedorenko_words_nonwords", "language", "Nonwords"),
        # ("theory_of_mind", "cognitive", "viridis"),
        # ("multi_demand", "cognitive", "viridis"),
    ]

    dirpath = args.results_dir
    if args.categories:
        _keep = {c.strip() for c in args.categories.split(",")}
        selectivity = [t for t in selectivity if t[0] in _keep]
        missing = _keep - {t[0] for t in selectivity}
        if missing:
            raise SystemExit(f"unknown categories: {sorted(missing)}")
        print(f"> categories: {[t[0] for t in selectivity]}")
    if args.pre_gridded and args.coords:
        raise SystemExit("--pre-gridded and --coords are mutually exclusive: the first says "
                         "the vector is already in grid order, the second says it is in "
                         "per-unit order. Applying both permutes the map twice.")
    _coords = np.load(args.coords) if args.coords else None
    if _coords is not None:
        print(f"> placing units via {args.coords} {_coords.shape}")

    def _place(vec):
        """Per-unit vector -> (H, W) sheet, oriented like run_selectivity's output.

        With --coords this uses the TRAINING assignment (the only correct one). The old
        code did `reshape(W, H)` then rot90, i.e. it assumed unit k sits at the k-th grid
        cell -- true only if coordinates were never shuffled, which they always are.
        """
        if args.pre_gridded or _coords is None:
            # reshape(W, H) -- NOT (H, W). The smoother builds its grid as
            # H=len(ys)=span(coords[:,1])=n_col rows by W=len(xs)=span(coords[:,0])=n_row
            # cols, i.e. y-by-x, the transpose of the (n_row, n_col) sheet convention.
            # Reshaping to (n_row, n_col) preserves adjacency ALONG each row and destroys
            # it ACROSS rows: on tiled32 faces, horizontal neighbour r stays +0.982 while
            # vertical drops to -0.008, rendering as 1-cell-tall horizontal streaks that
            # look like noise. (W, H) + rot90 gives +0.983 both ways and matches the
            # coords branch below exactly.
            return np.rot90(np.asarray(vec, dtype=np.float64).reshape(W, H), k=1)
        ij = _coords.astype(np.int64)
        grid = np.full((W, H), np.nan, dtype=np.float32)
        grid[ij[:, 1], ij[:, 0]] = np.asarray(vec, dtype=np.float32)
        return np.rot90(grid, k=1)
    _idx_cache = {}

    def _flat_index_grid(n):
        """(H, W) grid holding, at each sheet position, the flat index of the unit there.

        Placed through _place, so it inverts whichever placement mode is in force and lets
        a band defined in 2-D sheet space be translated back into per-unit indices.
        Positions with no unit stay NaN.
        """
        if n not in _idx_cache:
            _idx_cache[n] = _place(np.arange(n, dtype=np.float64))
        return _idx_cache[n]

    out_dir = args.out_dir or os.path.join(dirpath, "unified_map")
    os.makedirs(out_dir, exist_ok=True)

    anatomical_constraint = not args.no_anatomical
    filter_out_non_significant = args.filter_nonsig
    overlay = False
    H, W = SHEET.n_row, SHEET.n_col  # unified sheet size

    # ANATOMICAL BANDS, derived from the layout instead of hardcoded. The three regions
    # partition the sheet exactly:
    #     thinker  rows [0, THK)      x all n_col columns
    #     vision   rows [THK, n_row)  x columns [0, MID)
    #     audio    rows [THK, n_row)  x columns [MID, n_col)
    # so THK*n_col + 2*encoder_rows*MID == n_row*n_col. On tiled32:
    # 288*576 + 2*512*288 == 460,800 == 800*576.
    #
    # These were the literals 144 / 256 / 512 / 160, which describe `strip` only -- and not
    # even that one correctly: strip has encoder_rows=165, so the top-k denominator
    # `160 * 256` was 3% short on the very layout it was written for. strip_4096
    # (thinker_rows=288, encoder_rows=528) and tiled32 (288 / 512, n_col=576) were both
    # mis-banded: on tiled32 the literal 144 cut the 288-row thinker band in half, so
    # "vision" kept 144 rows of thinker plus the whole left encoder, and the 256 split fell
    # 32 columns short of the real 288 midline, leaking a strip of audio into vision.
    #
    # Left=vision / right=audio is the original convention and holds on tiled32: of the
    # top-1% units, faces/bodies/scenes put 0.0-0.1% in the right half, speech puts 100%.
    THK = SHEET.thinker_rows
    MID = SHEET.n_col // 2

    p_value_threshold = args.p_threshold
    top_k_pct = args.top_k_pct # top k percent of units to keep

    island_morans_I_results = {}
    _pending_labels = []   # (x, y, text), placed after the loop

    # ASPECT. figsize was hardcoded to (12, 6) -- a 2:1 WIDE canvas -- while the sheet is
    # H x W = n_row x n_col, which is 816 x 512 (1:1.59 TALL) under strip-4096 and
    # 309 x 512 (1.66:1 wide) under strip. sns.heatmap stretches to fill its axes, so the
    # map was being squeezed by a factor of ~2.5 vertically on the 4096 layouts. Island
    # shapes, cluster elongation and the centroid circle below are all read off this
    # figure, so the distortion changes what the plot appears to show. Derive the canvas
    # from the sheet and force square cells instead.
    _base_w = 9.0
    fig, ax = plt.subplots(figsize=(_base_w, max(2.5, _base_w * H / W)), dpi=300)

    if args.faded_background:
        # Drawn ONCE, before the loop, hence underneath every category. The disabled
        # per-category version further down did this inside the loop, so the backdrop
        # gained a translucent layer per category and its shade depended on how many were
        # requested. Category cells that are NaN stay masked by pcolormesh and are never
        # painted, so this grey shows through wherever a unit was not selected.
        sns.heatmap(
            np.ones((H, W)),
            ax=ax,
            cmap=mcolors.ListedColormap(["#B0B0B0"]),
            cbar=False,
            alpha=0.30,
            square=True,
            linewidths=0,
        )

    for category, modality, cmap_key in selectivity:

        print(f"> Visualizing selectivity for category: {category}, modality: {modality}")

        # cmap = cmaps[modality]
        if args.stats_suffix is not None:
            filepath = f"{dirpath}/{category}/{category}{args.stats_suffix}_selectivity_stats.pkl"
            if not os.path.exists(filepath):
                raise SystemExit(f"no stats for {category}: {filepath}")
        else:
            filepath = _guess_stats_path(dirpath, category)
        stats = read_pickle(filepath)

        t_values = stats['t']  # shape: (num_units, )
        if filter_out_non_significant:
            t_values[(stats["q"] >= p_value_threshold) | (stats["t"] <= 0)] = np.nan
        # t_values[stats["t"] <= 0] = np.nan

        t_flat = np.asarray(t_values, dtype=float).copy()   # per-unit, pre-placement
        t_values = _place(t_values)

        t_values_copy = t_values.copy()

        p_values = _place(stats['q'])

        band = np.ones((H, W), dtype=bool)
        if anatomical_constraint:
            band[:] = False
            if modality in ["language", "cognitive"]:
                band[:THK, :] = True
            elif modality == "audio":
                band[THK:, MID:] = True
            elif modality == "vision":
                band[THK:, :MID] = True
            else:
                raise SystemExit(f"no anatomical band defined for modality {modality!r}")
            t_values[~band] = np.nan

        if np.isnan(t_values).any():
            t_values = np.nan_to_num(t_values, nan=-np.inf)

        t_values = t_values.flatten()

        if anatomical_constraint:
            if modality in ["language", "cognitive"]:
                total_num_units = THK * SHEET.n_col
            elif modality in ("audio", "vision"):
                total_num_units = SHEET.encoder_rows * MID
            else:
                raise SystemExit(f"no anatomical band defined for modality {modality!r}")
        else:
            total_num_units = H * W

    
        if top_k_pct <= 0 or top_k_pct > 100:
            selectivity_mask = (stats["q"] < p_value_threshold) & (stats["t"] > 0)
            selectivity_mask = _place(selectivity_mask)
        else:

            active_num_units = int((top_k_pct / 100) * total_num_units)

            # Rank on the FLAT per-unit vector, then place. This used to argsort the
            # already-2-D `t_values`, which numpy sorts along the LAST AXIS -- so it
            # ranked within each sheet row independently and then used those 2-D indices
            # to set whole ROWS True. The mask it produced was rows, not top units.
            # Restrict the ranking to the band. The nan-ing above touched only the PLACED
            # 2-D copy; `t_flat` was snapshotted before placement and never saw it. So the
            # selection was the top-k over the WHOLE sheet while being reported against a
            # per-band denominator -- with --top-k-pct 1 on tiled32 that picked 1,474 units
            # ranked over all 460,800 and called them "top 1% of 147,456". Out-of-band
            # picks were blanked at draw time, so the figure looked right while the island
            # statistics and the centroid circle were computed over them -- which is why
            # circles appeared in the empty thinker band.
            allowed = np.ones(t_flat.size, dtype=bool)
            if anatomical_constraint:
                _g = _flat_index_grid(t_flat.size)[band]
                allowed[:] = False
                allowed[_g[np.isfinite(_g)].astype(np.int64)] = True
            t_rank = np.where(np.isfinite(t_flat) & allowed, t_flat, -np.inf)
            top_idx = np.argsort(t_rank)[::-1][:active_num_units]
            flat_mask = np.zeros_like(t_rank, dtype=bool)
            flat_mask[top_idx] = True
            n_real = int(np.isfinite(t_flat[top_idx]).sum())
            print(f"  top {top_k_pct}% -> {active_num_units:,} of {total_num_units:,} units "
                  f"({n_real:,} with a finite t)")
            selectivity_mask = _place(flat_mask)


        t_values = t_values.reshape(H, W)

        # top_k_pct_mask = np.rot90(top_k_pct_mask, k=1)
        # top_k_pct_mask = t_values >= np.percentile(t_values, 100 - top_k_pct)
        # t_values[~top_k_pct_mask] = np.nan
        # t_values[(stats["q"] >= p_value_threshold) | (stats["t"] < 0)] = np.nan

        if args.morph_clean:
            mask_clean = binary_opening(selectivity_mask, structure=np.ones((3, 3)))
            mask_clean = remove_small_components(mask_clean, min_size=20)
        else:
            mask_clean = selectivity_mask.astype(bool)

        mask_labeled, num_islands = label_islands(mask_clean.astype(np.uint8), connectivity=8)
        stats_islands = island_stats(mask_labeled, num_islands, t_values)
        print_stats(stats_islands)
        
        # island_stats() returns islands sorted by size descending, so the first N are the
        # N largest. --n-islands wins over --largest-island-only when both are given.
        _n_keep = args.n_islands if args.n_islands else (1 if args.largest_island_only else None)
        if args.island_rank == "corrected" and stats_islands:
            _sp = f"{dirpath}/{category}/{category}{args.stats_suffix or ''}_island_null.json"
            with open(_sp) as _f:
                _sc = {d["id"]: d for d in json.load(_f)["islands"]}
            for d in stats_islands:
                s_ = _sc.get(int(d["id"]))
                # the scores must describe THIS map's islands (same stats, top-k, cleaning)
                if s_ is None or s_["size"] != d["size"]:
                    raise SystemExit(f"{category}: island {d['id']} (size {d['size']}) does "
                                     f"not match {_sp}; regenerate it for these settings")
                d["corrected_mass"] = s_["corrected_mass"]
            stats_islands.sort(key=lambda d: d["corrected_mass"], reverse=True)
            print("  ranked by depth-corrected mass: " + ", ".join(
                f"id {d['id']} (n={d['size']}, {d['corrected_mass']:.0f})"
                for d in stats_islands[:_n_keep or 3]))
        kept_islands = stats_islands[:_n_keep] if (stats_islands and _n_keep) else []

        if kept_islands:
            mask_labeled = keep_only_ids(mask_labeled, [d["id"] for d in kept_islands])
            _sizes = ", ".join(str(d["size"]) for d in kept_islands)
            _by = "largest" if args.island_rank == "size" else "highest depth-corrected-mass"
            _what = (f"ONLY the {_by} island" if len(kept_islands) == 1
                     else f"the {len(kept_islands)} {_by} islands")
            print(f"  drawing {_what}: {_sizes} units "
                  f"(of {len(stats_islands)} components)")
        elif stats_islands:
            print(f"  drawing all {len(stats_islands)} components "
                  f"({int(mask_labeled.astype(bool).sum()):,} units)")
        else:
            # No island survives binary_opening(3x3) + min_size=20. Expected at low
            # top-k on a shuffled sheet: the selected units are ~1% of positions and
            # coordinates are permuted within each layer, so solid 3x3 blocks are rare
            # and the opening erases everything. Fall back to drawing the UNOPENED
            # selection so the figure still shows where the top units are -- and say so,
            # because "largest island" statistics do not exist for this panel.
            print(f"  ! no island survived opening(3x3)+min_size=20 at top {top_k_pct}%; "
                  f"drawing the raw selection instead (no island/Moran's I stats)")
            mask_labeled = selectivity_mask.astype(np.uint8)

        # writepath = f"{dirpath}/{category}/selectivity_{category}_t_values_top{top_k_pct}_island=largest_mask.pkl"
        # dump_pickle(mask_labeled, writepath)
        # continue

        # t_values[~mask_labeled.astype(bool)] = -np.inf

        new_mask = mask_labeled

        # create a new mask with the top k t_values
        # t_values_indices_sorted = np.argsort(t_values.flatten())[::-1]
        # top_k_pct_indices = t_values_indices_sorted[:5]
        # new_mask = np.zeros_like(t_values.flatten(), dtype=bool)
        # new_mask[top_k_pct_indices] = True
        # new_mask = new_mask.reshape(H, W)

        p_values[~new_mask.astype(bool)] = np.inf

        if stats_islands:
            island_morans_I_result = island_morans_I(p_map=p_values, t_map=t_values, p_threshold=p_value_threshold)
            print(f"Island Moran's I: {island_morans_I_result['average_moran_I']:.3f} | Significant Islands: {island_morans_I_result['num_significant_components']}/{island_morans_I_result['num_components']}")
        else:
            island_morans_I_result = {"average_moran_I": float("nan"),
                                      "num_significant_components": 0, "num_components": 0}

        # if filter_out_non_significant:
        t_values[~new_mask.astype(bool)] = np.nan
        t_values[t_values == -np.inf] = np.nan
            

        island_morans_I_results[category] = island_morans_I_result

        t_values[t_values == -np.inf] = np.nan
  
        # if anatomical_constraint:
        #     if modality in ["language", "cognitive"]:
        #         t_values[144:, :] = t_values_copy[144:, :]
        #     elif modality == "audio":
        #         t_values[:144, :] = t_values_copy[:144, :]
        #         t_values[144:, :256] = t_values_copy[144:, :256]
        #     elif modality == "vision":
        #         t_values[:144, :] = t_values_copy[:144, :]
        #         t_values[144:, 256:] = t_values_copy[144:, 256:]

        # t_values[~top_k_pct_mask] = np.nan
        # if modality == "vision" and anatomical_constraint:
        #     t_values = t_values[144:, :256] 
        # elif modality == "audio" and anatomical_constraint:
        #     t_values = t_values[144:, 256:]

        # rotate grid_lm_mask 90 degrees to the left
        
        plot_overlay = np.where(np.isnan(t_values), np.nan, 1.0)
        sns.heatmap(
            plot_overlay,
            cbar=False,
            cmap=cmaps[cmap_key],
            ax=ax,
            square=True,          # square cells: without this the sheet is stretched
            linewidths=0,
            linecolor=None
        )

        # add a dashed circle around the centroid of the largest island -- only if one
        # exists. With no surviving island there is no "largest", and drawing a circle
        # anywhere would assert a cluster the data does not contain.
        # One circle per DRAWN island. Circling only stats_islands[0] while the panel
        # shows N of them would point at one cluster and leave the others unmarked.
        for _isl in (kept_islands or stats_islands[:1]):
            centroid = _isl["centroid"]  # (row, col)
            circle = plt.Circle((centroid[1], centroid[0]), radius=20,
                                color=colors[conditions_labels.index(cmap_key)],
                                fill=False, linestyle='--', linewidth=1.5)
            ax.add_patch(circle)

        if args.label_clusters:
            # Anchor on the DRAWN mask, not the island bbox: with --largest-island-only the
            # two agree, but without it the bbox describes only the largest component while
            # the figure shows all of them.
            _r, _c = np.nonzero(np.asarray(new_mask).astype(bool))
            if _r.size:
                _pending_labels.append(
                    (float(_c.mean()), float(_r.max()) + args.label_pad,
                     CLUSTER_NAMES.get(category, category.replace("_", " ").title())))

        # if overlay and not filter_out_non_significant:
        #     gray_overlay = np.where(~new_mask.astype(bool), 1.0, np.nan)

        #     sns.heatmap(
        #         gray_overlay,
        #         ax=ax,
        #         cmap=plt.cm.Greys,
        #         vmin=0, vmax=1,
        #         cbar=False,
        #         alpha=0.35,
        #         linewidths=0
        #     )

        # if not anatomical_constraint:
        # add horizontal dashed line at y = 160
        ax.axhline(y=SHEET.thinker_rows, color='gray', linestyle='--')

        # add vertical dashed line at x = 256 until y = 160
        ax.vlines(x=SHEET.n_col // 2, ymin=SHEET.thinker_rows, ymax=SHEET.n_row, color='gray', linestyle='--')

        # remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])

    if _pending_labels:
        # Nudge overlapping labels downward. Adjacent clusters (Language/Multiple Demand,
        # Scenes/Bodies) otherwise print their names on top of each other. Width is
        # estimated from the axes scale: a bold char is roughly 0.62 em wide.
        fig.canvas.draw()
        _bb = ax.get_window_extent(renderer=fig.canvas.get_renderer())
        _pts_per_col = (_bb.width * 72.0 / fig.dpi) / max(W, 1)
        _char = 0.62 * args.label_size / max(_pts_per_col, 1e-6)     # sheet px per character
        _line = 1.35 * args.label_size / max(_pts_per_col, 1e-6)
        _placed = []
        for _x, _y, _t in sorted(_pending_labels, key=lambda z: (z[1], z[0])):
            _hw = 0.5 * _char * len(_t)
            for _ in range(40):
                if not any(abs(_y - py) < _line and (_x - _hw) < (px + phw) and (px - phw) < (_x + _hw)
                           for px, py, phw in _placed):
                    break
                _y += _line
            _placed.append((_x, _y, _hw))
            ax.text(_x, _y, _t, ha="center", va="top", fontsize=args.label_size,
                    fontweight="bold", color="black", zorder=8)

    # Name the categories drawn whenever --categories selects a subset. Without this every
    # per-category run writes the same "unified_map" filename, so a loop over the 10
    # categories silently overwrites one file 10 times and leaves only the last.
    _cat_tag = ""
    if args.categories:
        _cat_tag = "_cat=" + "-".join(t[0] for t in selectivity)
    _islands_tag = (f"_islands{_n_keep}" + ("corrected" if args.island_rank == "corrected" else "")
                    if _n_keep else "")

    if top_k_pct <= 0 or top_k_pct > 100:
        savepath = f"{out_dir}/selectivity_unified_map_t_values_p{p_value_threshold}_anatomical={anatomical_constraint}_filternonsig={filter_out_non_significant}{_islands_tag}{_cat_tag}"
    else:
        savepath = f"{out_dir}/selectivity_unified_map_t_values_top{top_k_pct}_anatomical={anatomical_constraint}_filternonsig={filter_out_non_significant}{_islands_tag}{_cat_tag}"

    if not args.no_svg:
        fig.savefig(f"{savepath}.svg", format="svg", bbox_inches="tight")
        print(f"> saved {savepath}.svg")
    fig.savefig(f"{savepath}.png", dpi=300, bbox_inches="tight")
    print(f"> saved {savepath}.png")
    plt.cla()
    plt.clf()
    plt.close()