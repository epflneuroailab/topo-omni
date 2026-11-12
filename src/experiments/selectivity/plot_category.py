"""Single-category selectivity map: the selected islands in colour, the rest greyed out.

The per-category counterpart to selectivity/plot.py, which overlays every category on one
unified sheet. Was spacetop_discovery/extract_clusters.py; it is a selectivity plot, not a
spacetop-discovery step, and nothing but a README file-tree line referred to it by the old
name.

Also dumps the drawn mask as a pickle. ablation/run.py --mode cluster reads that pickle at
the "island=largest" name, so the default single-island filename is deliberately unchanged.
"""
import os
import argparse
import json
import numpy as np
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.ndimage import binary_opening

from skimage import measure
from src.utils.connected_components import (label_islands, island_stats, print_stats,
                                            keep_only_id, keep_only_ids)
from src.utils.island_morans_I import island_morans_I
from src.models.spatial_utils import DEFAULT_SHEET_LAYOUT as SHEET

def remove_small_components(mask, min_size):
    labeled = measure.label(mask)
    cleaned = np.zeros_like(mask)
    for region in measure.regionprops(labeled):
        if region.area >= min_size:
            cleaned[labeled == region.label] = 1
    return cleaned

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

if __name__ == "__main__":

    MODALITY = {
        "faces": "vision", "scenes": "vision", "bodies": "vision",
        "vwfa": "vision", "objects": "vision",
        "speech": "audio", "vocals": "audio",
        "fedorenko_words_nonwords": "language",
        "language_text": "language", "language_text_ALL": "language",
        "theory_of_mind_text": "cognitive", "theory_of_mind_text_ALL": "cognitive",
        "multiple_demand_text": "cognitive", "multiple_demand_text_ALL": "cognitive",
    }
    # viridis for every modality: these panels show a continuous t-value field, and one
    # perceptually uniform map keeps them comparable across categories. --cmap still
    # overrides per run. (The unified map in selectivity/plot.py keeps its CATEGORICAL
    # palette -- one hue per category is the whole point there.)
    CMAPS = {"vision": "viridis", "audio": "viridis",
             "language": "viridis", "cognitive": "viridis", "all": "viridis"}

    parser = argparse.ArgumentParser(
        description="Single-category t-value map: the top-k%% islands in colour, everything "
                    "else greyed out. Also dumps the drawn mask as a pickle.")
    parser.add_argument("--results-dir",
                        default=f"{os.getenv('SAVE_DIR', 'results')}/topo-omni",
                        help="directory holding <category>/<category><suffix>_selectivity_stats.pkl")
    parser.add_argument("--category", default="faces",
                        help="the one category to draw: a localizer in MODALITY, or any "
                             "other stats directory name when --modality is given (e.g. "
                             "the spacetop_cluster_<id> maps from "
                             "spacetop_discovery/cluster_maps.py)")
    parser.add_argument("--modality", default=None,
                        choices=["vision", "audio", "language", "cognitive", "all"],
                        help="anatomical band to rank top-k in. Default: the category's "
                             "entry in MODALITY. 'all' ranks over the whole sheet, for "
                             "audiovisual contrasts that are not tied to one band.")
    parser.add_argument("--stats-suffix", default="_even",
                        help="stats filename suffix: <cat><suffix>_selectivity_stats.pkl. "
                             "Default '_even' is the odd/even-split sweep. A SMOOTHED "
                             "sweep writes '_fwhm<f>' instead -- run_selectivity appends "
                             "it to category_name whenever smoothing is on.")
    parser.add_argument("--top-k-pct", type=float, default=1.0,
                        help="top-k%% most selective units to keep (<=0 or >100 falls back "
                             "to the q<p-threshold significance mask)")
    parser.add_argument("--p-threshold", type=float, default=0.001, help="FDR q threshold")
    parser.add_argument("--no-anatomical", action="store_true",
                        help="rank over the whole sheet instead of the modality band")
    parser.add_argument("--no-overlay", action="store_true",
                        help="drop the grey wash over the non-selected units")
    parser.add_argument("--cmap", default=None,
                        help="matplotlib colormap for the t-values. Default: the "
                             "per-modality entry in CMAPS above.")
    parser.add_argument("--no-morph-clean", action="store_true",
                        help="skip binary_opening(3x3)+remove_small_components(20). The "
                             "opening needs solid 3x3 blocks, so at low top-k it deletes "
                             "most of the selection rather than cleaning it.")
    parser.add_argument("--all-islands", action="store_true",
                        help="draw every surviving component, not only the largest")
    parser.add_argument("--n-islands", type=int, default=None,
                        help="draw the N largest components instead of just one. Overrides "
                             "--all-islands. A category with fewer than N surviving islands "
                             "draws all it has.")
    parser.add_argument("--island-scores", default=None,
                        help="island-score JSON for --island-rank corrected, written by "
                             "spacetop_discovery/cluster_null.py for the same stats file")
    parser.add_argument("--island-rank", default="size", choices=["size", "mass", "corrected"],
                        help="which islands --n-islands keeps. 'size' (default): the N "
                             "largest. 'mass': the N with the largest cluster mass, i.e. "
                             "size x mean t = the summed t over the island, so an island "
                             "must be both large and strongly selective to rank high. "
                             "'corrected': mass with size divided by the chance-island size "
                             "at that depth (--island-scores), so deep layers' larger "
                             "spatial-loss radius does not win by itself.")
    parser.add_argument("--no-svg", action="store_true", help="write only the PNG")
    parser.add_argument("--out-dir", default=None,
                        help="where to write (default: <results-dir>/<category>)")
    args = parser.parse_args()

    modality = args.modality or MODALITY.get(args.category)
    if modality is None:
        parser.error(f"--category {args.category!r} has no default modality; pass --modality "
                     f"(known localizers: {', '.join(sorted(MODALITY))})")
    selectivity = [(args.category, modality, args.cmap or CMAPS[modality])]

    dirpath = args.results_dir

    anatomical_constraint = not args.no_anatomical
    filter_out_non_significant = False
    overlay = not args.no_overlay
    H, W = SHEET.n_row, SHEET.n_col  # unified sheet size

    # ANATOMICAL BANDS derived from the layout. The literals 144 / 256 below described
    # `strip` only; on tiled32 (thinker_rows=288, n_col=576) the 144 cut the thinker band
    # in half and the 256 split fell 32 columns short of the real 288 midline. Note the
    # total_num_units block further down ALREADY derived from SHEET, so the two halves of
    # this file disagreed with each other.
    THK = SHEET.thinker_rows
    MID = SHEET.n_col // 2

    cmaps = {
        "vision": "Blues_r",
        "audio": "PuBu_r",
        "language": "GnBu_r",
        "cognitive": "GnBu_r"
    }

    p_value_threshold = args.p_threshold
    top_k_pct = args.top_k_pct

    island_morans_I_results = {}

    for category, modality, cmap in selectivity:

        print(f"> Visualizing selectivity for category: {category}, modality: {modality}")

        # ASPECT. figsize was hardcoded to (12, 6) -- a 2:1 WIDE canvas -- while the
        # sheet is H x W = n_row x n_col, which is 800 x 576 (TALL) on tiled32 and
        # 816 x 512 on strip_4096. sns.heatmap stretches to fill its axes, so the map
        # was squeezed ~2.8x horizontally. Island shape and elongation are read off
        # this figure, so the distortion changes what the plot appears to show.
        # src/experiments/selectivity/plot.py already carries this same fix.
        _base_w = 9.0
        fig, ax = plt.subplots(figsize=(_base_w, max(2.5, _base_w * H / W)), dpi=300)
        filepath = f"{dirpath}/{category}/{category}{args.stats_suffix}_selectivity_stats.pkl"
        if not os.path.exists(filepath):
            raise SystemExit(f"no stats for {category}: {filepath}")
        stats = read_pickle(filepath)

        t_values = stats['t']  # shape: (num_units, )
        if filter_out_non_significant:
            t_values[(stats["q"] >= p_value_threshold) | (stats["t"] <= 0)] = np.nan

        t_values = t_values.reshape(W, H)
        t_values = np.rot90(t_values, k=1)

        t_values_copy = t_values.copy()

        p_values = stats['q'].reshape(W, H)
        p_values = np.rot90(p_values, k=1)

        band = np.ones((H, W), dtype=bool)
        if anatomical_constraint:
            band[:] = False
            if modality in ["language", "cognitive"]:
                band[:THK, :] = True
            elif modality == "audio":
                band[THK:, MID:] = True
            elif modality == "vision":
                band[THK:, :MID] = True
            elif modality == "all":
                band[:] = True
            else:
                raise SystemExit(f"no anatomical band for modality {modality!r}")
            t_values[~band] = np.nan

        if np.isnan(t_values).any():
            t_values = np.nan_to_num(t_values, nan=-np.inf)

        t_values = t_values.flatten()

        if anatomical_constraint:
            if modality in ["language", "cognitive"]:
                total_num_units = SHEET.thinker_rows * SHEET.n_col
            elif modality == "audio":
                total_num_units = SHEET.encoder_rows * (SHEET.n_col // 2)
            elif modality == "vision":
                total_num_units = SHEET.encoder_rows * (SHEET.n_col // 2)
            elif modality == "all":
                total_num_units = H * W
        else:
            total_num_units = H * W

        if top_k_pct <= 0 or top_k_pct > 100:
            selectivity_mask = (stats["q"] < p_value_threshold) & (stats["t"] > 0)
            selectivity_mask = selectivity_mask.reshape(W, H)
            selectivity_mask = np.rot90(selectivity_mask, k=1)
        else:
            active_num_units = (top_k_pct / 100) * total_num_units

            t_values_indices_sorted = np.argsort(t_values)[::-1]
            top_k_pct_indices = t_values_indices_sorted[:int(active_num_units)]
            top_k_pct_mask = np.zeros_like(t_values, dtype=bool)
            top_k_pct_mask[top_k_pct_indices] = True

            selectivity_mask = top_k_pct_mask.reshape(H, W)

        t_values = t_values.reshape(H, W)

        # top_k_pct_mask = np.rot90(top_k_pct_mask, k=1)
        # top_k_pct_mask = t_values >= np.percentile(t_values, 100 - top_k_pct)
        # t_values[~top_k_pct_mask] = np.nan
        # t_values[(stats["q"] >= p_value_threshold) | (stats["t"] < 0)] = np.nan

        if args.no_morph_clean:
            mask_clean = selectivity_mask.astype(bool)
        else:
            mask_clean = binary_opening(selectivity_mask, structure=np.ones((3, 3)))
            mask_clean = remove_small_components(mask_clean, min_size=20)

        mask_labeled, num_islands = label_islands(mask_clean.astype(np.uint8), connectivity=8)
        stats_islands = island_stats(mask_labeled, num_islands, t_values)
        print_stats(stats_islands)
        
        if not stats_islands:
            raise SystemExit(
                f"no island survived binary_opening(3x3)+min_size=20 for {category} at "
                f"top {top_k_pct}%. The opening needs solid 3x3 blocks; raise --top-k-pct "
                f"or drop the morphological clean.")
        # island_stats() sorts by size descending, so the first N are the N largest.
        # --n-islands wins over --all-islands.
        if args.island_rank == "corrected":
            if not args.island_scores:
                parser.error("--island-rank corrected needs --island-scores")
            with open(args.island_scores) as _f:
                _sc = {d["id"]: d for d in json.load(_f)["islands"]}
            for d in stats_islands:
                s_ = _sc.get(int(d["id"]))
                # the JSON must describe THIS map's islands, not a re-run with other settings
                if s_ is None or s_["size"] != d["size"]:
                    raise SystemExit(f"island {d['id']} (size {d['size']}) does not match "
                                     f"{args.island_scores}; regenerate it for this map")
                d["corrected_mass"], d["p_fwe"] = s_["corrected_mass"], s_["p_fwe"]
            stats_islands.sort(key=lambda d: d["corrected_mass"], reverse=True)
            print("  ranked by depth-corrected mass:")
            for r, d in enumerate(stats_islands[:max(args.n_islands or 1, 5)], 1):
                print(f"    #{r}: id {d['id']}  size {d['size']}  mean t "
                      f"{d['average_t_values']:.2f}  corrected mass {d['corrected_mass']:.1f}  "
                      f"p_FWE {d['p_fwe']:.3f}")
        if args.island_rank == "mass":
            for d in stats_islands:
                d["mass"] = float(d["size"] * d["average_t_values"])
            stats_islands.sort(key=lambda d: d["mass"], reverse=True)
            print("  ranked by cluster mass (size x mean t):")
            for r, d in enumerate(stats_islands[:max(args.n_islands or 1, 5)], 1):
                print(f"    #{r}: id {d['id']}  size {d['size']}  mean t "
                      f"{d['average_t_values']:.2f}  mass {d['mass']:.0f}  centroid "
                      f"({d['centroid'][0]:.0f},{d['centroid'][1]:.0f})")
        _n_keep = args.n_islands if args.n_islands else (None if args.all_islands else 1)
        if _n_keep:
            kept = stats_islands[:_n_keep]
            mask_labeled = keep_only_ids(mask_labeled, [d["id"] for d in kept])
            _by = {"size": "largest", "mass": "highest-mass",
                   "corrected": "highest depth-corrected-mass"}[args.island_rank]
            _what = (f"ONLY the {_by} island" if len(kept) == 1
                     else f"the {len(kept)} {_by} islands")
            print(f"  drawing {_what}: {', '.join(str(d['size']) for d in kept)} units "
                  f"(of {len(stats_islands)} components)")
        else:
            print(f"  drawing all {len(stats_islands)} components "
                  f"({int(mask_labeled.astype(bool).sum()):,} units)")

        # "island=largest" is the name ablation/run.py --mode cluster looks for, so the
        # single-island default keeps it. Other counts get their own name rather than
        # overwriting that pickle with a different mask under the same label.
        _isl_name = "largest" if _n_keep == 1 else (f"top{_n_keep}" if _n_keep else "all")
        if args.island_rank != "size" and _n_keep:
            # its own name: a mass-ranked mask must not overwrite the size-ranked one that
            # ablation/run.py and earlier figures read
            _isl_name = f"top{_n_keep}{args.island_rank}"
        writepath = f"{dirpath}/{category}/cluster_selectivity_{category}_t_values_top{top_k_pct}_island={_isl_name}_mask.pkl"
        dump_pickle(mask_labeled, writepath)

        # create a new mask with the top k t_values
        # t_values_indices_sorted = np.argsort(t_values.flatten())[::-1]
        # top_k_pct_indices = t_values_indices_sorted[:5]
        # new_mask = np.zeros_like(t_values.flatten(), dtype=bool)
        # new_mask[top_k_pct_indices] = True
        # new_mask = new_mask.reshape(H, W)

        p_values[~mask_labeled.astype(bool)] = np.inf

        island_morans_I_result = island_morans_I(p_map=p_values, t_map=t_values, p_threshold=p_value_threshold)
        print(f"Island Moran's I: {island_morans_I_result['average_moran_I']:.3f} | Significant Islands: {island_morans_I_result['num_significant_components']}/{island_morans_I_result['num_components']}")

        if filter_out_non_significant:
            t_values[~mask_labeled.astype(bool)] = np.nan
            t_values[t_values == -np.inf] = np.nan
            
        island_morans_I_results[category] = island_morans_I_result

        t_values[t_values == -np.inf] = np.nan
  
        if anatomical_constraint:
            # put the out-of-band t-values back so the whole sheet is drawn; the grey
            # overlay below is what marks them as not-selected.
            t_values[~band] = t_values_copy[~band]

        # t_values[~top_k_pct_mask] = np.nan
        # if modality == "vision" and anatomical_constraint:
        #     t_values = t_values[144:, :256] 
        # elif modality == "audio" and anatomical_constraint:
        #     t_values = t_values[144:, 256:]

        # cbar=False here; the colorbar is added below off a divider. seaborn's built-in
        # colorbar steals space using the axes position BEFORE square=True applies the
        # aspect, so it spanned the full figure height while the sheet shrank to fit --
        # the bar ran well above and below the map.
        sns.heatmap(t_values, cbar=False, cmap=cmap, ax=ax, square=True,
                    linewidths=0, linecolor=None)

        if overlay and not filter_out_non_significant:
            gray_overlay = np.where(~mask_labeled.astype(bool), 1.0, np.nan)

            sns.heatmap(
                gray_overlay,
                ax=ax,
                cmap=plt.cm.Greys,
                vmin=0, vmax=1,
                cbar=False,
                alpha=0.35,
                square=True,
                linewidths=0
            )

        # Colorbar matched to the sheet. append_axes sizes the new axes off the parent's
        # box, which square=True has already locked to the data aspect, so the bar ends up
        # exactly as tall as the map. collections[0] is the t-value mesh; collections[1],
        # when present, is the grey wash, whose 0-1 scale must not drive the bar.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.15)
        fig.colorbar(ax.collections[0], cax=cax)

        # if not anatomical_constraint:
        # add horizontal dashed line at the encoder / thinker band boundary
        ax.axhline(y=SHEET.thinker_rows, color='gray', linestyle='--')

        # add vertical dashed line splitting the vision / audio halves of the band
        ax.vlines(x=SHEET.n_col // 2, ymin=SHEET.thinker_rows, ymax=SHEET.n_row, color='gray', linestyle='--')

        # remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])

        _isl_tag = "" if (_n_keep == 1 and args.island_rank == "size") else f"_island={_isl_name}"
        if top_k_pct <= 0 or top_k_pct > 100:
            savepath = f"{dirpath}/{category}/cluster_selectivity_{category}_t_values_p{p_value_threshold}_anatomical={anatomical_constraint}_filternonsig={filter_out_non_significant}{_isl_tag}"
        else:
            savepath = f"{dirpath}/{category}/cluster_selectivity_{category}_t_values_top{top_k_pct}_anatomical={anatomical_constraint}_filternonsig={filter_out_non_significant}{_isl_tag}"

        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            savepath = os.path.join(args.out_dir, os.path.basename(savepath))
        if not args.no_svg:
            fig.savefig(f"{savepath}.svg", format="svg", bbox_inches="tight")
            print(f"> saved {savepath}.svg")
        fig.savefig(f"{savepath}.png", dpi=300, bbox_inches="tight")
        print(f"> saved {savepath}.png")
        plt.cla()
        plt.clf()
        plt.close()