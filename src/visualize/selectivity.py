import os
import json
import argparse
import numpy as np
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from scipy.ndimage import binary_opening

from skimage import measure
from src.utils.connected_components import label_islands, island_stats, print_stats, keep_only_id
from src.utils.island_morans_I import island_morans_I
from src.core.model_loading import MODEL_TITLE

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
colors = ["#E64B35", "#F39B2F", "#F1C232", "#EFC94C", "#D4AC0D",
     "#5AB4AC", "#4C9F9B", "#3F8F9C", "#3B5B92", "#2F3E75"]

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
    args = parser.parse_args()

    selectivity = [
        ("faces", "vision", "Faces"),
        ("bodies", "vision", "Bodies"),
        ("scenes", "vision", "Scenes"),
        ("objects", "vision", "Objects"),
        ("vwfa", "vision", "Words"),
        ("speech", "audio", "Quilted Speech"),
        ("vocals", "audio", "False Photo"),
        ("language_text", "language", "Nonwords"),
        ("theory_of_mind_text", "cognitive", "False Belief"),
        ("multiple_demand_text", "cognitive", "Math"),

        # ("fedorenko_words_nonwords", "language", "Nonwords"),
        # ("theory_of_mind", "cognitive", "viridis"),
        # ("multi_demand", "cognitive", "viridis"),
    ]

    dirpath = args.results_dir
    out_dir = args.out_dir or os.path.join(dirpath, "unified_map")
    os.makedirs(out_dir, exist_ok=True)

    anatomical_constraint = not args.no_anatomical
    filter_out_non_significant = args.filter_nonsig
    overlay = False
    H, W = 304, 512  # unified sheet size

    p_value_threshold = args.p_threshold
    top_k_pct = args.top_k_pct # top k percent of units to keep

    island_morans_I_results = {}

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    for category, modality, cmap_key in selectivity:

        print(f"> Visualizing selectivity for category: {category}, modality: {modality}")

        # cmap = cmaps[modality]
        filepath = f"{dirpath}/{category}/{category}_all_selectivity_stats.pkl"
        stats = read_pickle(filepath)

        t_values = stats['t']  # shape: (num_units, )
        if filter_out_non_significant:
            t_values[(stats["q"] >= p_value_threshold) | (stats["t"] <= 0)] = np.nan
        # t_values[stats["t"] <= 0] = np.nan

        t_values = t_values.reshape(W, H)
        t_values = np.rot90(t_values, k=1)

        t_values_copy = t_values.copy()

        p_values = stats['q'].reshape(W, H)
        p_values = np.rot90(p_values, k=1)

        if anatomical_constraint:
            if modality in ["language", "cognitive"]:
                t_values[144:, :] = np.nan
            elif modality == "audio":
                t_values[:144, :] = np.nan
                t_values[144:, :256] = np.nan
            elif modality == "vision":
                t_values[:144, :] = np.nan
                t_values[144:, 256:] = np.nan

        if np.isnan(t_values).any():
            t_values = np.nan_to_num(t_values, nan=-np.inf)

        t_values = t_values.flatten()

        if anatomical_constraint:
            if modality in ["language", "cognitive"]:
                total_num_units = 144 * 512
            elif modality == "audio":
                total_num_units = 160 * 256
            elif modality == "vision":
                total_num_units = 160 * 256
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

        mask_clean = binary_opening(selectivity_mask, structure=np.ones((3, 3)))
        mask_clean = remove_small_components(mask_clean, min_size=20)

        mask_labeled, num_islands = label_islands(mask_clean.astype(np.uint8), connectivity=8)
        stats_islands = island_stats(mask_labeled, num_islands, t_values)
        print_stats(stats_islands)
        
        keep_id_largest = stats_islands[0]["id"]  # largest island
        mask_labeled = keep_only_id(mask_labeled, keep_id=keep_id_largest)  # keep largest island

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

        island_morans_I_result = island_morans_I(p_map=p_values, t_map=t_values, p_threshold=p_value_threshold)
        print(f"Island Moran's I: {island_morans_I_result['average_moran_I']:.3f} | Significant Islands: {island_morans_I_result['num_significant_components']}/{island_morans_I_result['num_components']}")

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
            linewidths=0, 
            linecolor=None
        )

        # add a dashed circle around the centroid of the largest island
        centroid = stats_islands[0]["centroid"]  # (row, col)
        circle = plt.Circle((centroid[1], centroid[0]), radius=20, color=colors[conditions_labels.index(cmap_key)], fill=False, linestyle='--', linewidth=1.5)
        ax.add_patch(circle)

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
        ax.axhline(y=304 - 160, color='gray', linestyle='--')

        # add vertical dashed line at x = 256 until y = 160
        ax.vlines(x=256, ymin=304 - 160, ymax=304, color='gray', linestyle='--')

        # remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])

    if top_k_pct <= 0 or top_k_pct > 100:
        savepath = f"{out_dir}/selectivity_unified_map_t_values_p{p_value_threshold}_anatomical={anatomical_constraint}_filternonsig={filter_out_non_significant}"
    else:
        savepath = f"{out_dir}/selectivity_unified_map_t_values_top{top_k_pct}_anatomical={anatomical_constraint}_filternonsig={filter_out_non_significant}"

    fig.savefig(f"{savepath}.svg", format="svg", bbox_inches="tight")
    fig.savefig(f"{savepath}.png", dpi=300, bbox_inches="tight")
    plt.cla()
    plt.clf()
    plt.close()