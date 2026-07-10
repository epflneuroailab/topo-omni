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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Visualize selectivity maps for SpaceTop clusters (Fig 2c / 7).")
    parser.add_argument("--cluster_id", type=int, default=32, help="ID of the target cluster to visualize")
    parser.add_argument("--topk", type=int, default=1, help="Percentage of top-k significant units to consider")
    parser.add_argument("--results-dir", default=os.path.join(os.getenv("SAVE_DIR", "results"), MODEL_TITLE, "spacetop_clusters_figures"),
                        help="Directory with per-cluster t-maps (<dir>/<cluster_id>/cluster_<cluster_id>_t_map.npy).")

    args = parser.parse_args()

    selectivity = [str(args.cluster_id).zfill(2)]

    dirpath = args.results_dir

    anatomical_constraint = False
    filter_out_non_significant = False
    overlay = True
    H, W = 304, 512  # unified sheet size

    p_value_threshold = 0.001
    top_k_pct = args.topk  # top k percent of units to keep

    island_morans_I_results = {}

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    for category in selectivity:

        print(f"> Visualizing selectivity for category: {category}")

        # cmap = cmaps[modality]
        filepath = f"{dirpath}/{category}/cluster_{category}_t_map.npy"
        t_values = np.load(filepath)

        # t_values = stats['t']  # shape: (num_units, )
        # if filter_out_non_significant:
        #     t_values[(stats["q"] >= p_value_threshold) | (stats["t"] <= 0)] = np.nan
        # t_values[stats["t"] <= 0] = np.nan

        # t_values = t_values.reshape(W, H)
        t_values = np.rot90(t_values, k=1)

        t_values_copy = t_values.copy()

        if np.isnan(t_values).any():
            t_values = np.nan_to_num(t_values, nan=-np.inf)

        t_values = t_values.flatten()

        # Discovered clusters have no anatomical prior; rank over the whole sheet.
        total_num_units = H * W

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

        # mask_labeled, num_islands = label_islands(mask_clean.astype(np.uint8), connectivity=8)
        # stats_islands = island_stats(mask_labeled, num_islands, t_values)
        # print_stats(stats_islands)
        
        # keep_id_largest = stats_islands[0]["id"]  # largest island
        # mask_labeled = keep_only_id(mask_labeled, keep_id=keep_id_largest)  # keep largest island

        # writepath = f"{dirpath}/{category}/selectivity_{category}_t_values_top{top_k_pct}_island=largest_mask.pkl"
        # dump_pickle(mask_labeled, writepath)
        # continue

        # t_values[~mask_labeled.astype(bool)] = -np.inf

        new_mask = mask_clean

        # create a new mask with the top k t_values
        # t_values_indices_sorted = np.argsort(t_values.flatten())[::-1]
        # top_k_pct_indices = t_values_indices_sorted[:5]
        # new_mask = np.zeros_like(t_values.flatten(), dtype=bool)
        # new_mask[top_k_pct_indices] = True
        # new_mask = new_mask.reshape(H, W)

        # p_values[~new_mask.astype(bool)] = np.inf

        # island_morans_I_result = island_morans_I(p_map=p_values, t_map=t_values, p_threshold=p_value_threshold)
        # print(f"Island Moran's I: {island_morans_I_result['average_moran_I']:.3f} | Significant Islands: {island_morans_I_result['num_significant_components']}/{island_morans_I_result['num_components']}")
        # island_morans_I_results[category] = island_morans_I_result

        if filter_out_non_significant:
            t_values[~new_mask.astype(bool)] = np.nan
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
        sns.heatmap(t_values, cbar=True, cmap="viridis", ax=ax, linewidths=0, linecolor=None)

        # add a dashed circle around the centroid of the largest island
        # centroid = stats_islands[0]["centroid"]  # (row, col)
        # circle = plt.Circle((centroid[1], centroid[0]), radius=20, color=colors[conditions_labels.index(cmap_key)], fill=False, linestyle='--', linewidth=1.5)
        # ax.add_patch(circle)

        if overlay and not filter_out_non_significant:
            gray_overlay = np.where(~new_mask.astype(bool), 1.0, np.nan)

            sns.heatmap(
                gray_overlay,
                ax=ax,
                cmap=plt.cm.Greys,
                vmin=0, vmax=1,
                cbar=False,
                alpha=0.35,
                linewidths=0
            )

        # if not anatomical_constraint:
        # add horizontal dashed line at y = 160
        ax.axhline(y=304 - 160, color='gray', linestyle='--')

        # add vertical dashed line at x = 256 until y = 160
        ax.vlines(x=256, ymin=304 - 160, ymax=304, color='gray', linestyle='--')

        # remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])

    os.makedirs(f"{dirpath}/{category}", exist_ok=True)
    if top_k_pct <= 0 or top_k_pct > 100:
        savepath = f"{dirpath}/{category}/selectivity_{category}_t_values_p{p_value_threshold}_anatomical={anatomical_constraint}_filternonsig={filter_out_non_significant}"
    else:
        savepath = f"{dirpath}/{category}/selectivity_{category}_t_values_top{top_k_pct}_anatomical={anatomical_constraint}_filternonsig={filter_out_non_significant}"

    # fig.savefig(f"{savepath}.svg", format="svg", bbox_inches="tight")
    fig.savefig(f"{savepath}.png", dpi=300, bbox_inches="tight")
    plt.cla()
    plt.clf()
    plt.close()