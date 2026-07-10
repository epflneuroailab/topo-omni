import os
import json
import numpy as np
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from scipy.ndimage import binary_opening

from skimage import measure
from src.utils.connected_components import label_islands, island_stats, print_stats, keep_only_id
from src.utils.island_morans_I import island_morans_I

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

def retrieve_t_map(filepath, 
    p_value_threshold=0.001, 
    anatomical_constraint=False, 
    modality=None, 
    top_k_pct=0
):

    stats = read_pickle(filepath)

    t_values = stats['t'].copy()  # shape: (num_units, )

    t_values = t_values.reshape(W, H)
    t_values = np.rot90(t_values, k=1)

    p_values = stats['q'].copy().reshape(W, H)
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

    # t_values = t_values.flatten()

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
        selectivity_mask = (p_values < p_value_threshold) & (t_values > 0)
    else:

        active_num_units = (top_k_pct / 100) * total_num_units

        t_values_indices_sorted = np.argsort(t_values)[::-1]
        top_k_pct_indices = t_values_indices_sorted[:int(active_num_units)]
        top_k_pct_mask = np.zeros_like(t_values, dtype=bool)
        top_k_pct_mask[top_k_pct_indices] = True

        selectivity_mask = top_k_pct_mask.reshape(H, W)

    # t_values = t_values.reshape(H, W)

    mask_clean = binary_opening(selectivity_mask, structure=np.ones((3, 3)))
    mask_clean = remove_small_components(mask_clean, min_size=20)

    p_values[~mask_clean.astype(bool)] = np.inf

    # island_morans_I_result = island_morans_I(p_map=p_values, t_map=t_values, p_threshold=p_value_threshold)
    # print(f"Island Moran's I: {island_morans_I_result['average_moran_I']:.3f} | Significant Islands: {island_morans_I_result['num_significant_components']}/{island_morans_I_result['num_components']}")

    t_values[~mask_clean.astype(bool)] = np.nan        
    t_values[t_values == -np.inf] = np.nan

    return t_values

# -----------------------------
# Color palettes
# -----------------------------
colors = ["#E64B35", "#F39B2F", "#F1C232", "#EFC94C", "#D4AC0D",
     "#5AB4AC", "#4C9F9B", "#3F8F9C", "#3B5B92", "#2F3E75"]

cmaps = {label: mcolors.LinearSegmentedColormap.from_list("custom", [color, "#E7E7E7"]) for color, label in zip(colors, conditions_labels)}

if __name__ == "__main__":

    selectivity = [
        # ("faces", "vision", "Faces"), 
        # ("bodies", "vision", "Bodies"), 
        # ("scenes", "vision", "Scenes"), 
        # ("objects", "vision", "Objects"), 
        # ("vwfa", "vision", "Words"), 
        # ("speech", "audio", "Quilted Speech"), 
        # ("vocals", "audio", "False Photo"), 
        ("language_text_ALL", "language", "Nonwords"),
        ("multiple_demand_text_ALL", "cognitive", "Math"),
        # ("theory_of_mind_text_ALL", "cognitive", "False Belief"),
    ]

    dirpath = os.getenv("SAVE_DIR", "results")

    models = ("topo-omni", "qwen2_5_3b_task_7")

    anatomical_constraint = False
    filter_out_non_significant = False
    overlay = False
    H, W = 304, 512  # unified sheet size
    fwhm_mm = 4.0

    p_value_threshold = 0.05
    top_k_pct = 0 # top k percent of units to keep

    island_morans_I_results = {}

    for category, modality, cmap_key in selectivity:

        print(f"> Visualizing selectivity for category: {category}, modality: {modality}")

        filepath = f"{dirpath}/{models[0]}/{category}/{category}_fwhm{fwhm_mm}_selectivity_stats.pkl"
        topo_tmap = retrieve_t_map(filepath,
            p_value_threshold=p_value_threshold, 
            anatomical_constraint=anatomical_constraint, 
            modality=modality, 
            top_k_pct=top_k_pct
        )

        filepath = f"{dirpath}/{models[1]}/{category}/{category}_fwhm{fwhm_mm}_selectivity_stats.pkl"
        nontopo_tmap = retrieve_t_map(filepath,
            p_value_threshold=p_value_threshold, 
            anatomical_constraint=anatomical_constraint, 
            modality=modality, 
            top_k_pct=top_k_pct
        )

        fig, axes = plt.subplots(1, 2, figsize=(20, 6), dpi=300)

        vmin = np.nanmin([np.nanmin(topo_tmap), np.nanmin(nontopo_tmap)])
        vmax = np.nanmax([np.nanmax(topo_tmap), np.nanmax(nontopo_tmap)])

        cmap = "viridis"

        for ax, tmap, title in zip(axes, [topo_tmap, nontopo_tmap], ["Topo", "Non-Topo"]):
            sns.heatmap(
                tmap,
                cbar=False,
                cmap=cmap,
                ax=ax,
                linewidths=0,
                linecolor=None,
                vmin=vmin,
                vmax=vmax,
            )

            ax.axhline(y=304 - 160, color='gray', linestyle='--')
            ax.vlines(x=256, ymin=304 - 160, ymax=304, color='gray', linestyle='--')
            ax.set_xticks([])
            ax.set_yticks([])
            # ax.set_title(title, fontsize=20)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.02, pad=0.02, label='t-value')

        if top_k_pct <= 0 or top_k_pct > 100:
            savepath = f"{dirpath}/comparison/{category}_selectivity_comparison_t_values_p{p_value_threshold}_anatomical={anatomical_constraint}"
        else:
            savepath = f"{dirpath}/comparison/{category}_selectivity_comparison_t_values_top{top_k_pct}_anatomical={anatomical_constraint}"

        fig.savefig(f"{savepath}.svg", format="svg", bbox_inches="tight")
        fig.savefig(f"{savepath}.png", dpi=300, bbox_inches="tight")
        plt.cla()
        plt.clf()
        plt.close()