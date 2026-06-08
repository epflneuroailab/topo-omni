"""
Plot a tonotopic preferred-frequency map on the Topo-Omni audio cortical sheet,
masking units that are not significantly frequency-tuned.

Inputs
------
preferred   : (n_units,) int   index of preferred frequency per unit (argmax of mean response)
tuned       : (n_units,) bool  True where the unit passed the ANOVA tuning test (FDR-corrected)
freqs_hz    : (n_freqs,) float stimulus frequencies in Hz, same index order as `preferred`
sheet_shape : (H, W)           2D shape of the (layer-concatenated) audio component sheet,
                               with H * W == n_units. Units assumed row-major (C order) --
                               pass order='F' to reshape if your sheet is column-major.
coords      : (n_units, 2)     alternative to sheet_shape: continuous (x, y) unit positions.

`preferred` holds *indices*; because the 30 frequencies are log-spaced, the index axis is
already log-frequency, so a sequential colormap over indices is honest. The colorbar is
relabelled with actual Hz.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def _hz_colorbar(cbar, freqs_hz, n_ticks=6):
    """Relabel an index-based colorbar with actual frequency values."""
    n_freqs = len(freqs_hz)
    tick_idx = np.linspace(0, n_freqs - 1, min(n_ticks, n_freqs)).round().astype(int)
    cbar.set_ticks(tick_idx)
    cbar.set_ticklabels([f"{freqs_hz[i]:.0f}" for i in tick_idx])
    cbar.set_label("preferred frequency (Hz)")


def plot_tonotopy(preferred, tuned, freqs_hz, sheet_shape,
                  cmap="turbo", figsize=(7, 6), savepath=None):
    """Grid layout: reshape the sheet to (H, W) and use a masked seaborn heatmap.

    Untuned units are hidden by the mask and show through as light-grey 'tissue'.
    """
    preferred = np.asarray(preferred)
    tuned = np.asarray(tuned, dtype=bool)
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    n_freqs = len(freqs_hz)
    H, W = sheet_shape
    assert preferred.size == H * W == tuned.size, "sheet_shape does not match n_units"

    pref_2d = preferred.reshape(H, W).astype(float)
    mask_2d = ~tuned.reshape(H, W)                 # seaborn heatmap: True == hidden

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("0.92")                       # untuned units -> light-grey tissue

    # discrete colormap: one bin per frequency, indices map cleanly to colors
    hm = sns.heatmap(
        pref_2d, mask=mask_2d,
        cmap=plt.get_cmap(cmap, n_freqs),
        vmin=-0.5, vmax=n_freqs - 0.5,
        square=True, linewidths=0, ax=ax, cbar=True,
        cbar_kws={"shrink": 0.8},
        xticklabels=False, yticklabels=False,
    )

    _hz_colorbar(hm.collections[0].colorbar, freqs_hz)
    ax.set_title(f"Tonotopy \u2014 audio component sheet "
                 f"({tuned.sum()}/{tuned.size} units tuned)")
    ax.set_xlabel("sheet x")
    ax.set_ylabel("sheet y")
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_tonotopy_scatter(preferred, tuned, freqs_hz, coords,
                          cmap="turbo", figsize=(7, 6), point_size=8, savepath=None):
    """Continuous layout: use when units have (x, y) positions rather than a grid.

    Untuned units are drawn as a faint grey background layer; tuned units are
    colored by preferred frequency.
    """
    preferred = np.asarray(preferred, dtype=float)
    tuned = np.asarray(tuned, dtype=bool)
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    coords = np.asarray(coords, dtype=float)
    n_freqs = len(freqs_hz)

    sns.set_style("white")
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("0.92")

    ax.scatter(coords[~tuned, 0], coords[~tuned, 1],
               c="0.8", s=point_size * 0.7, linewidths=0)
    sc = ax.scatter(coords[tuned, 0], coords[tuned, 1],
                    c=preferred[tuned], cmap=plt.get_cmap(cmap, n_freqs),
                    vmin=-0.5, vmax=n_freqs - 0.5, s=point_size, linewidths=0)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
    _hz_colorbar(cbar, freqs_hz)
    ax.set_aspect("equal")
    ax.set_title(f"Tonotopy \u2014 audio component sheet "
                 f"({tuned.sum()}/{tuned.size} units tuned)")
    ax.set_xlabel("sheet x")
    ax.set_ylabel("sheet y")
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig, ax


if __name__ == "__main__":
    # synthetic demo: smooth gradient + noise so the output reads like a real map
    rng = np.random.default_rng(0)
    H, W = 40, 36
    n_units = H * W
    freqs_hz = np.logspace(np.log10(100), np.log10(8000), 30)

    yy, xx = np.mgrid[0:H, 0:W]
    grad = xx / W + 0.3 * yy / H
    grad = (grad - grad.min()) / (grad.max() - grad.min())
    preferred = np.clip((grad.ravel() * 29 + rng.normal(0, 1.5, n_units)).round(),
                        0, 29).astype(int)
    tuned = rng.random(n_units) < 0.6

    plot_tonotopy(preferred, tuned, freqs_hz, (H, W), savepath="tonotopy_demo.png")
    print("saved tonotopy_demo.png")

# import os
# import numpy as np
# import pickle as pkl
# from glob import glob 

# import seaborn as sns
# import matplotlib.pyplot as plt

# from dotenv import load_dotenv
# load_dotenv()

# SAVE_DIR = os.getenv("SAVE_DIR")

# def read_pickle(path):
#     with open(path, "rb") as f:
#         return pkl.load(f)

# if __name__ == "__main__":

#     N_col = 512
#     N_row = 304

#     topk_pct = 10
#     alpha = 0.001

#     dirpath = f"{SAVE_DIR}/qwen2_5_3b_spatial_task_final_7/tonotopy"
#     paths = glob(f"{dirpath}/stats_*.pkl")
#     paths = sorted(paths, key=lambda x: int(x.split("freq=")[-1].split("Hz")[0]))
#     paths = paths[::3] + [paths[-1]]  # take every 3rd frequency for better visualization

#     cmap = sns.color_palette("viridis", as_cmap=True)
#     cmap_colors = cmap(np.linspace(0, 1, len(paths)))

#     fig, ax = plt.subplots()
#     for idx, freq_path in enumerate(paths):
#         stats = read_pickle(freq_path)
#         freq = freq_path.split("=")[-1].split("Hz")[0]
#         print(f"Frequency: {freq}")
#         grid_t = stats["t"].copy()

#         grid_t[(stats["q"] > alpha) | (stats["t"] <= 0)] = np.nan
    
#         grid_t = grid_t.reshape(N_col, N_row)

#         # rotate grid_lm_mask 90 degrees to the left
#         grid_t = np.rot90(grid_t, k=1)

#         grid_t[:144, :] = np.nan
#         grid_t[144:, :256] = np.nan
#         total_num_units = 160 * 256

#         grid_t = grid_t.flatten()
#         active_num_units = (topk_pct/100) * total_num_units

#         grid_t = np.nan_to_num(grid_t, nan=-np.inf) 

#         t_values_indices_sorted = np.argsort(grid_t)[::-1]
#         top_k_pct_indices = t_values_indices_sorted[:int(active_num_units)]
#         top_k_pct_mask = np.zeros_like(grid_t, dtype=bool)
#         top_k_pct_mask[top_k_pct_indices] = True
#         grid_t[~top_k_pct_mask] = np.nan

#         grid_t = grid_t.reshape(N_row, N_col)
        
#         mask = ~np.isnan(grid_t)
#         rgba = np.zeros((*grid_t.shape, 4))
#         rgba[mask] = cmap_colors[idx]  # solid color with alpha=1
#         ax.imshow(rgba, interpolation='nearest')
        
#         # sns.heatmap(grid_t, cbar=False, color=cmap_colors[idx], ax=ax, linewidths=0, linecolor=None)

#     plt.title(f"Tonotopy Selectivity")
#     plt.axis("off")
#     # add horizontal dashed line at y = 160
#     ax.axhline(y=304 - 160, color='gray', linestyle='--')

#     # add vertical dashed line at x = 256 until y = 160
#     ax.vlines(x=256, ymin=304 - 160, ymax=304, color='gray', linestyle='--')

#     # remove axis ticks
#     ax.set_xticks([])
#     ax.set_yticks([])
#     fig.tight_layout()
#     plt.savefig(f"{dirpath}/tonotopy_selectivity.png", dpi=300, bbox_inches='tight')

