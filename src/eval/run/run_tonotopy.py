import json
import argparse
from pathlib import Path
from typing import List, Tuple

import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from PIL import Image
from tqdm import tqdm
from glob import glob
from omegaconf import OmegaConf

from qwen_omni_utils import process_mm_info
from src.core.model_loading import load_topo_omni, unified_grid_coords
from src.utils.smoothing import NeuronSmoothingConv


def _hz_colorbar(cbar, freqs_hz, n_ticks=6):
    """Relabel an index-based colorbar with actual frequency values."""
    n_freqs = len(freqs_hz)
    tick_idx = np.linspace(0, n_freqs - 1, min(n_ticks, n_freqs)).round().astype(int)
    cbar.set_ticks(tick_idx)
    cbar.set_ticklabels([f"{freqs_hz[i]:.0f}" for i in tick_idx])
    cbar.set_label("preferred frequency (Hz)")

def _real_tick_cbar(cbar, real_values, label, n_ticks=6, fmt="{:.2g}"):
    """Relabel an index-based colorbar with real units (deg, fraction, etc.)."""
    n = len(real_values)
    idx = np.linspace(0, n - 1, min(n_ticks, n)).round().astype(int)
    cbar.set_ticks(idx)
    cbar.set_ticklabels([fmt.format(real_values[i]) for i in idx])
    cbar.set_label(label)

def infer_grid_from_coords(coords: np.array) -> Tuple[int, int]:
    xs = coords[:, 0].astype(np.int64)
    ys = coords[:, 1].astype(np.int64)
    H = int(ys.max()) + 1
    W = int(xs.max()) + 1
    return H, W

def place_on_grid(values: np.ndarray, coords: np.array) -> np.ndarray:
    H, W = infer_grid_from_coords(coords)
    grid = np.full((H, W), np.nan, dtype=np.float32)
    ij = coords.astype(np.int64)
    grid[ij[:, 1], ij[:, 0]] = values
    return grid

def plot_tonotopy(preferred, tuned, freqs_hz, coords, sheet_shape,
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

    tuned = place_on_grid(tuned, coords).astype(bool)

    # pref_2d = place_on_grid(preferred, coords).reshape(W,H).astype(float)  
    pref_2d = preferred.reshape(W, H).astype(float)
    mask_2d = ~tuned.reshape(W, H)                 # seaborn heatmap: True == hidden

    pref_2d = np.rot90(pref_2d, k=1)
    mask_2d = np.rot90(mask_2d, k=1)

    pref_2d[144:, :256] = np.nan

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("0.92")                       # untuned units -> light-grey tissue

    # discrete colormap: one bin per frequency, indices map cleanly to colors
    hm = sns.heatmap(
        pref_2d, mask=mask_2d,
        cmap=plt.get_cmap(cmap, n_freqs),
        vmin=-0.5, vmax=n_freqs - 0.5,
        square=True, linewidths=0, ax=ax, cbar=True,
        cbar_kws={"shrink": 0.6},
        xticklabels=False, yticklabels=False,
    )

    _hz_colorbar(hm.collections[0].colorbar, freqs_hz)
    # ax.set_title(f"Tonotopy \u2014 audio component sheet "
    #              f"({tuned.sum()}/{tuned.size} units tuned)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_tonotopy_column(preferred, tuned, freqs_hz, coords, sheet_shape,
                         cmap="turbo", figsize=(2, 6), savepath=None):
    """Average preferred frequency across columns (ignoring NaN) and plot as a single vertical strip."""
    preferred = np.asarray(preferred)
    tuned = np.asarray(tuned, dtype=bool)
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    n_freqs = len(freqs_hz)
    H, W = sheet_shape

    tuned = place_on_grid(tuned, coords).astype(bool)

    # pref_2d = place_on_grid(preferred, coords).reshape(W, H).astype(float)
    pref_2d = preferred.reshape(W, H).astype(float)
    mask_2d = ~tuned.reshape(W, H)

    pref_2d = np.rot90(pref_2d, k=1)
    mask_2d = np.rot90(mask_2d, k=1)

    pref_2d[144:, :256] = np.nan
    pref_2d[mask_2d] = np.nan

    col = np.nanmedian(pref_2d, axis=1, keepdims=True)  # (H, 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("0.92")
    hm = sns.heatmap(
        col,
        cmap=plt.get_cmap(cmap, n_freqs),
        vmin=-0.5, vmax=n_freqs - 0.5,
        square=False, linewidths=0, ax=ax, cbar=True,
        cbar_kws={"shrink": 0.6},
        xticklabels=False, yticklabels=False,
    )
    _hz_colorbar(hm.collections[0].colorbar, freqs_hz)
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_tonotopy_depth_profile(preferred, tuned, freqs_hz, coords, sheet_shape,
                                figsize=(5, 4), savepath=None):
    """Line plot of mean preferred frequency ± 95% CI along the row (depth) dimension.

    Averages across columns using nanmean, restricted to the audio encoder region
    via the same NaN masking applied in plot_tonotopy.
    """
    preferred = np.asarray(preferred)
    tuned = np.asarray(tuned, dtype=bool)
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    n_freqs = len(freqs_hz)
    H, W = sheet_shape

    tuned = place_on_grid(tuned, coords).astype(bool)

    # pref_2d = place_on_grid(preferred, coords).reshape(W, H).astype(float)
    pref_2d = preferred.reshape(W, H).astype(float)  
    mask_2d = ~tuned.reshape(W, H)

    pref_2d = np.rot90(pref_2d, k=1)
    mask_2d = np.rot90(mask_2d, k=1)

    pref_2d[mask_2d] = np.nan

    # pref_2d = pref_2d[144:, 256:] 
    pref_2d[144:, :256] = np.nan 

    audio_part = pref_2d[144:, 256:].reshape(32, 1280)
    language_part = pref_2d[:144].reshape(36, 2048)

    audio_part_mean = np.nanmean(audio_part, axis=1)[::-1]
    language_part_mean = np.nanmean(language_part, axis=1)[::-1]
    
    row_mean = np.concatenate([audio_part_mean, language_part_mean], axis=0)  # (H,)

    print(f"> pref_2d shape: {pref_2d.shape}, mask_2d shape: {mask_2d.shape}")
    
    # Per-row stats across columns (index space)
    # row_mean = np.nanmean(pref_2d, axis=1)                               # (H,)
    # row_std  = np.nanstd(pref_2d, axis=1)
    # row_n    = np.sum(~np.isnan(pref_2d), axis=1).astype(float)
    # row_n    = np.where(row_n < 2, np.nan, row_n)
    # ci95     = 1.96 * row_std / np.sqrt(row_n)

    depth = np.arange(row_mean.shape[0])
    valid = ~np.isnan(row_mean)
    
    def idx_to_hz(idx):
        return freqs_hz[np.clip(np.round(idx).astype(int), 0, n_freqs - 1)]

    mean_hz = np.where(valid, idx_to_hz(np.nan_to_num(row_mean)), np.nan)
    # lo_hz   = np.where(valid, idx_to_hz(np.nan_to_num(row_mean - ci95)), np.nan)
    # hi_hz   = np.where(valid, idx_to_hz(np.nan_to_num(row_mean + ci95)), np.nan)

    fig, ax = plt.subplots(figsize=figsize)
    # ax.fill_between(depth[valid], lo_hz[valid], hi_hz[valid], alpha=0.25)
    ax.plot(depth[valid], mean_hz[valid], linewidth=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("Depth")
    ax.set_ylabel("Preferred Frequency (Hz)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.set_yticks([200, 800, 3200, 5000, 7000])
    sns.despine(ax=ax)
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

    # rotate coords 90 degrees to the left
    coords = np.dot(coords, np.array([[0, -1], [1, 0]]))

    # flip vertically so that y=0 is at the bottom
    coords[:, 1] = coords[:, 1].max() - coords[:, 1]

    ax.scatter(coords[~tuned, 0], coords[~tuned, 1],
               c="0.8", s=point_size * 0.7, linewidths=0)
    sc = ax.scatter(coords[tuned, 0], coords[tuned, 1],
                    c=preferred[tuned], cmap=plt.get_cmap(cmap, n_freqs),
                    vmin=-0.5, vmax=n_freqs - 0.5, s=point_size, linewidths=0)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
    _hz_colorbar(cbar, freqs_hz)
    ax.set_aspect("equal")
    ax.set_title(f"Tonotopy")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig, ax

freqs_hz = np.logspace(np.log10(100), np.log10(7000), 30)

def anova_tonotopy(responses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    responses: array of shape (n_freqs, n_trials, n_units)

    Returns:
        f_stat: (n_units,)
        p_vals: (n_units,)
    """
    from scipy.stats import f_oneway

    mean_resp = responses.mean(axis=1)                                  # (n_freqs, n_units)
    preferred = np.argmax(mean_resp, axis=0)

    n = mean_resp.shape[0]

    num = (mean_resp.sum(0)) ** 2
    den = n * (mean_resp ** 2).sum(0) + 1e-12
    sparsity = 1 - num / den                            # Treves–Rolls / Vinje–Gallant

    F, p_values = f_oneway(*[responses[f] for f in range(responses.shape[0])])         # one-way ANOVA, per unit
    
    tuned = (p_values < 0.05) #& (sparsity > 0.001)         # adjust thresholds to taste

    return preferred, tuned

def tonotopy(responses: np.ndarray, freqs_hz: np.ndarray, alpha: float = 0.05):
    """
    responses: (n_freqs, n_trials, n_units)
    freqs_hz:  (n_freqs,)  stimulus frequencies in Hz
    """
    from scipy.stats import f_oneway

    log_f = np.log2(freqs_hz)                          # log-frequency axis
    mean_resp = responses.mean(axis=1)                 # (n_freqs, n_units)

    # Rectify so weights are non-negative, then normalize per unit
    w = np.clip(mean_resp, 0, None)
    w = w / (w.sum(axis=0, keepdims=True) + 1e-12)

    # Weighted average best frequency (log-Hz), back to Hz
    bf_log = (w * log_f[:, None]).sum(axis=0)
    bf_hz  = 2.0 ** bf_log

    # Selectivity: sparsity (1 = perfectly peaky, 0 = flat)
    n = mean_resp.shape[0]
    num = (mean_resp.sum(0)) ** 2
    den = n * (mean_resp ** 2).sum(0) + 1e-12
    sparsity = 1 - num / den                            # Treves–Rolls / Vinje–Gallant

    # Tuned units: significant ANOVA across trials, per unit
    # p_values = np.array([
    #     f_oneway(*[responses[f, :, u] for f in range(n)]).pvalue
    #     for u in tqdm(range(responses.shape[2]))
    # ])
    # tuned = (p_values < alpha) & (sparsity > 0.1)         # adjust threshold to taste

    tuned = sparsity > 0.1

    return bf_hz, tuned

import numpy as np
from scipy.stats import ttest_1samp

def dump_pickle(obj, fpath: Path):
    with fpath.open("wb") as f:
        pickle.dump(obj, f)

def ensure_device_dtype(inputs, device):
    """Move all tensors in the inputs dict to the specified device and dtype."""
    dev = torch.device(device)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device=dev, dtype=dtype)
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
            inputs[k] = [t.to(device=dev, dtype=dtype) for t in v]
    return inputs

@torch.no_grad()
def extract_features_for_audio(
    model,
    processor,
    audio_paths: List[str],
    batch_size: int = 8,
    device: str = "cuda",
):
    """
    Returns: feats_lm (N, D_lm) or None, feats_vis (N, D_vis) or None.
    Feeds audio with NO accompanying text. Uses the chat template with <audio> only.
    """
    model.eval()
    processor.tokenizer.padding_side = "right"

    cortical_sheets = []

    for i in tqdm(range(0, len(audio_paths), batch_size)):
        chunk_paths = audio_paths[i:i + batch_size]

        chats = [[{"role": "user", "content": [{"type": "audio", "audio": audio}]}] for audio in chunk_paths]

        text = processor.apply_chat_template(
            chats, 
            add_generation_prompt=False, 
            tokenize=False,
        )

        audios, images, videos = process_mm_info(chats, use_audio_in_video=False)

        inputs = processor(
            text=text, 
            audio=audios, 
            images=images, 
            videos=videos, 
            return_tensors="pt", 
            padding=True, 
            use_audio_in_video=False, 
        ).to(device)

        # inputs = ensure_device_dtype(inputs, device)

        out = model(**inputs, return_dict=True)

        unified_sheet = out.unified_sheet

        unified_sheet = unified_sheet.mean(dim=0)
        unified_sheet = unified_sheet.float().numpy()
        cortical_sheets.append(unified_sheet)

    cortical_sheets = np.stack(cortical_sheets, axis=0).reshape(len(cortical_sheets), -1)
    return cortical_sheets


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/configs/eval_tonotopy.yml",
                        help="Path to YAML config for selectivity analysis.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    run_title     = str(cfg.run.run_title)
    category_name = str(cfg.run.category_name)
    output_root   = str(cfg.run.output_root)
    do_pretrained = bool(cfg.run.get("do_pretrained", False))

    model_override = cfg.model.get("model", None)
    device        = str(cfg.model.device)

    mode          = str(cfg.data.mode).lower()      # "text" or "image"
    stimuli_root  = Path(cfg.data.stimuli_root).resolve()
    batch_size    = int(cfg.data.batch_size)
    lm_reduce     = str(cfg.data.lm_reduce)
    vis_reduce    = str(cfg.data.vis_reduce) if mode == "image" and "vis_reduce" in cfg.data else "mean"

    alpha         = float(cfg.stats.alpha)
    smooth        = bool(cfg.stats.smooth)
    fwhm_mm       = float(cfg.stats.fwhm_mm)
    resolution_mm = float(cfg.stats.resolution_mm)
    topk_pct      = float(cfg.stats.get("topk_pct", 0.0))

    outdir = Path(output_root) / run_title / category_name
    outdir.mkdir(parents=True, exist_ok=True)

    stimuli_paths = sorted(glob(str(stimuli_root / "*.wav")))
    print(f"> Found {len(stimuli_paths)} stimuli in category '{category_name}'")

    cache_path = outdir / f"cortical_sheets.npy"

    if not os.path.exists(cache_path):

        model, processor, _ = load_topo_omni(
            model=model_override, device=device, baseline=do_pretrained
        )

        cortical_sheets = extract_features_for_audio(
            model, processor, stimuli_paths,
            batch_size=batch_size, device=device
        )

        print(f"> Extracted cortical sheets with shape: {cortical_sheets.shape}")
        np.save(cache_path, cortical_sheets)
 
    else:
        print(f"> Loading cached cortical sheets from: {cache_path}")
        cortical_sheets = np.load(cache_path)

    coords_lm = unified_grid_coords()

    cortical_sheets = cortical_sheets.reshape(len(freqs_hz), -1, cortical_sheets.shape[-1])

    print(f"> Loaded cortical sheets with shape: {cortical_sheets.shape}")

    smoother = NeuronSmoothingConv(fwhm_mm=4.0, resolution_mm=1.0)
    smoothed_sheet = smoother(coords_lm, cortical_sheets.reshape(-1, cortical_sheets.shape[-1]))
    
    preferred, tuned = anova_tonotopy(smoothed_sheet.reshape(len(freqs_hz), -1, cortical_sheets.shape[-1]))

    # preferred, tuned = tonotopy(smoothed_sheet.reshape(len(freqs_hz), -1, cortical_sheets.shape[-1]), freqs_hz, alpha=alpha)

    # plot_tonotopy_scatter(
    #     preferred, tuned, freqs_hz, coords_lm,
    #     savepath=outdir / f"tonotopy_selectivity_scatter.png"
    # )

    sheet_shape = coords_lm.max(axis=0).astype(int) + 1

    plot_tonotopy(
        preferred, tuned, freqs_hz, coords_lm, sheet_shape=sheet_shape,
        savepath=outdir / f"tonotopy_selectivity_heatmap.png"
    )

    plot_tonotopy_column(
        preferred, tuned, freqs_hz, coords_lm, sheet_shape=sheet_shape,
        savepath=outdir / f"tonotopy_selectivity_column.png"
    )

    plot_tonotopy_depth_profile(
        preferred, tuned, freqs_hz, coords_lm, sheet_shape=sheet_shape,
        savepath=outdir / f"tonotopy_selectivity_depth_profile.png"
    )
    

if __name__ == "__main__":
    main()