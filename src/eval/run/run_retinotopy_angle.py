"""
Plot retinotopic maps (polar angle) on the Topo-Omni
cortical sheet, masking units that are not significantly angle-tuned.

Inputs
------
pref_ang    : (n_units,) int   preferred angle index per unit
tuned       : (n_units,) bool  True where the unit passed the ANOVA tuning test (FDR-corrected)
ang_deg     : (n_ang,) float   angles in degrees, same index order as pref_ang
sheet_shape : (H, W)           grid case, with H * W == n_units
coords      : (n_units, 2)     scatter case, continuous unit positions
"""

import os
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from scipy.stats import f_oneway, false_discovery_control

from qwen_omni_utils import process_mm_info
from src.core.model_loading import load_topo_omni, unified_grid_coords
from src.utils.smoothing import NeuronSmoothingConv


_ANGLE_TICKS = [0, 60, 120, 180, 240, 300, 360]


def _angle_tick_cbar(cbar, ang_deg):
    """Set colorbar ticks at fixed degree values, mapped to index space."""
    ang_deg = np.asarray(ang_deg, dtype=float)
    tick_idx = [np.argmin(np.abs(ang_deg - d)) for d in _ANGLE_TICKS]
    cbar.set_ticks(tick_idx)
    cbar.set_ticklabels([str(d) for d in _ANGLE_TICKS])
    cbar.set_label("preferred angle (deg)")


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


def plot_angle(pref_ang, tuned, ang_deg, coords, sheet_shape,
               ang_cmap="twilight_shifted", figsize=(7, 6), savepath=None):
    """Grid layout: single-panel masked seaborn heatmap (polar angle)."""
    pref_ang = np.asarray(pref_ang)
    tuned = np.asarray(tuned, dtype=bool)
    ang_deg = np.asarray(ang_deg, dtype=float)
    H, W = tuple(map(int, sheet_shape))

    ang_2d = pref_ang

    assert pref_ang.size == H * W == tuned.size, "sheet_shape does not match n_units"

    tuned = place_on_grid(tuned, coords).astype(bool)

    ang_2d = ang_2d.reshape(W, H).astype(float)
    mask_2d = ~tuned.reshape(W, H)

    ang_2d = np.rot90(ang_2d, k=1)
    mask_2d = np.rot90(mask_2d, k=1)

    ang_2d[144:, 256:] = np.nan

    fig, ax_a = plt.subplots(1, 1, figsize=figsize)
    ax_a.set_facecolor("0.92")
    hm = sns.heatmap(
        ang_2d, mask=mask_2d,
        cmap=plt.get_cmap(ang_cmap, len(ang_deg)),
        vmin=-0.5, vmax=len(ang_deg) - 0.5,
        square=True, linewidths=0, ax=ax_a, cbar=True,
        cbar_kws={"shrink": 0.6},
        xticklabels=False, yticklabels=False,
    )
    _angle_tick_cbar(hm.collections[0].colorbar, ang_deg)
    ax_a.set_title("Polar Angle")
    ax_a.set_xlabel("")
    ax_a.set_ylabel("")

    # fig.suptitle(f"Retinotopy — vision component sheet "
    #              f"({tuned.sum()}/{tuned.size} units tuned)")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig, ax_a


def plot_angle_scatter(pref_ang, tuned, ang_deg, coords,
                       ang_cmap="twilight_shifted", figsize=(7, 6), point_size=8, savepath=None):
    """Continuous layout: single-panel scatter for (x, y) unit positions (polar angle)."""
    pref_ang = np.asarray(pref_ang, dtype=float)
    tuned = np.asarray(tuned, dtype=bool)
    ang_deg = np.asarray(ang_deg, dtype=float)
    coords = np.asarray(coords, dtype=float)

    coords = np.stack([coords[:, 1], coords[:, 0]], axis=1)

    sns.set_style("white")
    fig, ax_a = plt.subplots(1, 1, figsize=figsize)
    ax_a.set_facecolor("0.92")
    ax_a.scatter(coords[~tuned, 0], coords[~tuned, 1],
                 c="0.8", s=point_size * 0.7, linewidths=0)
    sc = ax_a.scatter(coords[tuned, 0], coords[tuned, 1],
                      c=pref_ang[tuned], cmap=plt.get_cmap(ang_cmap, len(ang_deg)),
                      vmin=-0.5, vmax=len(ang_deg) - 0.5, s=point_size, linewidths=0)
    cbar = fig.colorbar(sc, ax=ax_a, shrink=0.6)
    _angle_tick_cbar(cbar, ang_deg)
    ax_a.set_aspect("equal")
    ax_a.set_title("Polar Angle")
    ax_a.set_xlabel("sheet x")
    ax_a.set_ylabel("sheet y")
    ax_a.set_xticks([])
    ax_a.set_yticks([])

    # fig.suptitle(f"Retinotopy — vision component sheet "
    #              f"({tuned.sum()}/{tuned.size} units tuned)")

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig, ax_a


def anova_retinotopy(responses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    responses: (n_ang, n_exemplars, n_units)

    Returns:
        pref_ang : (n_units,) preferred angle index
        tuned    : (n_units,) bool, FDR-corrected at 0.05
    """
    n_ang, _, _ = responses.shape
    mean_resp = responses.mean(axis=1)       # (n_ang, n_units)
    pref_ang = np.argmax(mean_resp, axis=0)  # (n_units,)

    _, p_values = f_oneway(*[responses[ang] for ang in range(n_ang)])
    p_values = np.where(np.isnan(p_values), 1.0, p_values)
    tuned = false_discovery_control(p_values) < 0.05
    return pref_ang, tuned


@torch.no_grad()
def extract_features_for_images(
    model,
    processor,
    image_paths: List[str],
    batch_size: int = 8,
    device: str = "cuda",
):
    """Returns cortical_sheets of shape (N, n_units)."""
    model.eval()
    processor.tokenizer.padding_side = "right"

    cortical_sheets = []

    for i in tqdm(range(0, len(image_paths), batch_size)):
        chunk_paths = image_paths[i:i + batch_size]

        chats = [
            [{"role": "user", "content": [{"type": "image", "image": img}]}]
            for img in chunk_paths
        ]

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
    parser.add_argument("--config", type=str, default="src/configs/eval_retinotopy.yml",
                        help="Path to YAML config for retinotopy analysis.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    run_title        = str(cfg.run.run_title)
    category_name    = str(cfg.run.category_name)
    output_root      = str(cfg.run.output_root)

    model_override   = cfg.model.get("model", None)
    device           = str(cfg.model.device)

    stimuli_root     = Path(cfg.data.stimuli_root).resolve()
    batch_size       = int(cfg.data.batch_size)

    outdir = Path(output_root) / run_title / f"{category_name}_angle"
    outdir.mkdir(parents=True, exist_ok=True)

    # path example: angle_target000.00_actual000.01_eccmean05.34_pattern002_frame0751_mask0421.png
    image_paths = sorted((stimuli_root / "angle").glob("*.png"))
    ang_deg = sorted(set(float(p.stem.split("_")[1].split("target")[-1]) for p in image_paths))
    n_ang = len(ang_deg)
    n_exemplars = len(set(p.stem.split("_")[4].split("pattern")[-1] for p in image_paths))
    image_paths = [str(p) for p in image_paths]

    print(f"> Found {len(image_paths)} stimuli "
          f"({n_ang} angles x {n_exemplars} exemplars)")

    cache_path = outdir / "cortical_sheets.npy"

    if not os.path.exists(cache_path):
        model, processor, _ = load_topo_omni(model=model_override, device=device)

        cortical_sheets = extract_features_for_images(
            model, processor, image_paths,
            batch_size=batch_size, device=device,
        )

        print(f"> Extracted cortical sheets with shape: {cortical_sheets.shape}")
        np.save(cache_path, cortical_sheets)
    else:
        print(f"> Loading cached cortical sheets from: {cache_path}")
        cortical_sheets = np.load(cache_path)

    coords_lm = unified_grid_coords()

    # (N_total, n_units) -> (n_ang, n_exemplars, n_units)
    cortical_sheets = cortical_sheets.reshape(n_ang, n_exemplars, cortical_sheets.shape[-1])
    print(f"> Reshaped cortical sheets to: {cortical_sheets.shape}")

    smoother = NeuronSmoothingConv(fwhm_mm=4.0, resolution_mm=1.0)
    smoothed_sheet = smoother(coords_lm, cortical_sheets.reshape(-1, cortical_sheets.shape[-1]))

    pref_ang, tuned = anova_retinotopy(smoothed_sheet.reshape(n_ang, n_exemplars, -1))

    plot_angle_scatter(
        pref_ang, tuned, ang_deg, coords_lm,
        savepath=outdir / "angle_selectivity_scatter.png",
    )

    plot_angle(
        pref_ang, tuned, ang_deg, coords_lm, sheet_shape=coords_lm.max(axis=0) + 1,
        savepath=outdir / "angle_selectivity_sheet.png",
    )

    print(f"> Saved angle scatter plot to: {outdir / 'angle_selectivity_scatter.png'}")
    print(f"> Saved angle sheet plot to: {outdir / 'angle_selectivity_sheet.png'}")

if __name__ == "__main__":
    main()
