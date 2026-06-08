"""
Plot retinotopic maps (eccentricity + polar angle) on the Topo-Omni vision
cortical sheet, masking units that are not significantly position-tuned.

Inputs
------
pref_ecc    : (n_units,) int   preferred eccentricity index per unit
pref_ang    : (n_units,) int   preferred polar-angle index per unit
tuned       : (n_units,) bool  True where the unit passed the ANOVA tuning test (FDR-corrected)
ecc_frac    : (n_ecc,) float   eccentricities (fraction of image radius), same index order as pref_ecc
ang_deg     : (n_ang,) float   polar angles in degrees, same index order as pref_ang
sheet_shape : (H, W)           grid case, with H * W == n_units
coords      : (n_units, 2)     scatter case, continuous unit positions

How to derive pref_ecc / pref_ang from the response array R of shape
(n_ecc, n_ang, n_exemplars, n_units):

    flat = R.reshape(n_ecc * n_ang, n_exemplars, n_units)
    mean_resp = flat.mean(axis=1)                            # (n_pos, n_units)
    pref_pos = np.argmax(mean_resp, axis=0)
    pref_ecc = pref_pos // n_ang
    pref_ang = pref_pos %  n_ang

    F, p = f_oneway(*[flat[pos] for pos in range(flat.shape[0])])
    from scipy.stats import false_discovery_control
    tuned = false_discovery_control(p) < 0.05

Joint argmax (over the full ecc x ang grid) is the right operator for unit-level
preferred position; marginalizing separately gives biased answers when receptive
fields are point-like rather than ring- or wedge-shaped.

The eccentricity colormap is sequential (small = foveal, large = peripheral).
The polar-angle colormap is cyclic -- twilight_shifted is perceptually uniform;
hsv is the neuroscience-tradition fallback.
"""

import os
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from scipy.stats import f_oneway, false_discovery_control

from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerConfig
from src.models.qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration
from src.utils.smoothing import NeuronSmoothingConv


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

def plot_retinotopy(pref_ecc, pref_ang, tuned, ecc_frac, ang_deg, coords, sheet_shape,
                    ecc_cmap="magma", ang_cmap="twilight_shifted",
                    figsize=(13, 6), savepath=None):
    """Grid layout: two-panel masked seaborn heatmap (ecc + polar angle)."""
    pref_ecc = np.asarray(pref_ecc)
    pref_ang = np.asarray(pref_ang)
    tuned = np.asarray(tuned, dtype=bool)
    ecc_frac = np.asarray(ecc_frac, dtype=float)
    ang_deg = np.asarray(ang_deg, dtype=float)
    H, W = tuple(map(int, sheet_shape))

    assert pref_ecc.size == H * W == tuned.size, "sheet_shape does not match n_units"

    smoother = NeuronSmoothingConv(fwhm_mm=4.0, resolution_mm=1.0)
    ecc_2d = smoother(coords, pref_ecc[np.newaxis,:]).squeeze()
    ang_2d = smoother(coords, pref_ang[np.newaxis,:]).squeeze()

    # pref_ecc = place_on_grid(pref_ecc, coords)
    # pref_ang = place_on_grid(pref_ang, coords)
    tuned = place_on_grid(tuned, coords).astype(bool)

    ecc_2d = ecc_2d.reshape(W, H).astype(float)
    ang_2d = ang_2d.reshape(W, H).astype(float)
    mask_2d = ~tuned.reshape(W, H)

    ecc_2d  = np.rot90(ecc_2d,  k=1)
    ang_2d  = np.rot90(ang_2d,  k=1)
    mask_2d = np.rot90(mask_2d, k=1)



    fig, (ax_e, ax_a) = plt.subplots(1, 2, figsize=figsize)
    panels = [
        (ax_e, ecc_2d, ecc_cmap, len(ecc_frac), ecc_frac,
         "preferred ecc (frac of radius)", "Eccentricity", "{:.2f}"),
        (ax_a, ang_2d, ang_cmap, len(ang_deg), ang_deg,
         "preferred angle (deg)", "Polar angle", "{:.0f}"),
    ]
    for ax, vals_2d, cmap, n_lvl, real_vals, label, title, fmt in panels:
        ax.set_facecolor("0.92")
        hm = sns.heatmap(
            vals_2d, mask=mask_2d,
            cmap=plt.get_cmap(cmap, n_lvl),
            vmin=-0.5, vmax=n_lvl - 0.5,
            square=True, linewidths=0, ax=ax, cbar=True,
            cbar_kws={"shrink": 0.8},
            xticklabels=False, yticklabels=False,
        )
        _real_tick_cbar(hm.collections[0].colorbar, real_vals, label, fmt=fmt)
        ax.set_title(title)
        ax.set_xlabel("sheet x")
        ax.set_ylabel("sheet y")

    fig.suptitle(f"Retinotopy — vision component sheet "
                 f"({tuned.sum()}/{tuned.size} units tuned)")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig, (ax_e, ax_a)


def plot_retinotopy_scatter(pref_ecc, pref_ang, tuned, ecc_frac, ang_deg, coords,
                            ecc_cmap="magma", ang_cmap="twilight_shifted",
                            figsize=(13, 6), point_size=8, savepath=None):
    """Continuous layout: two-panel scatter for (x, y) unit positions."""
    pref_ecc = np.asarray(pref_ecc, dtype=float)
    pref_ang = np.asarray(pref_ang, dtype=float)
    tuned = np.asarray(tuned, dtype=bool)
    ecc_frac = np.asarray(ecc_frac, dtype=float)
    ang_deg = np.asarray(ang_deg, dtype=float)
    coords = np.asarray(coords, dtype=float)
    # rotate 90° right then flip vertically: (x, y) -> (y, x)
    coords = np.stack([coords[:, 1], coords[:, 0]], axis=1)

    sns.set_style("white")
    fig, (ax_e, ax_a) = plt.subplots(1, 2, figsize=figsize)
    panels = [
        (ax_e, pref_ecc, ecc_cmap, len(ecc_frac), ecc_frac,
         "preferred ecc (frac of radius)", "Eccentricity", "{:.2f}"),
        (ax_a, pref_ang, ang_cmap, len(ang_deg), ang_deg,
         "preferred angle (deg)", "Polar angle", "{:.0f}"),
    ]
    for ax, vals, cmap, n_lvl, real_vals, label, title, fmt in panels:
        ax.set_facecolor("0.92")
        ax.scatter(coords[~tuned, 0], coords[~tuned, 1],
                   c="0.8", s=point_size * 0.7, linewidths=0)
        sc = ax.scatter(coords[tuned, 0], coords[tuned, 1],
                        c=vals[tuned], cmap=plt.get_cmap(cmap, n_lvl),
                        vmin=-0.5, vmax=n_lvl - 0.5, s=point_size, linewidths=0)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
        _real_tick_cbar(cbar, real_vals, label, fmt=fmt)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("sheet x")
        ax.set_ylabel("sheet y")
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"Retinotopy — vision component sheet "
                 f"({tuned.sum()}/{tuned.size} units tuned)")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig, (ax_e, ax_a)


def anova_retinotopy(responses: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    responses: (n_ecc, n_ang, n_exemplars, n_units)

    Returns:
        pref_ecc : (n_units,) preferred eccentricity index
        pref_ang : (n_units,) preferred polar-angle index
        tuned    : (n_units,) bool, FDR-corrected at 0.05
    """
    n_ecc, n_ang, n_exemplars, n_units = responses.shape
    flat = responses.reshape(n_ecc * n_ang, n_exemplars, n_units)   # (n_pos, n_ex, n_units)
    mean_resp = flat.mean(axis=1)                                    # (n_pos, n_units)
    pref_pos = np.argmax(mean_resp, axis=0)                         # (n_units,)
    pref_ecc = pref_pos // n_ang
    pref_ang = pref_pos % n_ang

    _, p_values = f_oneway(*[flat[pos] for pos in range(flat.shape[0])])
    p_values = np.where(np.isnan(p_values), 1.0, p_values)  # NaN from zero-variance units -> not significant
    tuned = false_discovery_control(p_values) < 0.05
    return pref_ecc, pref_ang, tuned


@torch.no_grad()
def extract_features_for_images(
    model,
    processor,
    image_paths: List[str],
    batch_size: int = 8,
    device: str = "cuda",
):
    """
    Returns cortical_sheets of shape (N, n_units).
    Feeds each image with no accompanying text via the chat template.
    """
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

    run_dir          = Path(cfg.model.run_dir).resolve()
    device           = str(cfg.model.device)
    neighborhood_dir = str(cfg.model.get("neighborhood_dir", None))

    stimuli_root     = Path(cfg.data.stimuli_root).resolve()
    batch_size       = int(cfg.data.batch_size)

    outdir = Path(output_root) / run_title / category_name
    outdir.mkdir(parents=True, exist_ok=True)

    # Read manifest and sort into the canonical (ecc, ang, exemplar) order
    manifest = pd.read_csv(stimuli_root / "manifest.csv")
    manifest = manifest.sort_values(["ecc_idx", "ang_idx", "exemplar_idx"]).reset_index(drop=True)

    image_paths = manifest["filepath"].tolist()
    n_ecc       = int(manifest["ecc_idx"].nunique())
    n_ang       = int(manifest["ang_idx"].nunique())
    n_exemplars = int(manifest["exemplar_idx"].nunique())
    ecc_frac    = manifest.groupby("ecc_idx")["ecc_frac"].first().values
    ang_deg     = manifest.groupby("ang_idx")["ang_deg"].first().values

    print(f"> Found {len(image_paths)} stimuli "
          f"({n_ecc} ecc x {n_ang} ang x {n_exemplars} exemplars)")

    cache_path = outdir / "cortical_sheets.npy"

    if not os.path.exists(cache_path):
        print(f"> Loading processor & config from: {run_dir}")
        processor = Qwen2_5OmniProcessor.from_pretrained(run_dir)
        model_config = Qwen2_5OmniThinkerConfig.from_pretrained(run_dir)

        model_config.audio_config.is_training = False
        model_config.vision_config.is_training = False
        model_config.text_config.is_training = False
        model_config.apply_spatial_loss = True

        print(f"> Loading model from: {run_dir}")
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            run_dir,
            config=model_config,
            device_map=None,
            torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        )
        model.to(device)
        model.eval()

        cortical_sheets = extract_features_for_images(
            model, processor, image_paths,
            batch_size=batch_size, device=device,
        )

        print(f"> Extracted cortical sheets with shape: {cortical_sheets.shape}")
        np.save(cache_path, cortical_sheets)
    else:
        print(f"> Loading cached cortical sheets from: {cache_path}")
        cortical_sheets = np.load(cache_path)

    coords_lm = np.load(os.path.join(neighborhood_dir, "coords.npy"))

    # (N_total, n_units) -> (n_ecc, n_ang, n_exemplars, n_units)
    cortical_sheets = cortical_sheets.reshape(n_ecc, n_ang, n_exemplars, cortical_sheets.shape[-1])
    print(f"> Reshaped cortical sheets to: {cortical_sheets.shape}")

    pref_ecc, pref_ang, tuned = anova_retinotopy(cortical_sheets)

    plot_retinotopy_scatter(
        pref_ecc, pref_ang, tuned, ecc_frac, ang_deg, coords_lm,
        savepath=outdir / "retinotopy_selectivity_scatter.png",
    )

    plot_retinotopy(
        pref_ecc, pref_ang, tuned, ecc_frac, ang_deg, coords_lm, sheet_shape=coords_lm.max(axis=0) + 1,
        savepath=outdir / "retinotopy_selectivity_sheet.png",
    )

    print(f"> Saved retinotopy scatter plot to: {outdir / 'retinotopy_selectivity_scatter.png'}")
    print(f"> Saved retinotopy sheet plot to: {outdir / 'retinotopy_selectivity_sheet.png'}")

if __name__ == "__main__":
    main()
