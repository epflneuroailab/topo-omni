"""
Plot retinotopic maps (eccentricity) on the Topo-Omni
cortical sheet, masking units that are not significantly position-tuned.

Inputs
------
pref_ecc    : (n_units,) int   preferred eccentricity index per unit
tuned       : (n_units,) bool  True where the unit passed the ANOVA tuning test (FDR-corrected)
ecc_frac    : (n_ecc,) float   eccentricities (fraction of image radius), same index order as pref_ecc
sheet_shape : (H, W)           grid case, with H * W == n_units
coords      : (n_units, 2)     scatter case, continuous unit positions
"""

import os
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

# Matches the classic retinotopy eccentricity colormap:
# magenta (fovea) → red → orange → yellow → yellow-green (periphery)
ECC_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "retinotopy_ecc", ["#FF00FF", "#FF0000", "#FF8000", "#FFFF00", "#80FF00"]
)
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from scipy.stats import f_oneway, false_discovery_control

from qwen_omni_utils import process_mm_info
from src.core.model_loading import load_topo_omni, unified_grid_coords
from src.utils.smoothing import NeuronSmoothingConv
from src.models.spatial_utils import DEFAULT_SHEET_LAYOUT as SHEET

# The vision encoder's extent on the sheet, derived from the active layout. These were
# HARDCODED as 144 / 256 -- the old 309x512 strip's thinker_rows and n_col//2. On tiled32
# (800x576, thinker_rows 288, n_col//2 288) those literals crop 144 rows of LANGUAGE/
# COGNITIVE units into the map and cut 32 columns off the vision band, so the published
# panel was neither the vision encoder nor the whole sheet. Same stale constants that
# marvi.py's get_selectivity_mask was already fixed for.
def _vision_band(arr2d):
    """Blank the audio half and crop to the vision encoder, in the post-rot90 frame."""
    thk, mid = SHEET.thinker_rows, SHEET.n_col // 2
    arr2d[thk:, mid:] = np.nan          # audio band
    return arr2d[thk:, :mid]            # vision band only (drops the thinker rows)


# The config resolves ${oc.env:STIMULI_DIR} through OmegaConf, which reads the REAL
# environment -- not the repo .env. run_angle.py and run.py both load it here; this
# module did not, so any config using an env interpolation died with
# "Environment variable 'STIMULI_DIR' not found" unless the caller happened to export it.
from dotenv import load_dotenv
load_dotenv()


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

def plot_retinotopy(pref_ecc, tuned, ecc_frac, coords, sheet_shape,
                    ecc_cmap="magma", figsize=(7, 6), savepath=None):
    """Grid layout: single-panel masked seaborn heatmap (eccentricity)."""
    pref_ecc = np.asarray(pref_ecc)
    tuned = np.asarray(tuned, dtype=bool)
    ecc_frac = np.asarray(ecc_frac, dtype=float)
    H, W = tuple(map(int, sheet_shape))

    assert pref_ecc.size == H * W == tuned.size, "sheet_shape does not match n_units"

    ecc_2d = pref_ecc

    tuned = place_on_grid(tuned, coords).astype(bool)

    ecc_2d = ecc_2d.reshape(W, H).astype(float)
    mask_2d = ~tuned.reshape(W, H)

    ecc_2d = np.rot90(ecc_2d, k=1)
    mask_2d = np.rot90(mask_2d, k=1)

    # Crop to the vision band, exactly as run_angle does, so the polar-angle and
    # eccentricity panels cover the same units and can be read against each other. The
    # thinker-blanking line that used to sit here commented out did the same job by hand.
    mask_2d = None
    ecc_2d = _vision_band(ecc_2d)

    fig, ax_e = plt.subplots(1, 1, figsize=figsize)
    ax_e.set_facecolor("0.92")
    hm = sns.heatmap(
        ecc_2d, mask=mask_2d,
        cmap=ecc_cmap,
        vmin=-0.5, vmax=len(ecc_frac) - 0.5,
        square=True, linewidths=0, ax=ax_e, cbar=True,
        cbar_kws={"shrink": 0.6},
        xticklabels=False, yticklabels=False,
    )
    _real_tick_cbar(hm.collections[0].colorbar, ecc_frac,
                    "preferred eccentricity (deg)", fmt="{:.2f}")
    ax_e.set_title("Eccentricity")
    ax_e.set_xlabel("")
    ax_e.set_ylabel("")

    # fig.suptitle(f"Retinotopy — vision component sheet "
    #              f"({tuned.sum()}/{tuned.size} units tuned)")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig, ax_e


def plot_retinotopy_scatter(pref_ecc, tuned, ecc_frac, coords,
                            ecc_cmap=ECC_CMAP, figsize=(7, 6), point_size=8, savepath=None):
    """Continuous layout: single-panel scatter for (x, y) unit positions (eccentricity)."""
    pref_ecc = np.asarray(pref_ecc, dtype=float)
    tuned = np.asarray(tuned, dtype=bool)
    ecc_frac = np.asarray(ecc_frac, dtype=float)
    coords = np.asarray(coords, dtype=float)

    coords = np.stack([coords[:, 1], coords[:, 0]], axis=1)

    sns.set_style("white")
    fig, ax_e = plt.subplots(1, 1, figsize=figsize)
    ax_e.set_facecolor("0.92")
    ax_e.scatter(coords[~tuned, 0], coords[~tuned, 1],
                 c="0.8", s=point_size * 0.7, linewidths=0)
    sc = ax_e.scatter(coords[tuned, 0], coords[tuned, 1],
                      c=pref_ecc[tuned], cmap=ecc_cmap,
                      vmin=-0.5, vmax=len(ecc_frac) - 0.5, s=point_size, linewidths=0)
    cbar = fig.colorbar(sc, ax=ax_e, shrink=0.6)
    _real_tick_cbar(cbar, ecc_frac, "preferred eccentricity (deg)", fmt="{:.2f}")
    ax_e.set_aspect("equal")
    ax_e.set_title("Eccentricity")
    ax_e.set_xlabel("")
    ax_e.set_ylabel("")
    ax_e.set_xticks([])
    ax_e.set_yticks([])

    # fig.suptitle(f"Retinotopy — vision component sheet "
    #              f"({tuned.sum()}/{tuned.size} units tuned)")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig, ax_e


def anova_retinotopy(responses: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    responses: (n_ecc, n_exemplars, n_units)

    Returns:
        pref_ecc : (n_units,) preferred eccentricity index
        pref_ang : (n_units,) preferred polar-angle index
        tuned    : (n_units,) bool, FDR-corrected at 0.05
    """
    n_ecc, _, _ = responses.shape
    mean_resp = responses.mean(axis=1)                       # (n_ecc, n_units)
    pref_ecc = np.argmax(mean_resp, axis=0)                 # (n_units,)

    _, p_values = f_oneway(*[responses[ecc] for ecc in range(n_ecc)])
    p_values = np.where(np.isnan(p_values), 1.0, p_values)  # NaN from zero-variance units -> not significant
    tuned = false_discovery_control(p_values) < 0.05
    return pref_ecc, tuned


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
    parser.add_argument("--config", type=str, default="src/configs/eval/retinotopy.yml",
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

    outdir = Path(output_root) / run_title / f"{category_name}_eccentricity"
    outdir.mkdir(parents=True, exist_ok=True)

    # path example: ecc_target00.50_actual00.50_pattern002_frame1426_mask0616
    image_paths = sorted((stimuli_root / "eccentricity").glob("*.png"))
    ecc_frac = sorted(set(float(p.stem.split("_")[1].split("target")[-1]) for p in image_paths))
    n_ecc = len(ecc_frac)
    n_exemplars = len(set(p.stem.split("_")[3].split("pattern")[-1] for p in image_paths))
    image_paths = [str(p) for p in image_paths]

    print(f"> Found {len(image_paths)} stimuli "
          f"({n_ecc} ecc x {n_exemplars} exemplars)")
    
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

    # Coordinates must be the ones the run was TRAINED with. `build_sheet_coordinates`
    # shuffles every unit within its layer block, so the identity grid from
    # `unified_grid_coords()` puts only 94 of tiled32's 460,800 units back where they
    # belong, and another run's coords file has the wrong unit count outright (the
    # 309x512 strip has 158,208). Either way the map is scrambled rather than wrong-looking,
    # so take the path from the config and fail loudly if it does not match the sheet.
    _coords_path = cfg.stats.get("coords_path", None) if "stats" in cfg else None
    if _coords_path:
        coords_lm = np.load(_coords_path)
        print(f"> Unit coordinates: {_coords_path} {coords_lm.shape}")
    else:
        coords_lm = unified_grid_coords()
        print("> WARNING: no stats.coords_path in the config; falling back to "
              "the identity grid. This is only correct for the run it came from.")
    from src.models.spatial_utils import DEFAULT_SHEET_LAYOUT as _S
    if coords_lm.shape[0] != _S.num_units:
        raise SystemExit(
            f"coords have {coords_lm.shape[0]} units but the active sheet layout "
            f"({_S.mode}, {_S.n_row}x{_S.n_col}) has {_S.num_units}. Set "
            f"TOPO_SHEET_LAYOUT and stats.coords_path to the same run.")

    # (N_total, n_units) -> (n_ecc, n_exemplars, n_units)
    cortical_sheets = cortical_sheets.reshape(n_ecc, n_exemplars, cortical_sheets.shape[-1])
    print(f"> Reshaped cortical sheets to: {cortical_sheets.shape}")

    smoother = NeuronSmoothingConv(
        fwhm_mm=float(cfg.stats.get("fwhm_mm", 4.0)) if "stats" in cfg else 4.0,
        resolution_mm=float(cfg.stats.get("resolution_mm", 1.0)) if "stats" in cfg else 1.0)
    smoothed_sheet = smoother(coords_lm, cortical_sheets.reshape(-1, cortical_sheets.shape[-1]))
    
    pref_ecc, tuned = anova_retinotopy(smoothed_sheet.reshape(n_ecc, n_exemplars, -1))

    # plot_retinotopy_scatter(
    #     pref_ecc, tuned, ecc_frac, coords_lm,
    #     savepath=outdir / "retinotopy_selectivity_scatter.png",
    # )

    plot_retinotopy(
        pref_ecc, tuned, ecc_frac, coords_lm, sheet_shape=coords_lm.max(axis=0) + 1,
        savepath=outdir / "retinotopy_selectivity_sheet.png",
    )

    print(f"> Saved retinotopy scatter plot to: {outdir / 'retinotopy_selectivity_scatter.png'}")
    print(f"> Saved retinotopy sheet plot to: {outdir / 'retinotopy_selectivity_sheet.png'}")

if __name__ == "__main__":
    main()
