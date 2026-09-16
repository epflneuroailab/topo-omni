"""
Generate a bank of localized retinotopy stimuli for Topo-Omni.

For each visual-field position (eccentricity, polar angle) on a polar grid,
renders N exemplars that vary patch size, contrast, exact center jitter,
texture seed, background gray, and background noise -- while holding the
nominal position fixed. These exemplars are the within-position replicates
needed for a per-unit ANOVA tuning test instead of pseudoreplicates.

Design (matched as in the tone bank): the N exemplar parameter sets are drawn
ONCE and reused at every position, so nuisance variation is matched across
conditions and position is the only systematic difference. exemplar_idx is
therefore a crossed factor.

Each stimulus is a square gray RGB image with a Gaussian-windowed band-pass
noise patch placed at (ecc_frac, ang_deg). The Gaussian envelope avoids hard
edges that would create their own response. Band-pass noise puts power in the
mid-frequency band that drives V1-like simple-cell receptive fields strongly.

Coordinates: eccentricity is a fraction of image radius (no display geometry
assumed); polar angle is degrees, math convention -- 0 = right horizontal,
90 = up, increasing counterclockwise.

Output: <out_dir>/pos{p:03d}_ecc{e:02d}_ang{a:02d}_ex{ex:02d}.png + manifest.csv
"""

import os
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from dotenv import load_dotenv
load_dotenv()   # for reproducible random seeds via MASTER_SEED env var

STIMULI_DIR = os.getenv("STIMULI_DIR", "stimuli")


# ----- texture --------------------------------------------------------------

def _bandpass_noise(size, seed, sigma_low=6.0, sigma_high=1.5):
    """Band-pass noise via difference of Gaussians on white noise.

    Normalized so typical magnitude is ~1 (divide by ~2.5 sigma, then clip).
    Dividing by the peak instead would leave typical values near zero and
    produce barely-visible patches.
    """
    rng = np.random.default_rng(seed)
    n = rng.standard_normal((size, size)).astype(np.float32)
    bp = gaussian_filter(n, sigma_high) - gaussian_filter(n, sigma_low)
    bp /= 2.5 * float(bp.std()) + 1e-8
    return np.clip(bp, -1.0, 1.0)


# ----- single exemplar ------------------------------------------------------

def render_exemplar(canvas_size, ecc_frac, ang_deg, p):
    """Render one exemplar at (ecc_frac, ang_deg). `p` is from sample_params()."""
    H = W = canvas_size
    radius = min(H, W) / 2.0

    ecc_eff = max(0.0, ecc_frac + p["ecc_jitter_frac"])
    ang_eff = ang_deg + p["ang_jitter_deg"]

    cx = W / 2.0 + radius * ecc_eff * np.cos(np.deg2rad(ang_eff))
    cy = H / 2.0 - radius * ecc_eff * np.sin(np.deg2rad(ang_eff))   # image y inverted
    sigma_px = p["sigma_frac"] * radius

    bp = _bandpass_noise(H, p["seed"])
    yy, xx = np.mgrid[0:H, 0:W]
    envelope = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma_px ** 2))

    img = p["bg_gray"] + envelope * bp * p["contrast"] * 0.7
    if p["bg_noise_std"] > 0:
        rng = np.random.default_rng(p["seed"] + 1)
        img = img + rng.normal(0.0, p["bg_noise_std"], img.shape).astype(np.float32)

    img = np.clip(img, 0.0, 1.0)
    return img, (cx, cy)


# ----- parameter sampling ---------------------------------------------------

def sample_params(rng, idx):
    """Draw one exemplar's nuisance params (reused across all positions)."""
    return {
        "exemplar_idx": idx,
        "sigma_frac": float(rng.uniform(0.07, 0.16)),     # patch sigma / image radius
        "contrast": float(rng.uniform(0.6, 1.0)),
        "ecc_jitter_frac": float(rng.uniform(-0.025, 0.025)),
        "ang_jitter_deg": float(rng.uniform(-5.0, 5.0)),
        "bg_gray": float(rng.uniform(0.45, 0.55)),
        "bg_noise_std": float(rng.uniform(0.0, 0.02)),
        "seed": int(rng.integers(0, 2 ** 31)),
    }


# ----- bank generation ------------------------------------------------------

def generate_retino_bank(out_dir, n_ecc=8, n_ang=16,
                          ecc_min_frac=0.06, ecc_max_frac=0.85,
                          n_exemplars=12, canvas_size=448,
                          log_ecc=True, master_seed=0):
    """Render n_ecc * n_ang * n_exemplars images and write manifest.csv.

    The manifest is what you group on downstream: load it, group by `pos_idx`
    (or by `(ecc_idx, ang_idx)`), and the exemplars within each group are your
    replicates for the ANOVA. Feed a `(n_ecc, n_ang, n_exemplars, n_units)`
    response array to your analysis.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if log_ecc:
        ecc_grid = np.logspace(np.log10(ecc_min_frac), np.log10(ecc_max_frac), n_ecc)
    else:
        ecc_grid = np.linspace(ecc_min_frac, ecc_max_frac, n_ecc)
    ang_grid = np.linspace(0.0, 360.0, n_ang, endpoint=False)

    # exemplar params drawn ONCE -> matched design across positions
    rng = np.random.default_rng(master_seed)
    exemplar_params = [sample_params(rng, i) for i in range(n_exemplars)]

    rows = []
    pos_idx = 0
    for e_idx, ecc in enumerate(ecc_grid):
        for a_idx, ang in enumerate(ang_grid):
            for p in exemplar_params:
                img, (cx, cy) = render_exemplar(canvas_size, ecc, ang, p)
                img_u8 = (img * 255.0).astype(np.uint8)
                img_rgb = np.stack([img_u8] * 3, axis=-1)              # gray -> RGB
                fname = (f"pos{pos_idx:03d}_ecc{e_idx:02d}_ang{a_idx:02d}"
                         f"_ex{p['exemplar_idx']:02d}.png")
                fpath = out_dir / fname
                Image.fromarray(img_rgb).save(fpath)
                rows.append({
                    "filepath": str(fpath),
                    "pos_idx": pos_idx,
                    "ecc_idx": e_idx,
                    "ang_idx": a_idx,
                    "ecc_frac": round(float(ecc), 4),
                    "ang_deg": round(float(ang), 2),
                    "center_x_px": round(float(cx), 1),
                    "center_y_px": round(float(cy), 1),
                    "canvas_size": canvas_size,
                    **{k: p[k] for k in (
                        "exemplar_idx", "sigma_frac", "contrast",
                        "ecc_jitter_frac", "ang_jitter_deg",
                        "bg_gray", "bg_noise_std", "seed")},
                })
            pos_idx += 1

    manifest = out_dir / "manifest.csv"
    with open(manifest, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} images "
          f"({n_ecc} ecc x {n_ang} ang x {n_exemplars} exemplars) "
          f"+ manifest -> {out_dir}")
    return manifest


if __name__ == "__main__":
    generate_retino_bank(
        out_dir=os.path.join(STIMULI_DIR, "retino_bank"),
        n_ecc=8, n_ang=16,
        n_exemplars=12,
        canvas_size=448,
    )