"""Does the UNSMOOTHED sheet carry tonotopic structure at all?

Neighbour-correlation of the argmax map is a weak test: the preferred-frequency index is a
noisy 30-way readout, and any smoothing manufactures local correlation by construction.
This measures the thing the spatial loss actually optimises, on the raw per-unit responses
in unit-index order with NO rasterisation and NO smoothing -- so no placement convention
can influence the answer: correlation of the 30-point tuning curve against sheet distance.

Flat = no topography. Decaying = real.
"""
import numpy as np
from src.models.spatial_utils import DEFAULT_SHEET_LAYOUT as SHEET

rng = np.random.default_rng(0)
sheets = np.load("results/tiled32-4608-ckpt730-fwhm3/tonotopy/cortical_sheets.npy")
coords = np.load("src/neighborhoods/model=qwen2-5omni-3b-tiled32-4608-v7_radius=dynamic"
                 "_coords=42/coords.npy")
n_freq, n_ex = 30, sheets.shape[0] // 30

aud = np.where((coords[:, 0] < SHEET.encoder_rows) & (coords[:, 1] >= SHEET.n_col // 2))[0]
print(f"audio-band units: {len(aud)}")

sel = rng.choice(aud, 6000, replace=False)
prof = sheets.reshape(n_freq, n_ex, -1)[:, :, sel].mean(1).T          # (n_sel, 30) curves
prof = prof - prof.mean(1, keepdims=True)
prof /= np.linalg.norm(prof, axis=1, keepdims=True) + 1e-12
C = prof @ prof.T                                                     # tuning-curve corr
D = np.linalg.norm(coords[sel][:, None, :] - coords[sel][None, :, :], axis=-1)

iu = np.triu_indices(len(sel), k=1)
c, dist = C[iu], D[iu]
edges = [0, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
print("\nsheet distance (px) -> mean tuning-curve correlation")
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (dist >= lo) & (dist < hi)
    if m.sum() > 100:
        print(f"  {lo:5d}-{hi:<5d}  r={c[m].mean():+.4f}   (n={m.sum():,})")
print(f"\nshuffled-position control (same pairs, coords permuted):")
perm = rng.permutation(len(sel))
Dp = np.linalg.norm(coords[sel][perm][:, None, :] - coords[sel][perm][None, :, :], axis=-1)
dp = Dp[iu]
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (dp >= lo) & (dp < hi)
    if m.sum() > 100:
        print(f"  {lo:5d}-{hi:<5d}  r={c[m].mean():+.4f}   (n={m.sum():,})")
