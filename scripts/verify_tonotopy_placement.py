"""Sanity checks for the tiled32 tonotopy map: placement, tuning, and depth ordering.

Recomputed from the cached cortical_sheets.npy so it costs ~2 min, not a model pass.
"""
import sys
import numpy as np, torch
from src.utils.smoothing import NeuronSmoothingConv
from src.models.spatial_utils import DEFAULT_SHEET_LAYOUT as SHEET
from src.experiments.tonotopy.run import anova_tonotopy, freqs_hz, place_on_grid

FWHM = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
OUT = ("results/tiled32-4608-ckpt730-nosmooth/tonotopy" if FWHM == 0 else
       f"results/tiled32-4608-ckpt730-fwhm{FWHM:g}/tonotopy")
print(f"fwhm_mm = {FWHM}  ->  {OUT}")
COORDS = ("src/neighborhoods/model=qwen2-5omni-3b-tiled32-4608-v7_radius=dynamic"
          "_coords=42/coords.npy")

print(f"layout: {SHEET.mode} {SHEET.n_row}x{SHEET.n_col} enc_rows={SHEET.encoder_rows} "
      f"enc_slots={SHEET.num_encoder_layers} units={SHEET.num_units}")

sheets = np.load(f"{OUT}/cortical_sheets.npy")            # (480, n_units)
coords = np.load(COORDS)
n_units = sheets.shape[-1]
assert n_units == SHEET.num_units == len(coords)

sm = NeuronSmoothingConv(fwhm_mm=FWHM, resolution_mm=1.0)
smoothed = sm(coords, sheets)
print(f"smoother grid: H={sm.height} W={sm.width}  (expect n_col x n_row)")
pref, tuned = anova_tonotopy(smoothed.reshape(len(freqs_hz), -1, n_units))
np.savez(f"{OUT}/tonotopy_preferred.npz", preferred=pref, tuned=tuned,
         freqs_hz=freqs_hz, fwhm_mm=FWHM)

H, W = SHEET.n_row, SHEET.n_col

def nbr_corr(m):
    """Nearest-neighbour correlation along each axis, over the audio band only."""
    band = m[:SHEET.encoder_rows, W // 2:]               # sheet-ordered audio band
    out = []
    for a in (0, 1):
        x = np.moveaxis(band, a, 0)
        u, v = x[:-1].ravel(), x[1:].ravel()
        ok = ~(np.isnan(u) | np.isnan(v))
        out.append(np.corrcoef(u[ok], v[ok])[0, 1])
    return out                                            # [vertical, horizontal]

cands = {
    "reshape(n_col,n_row)+rot90 [used]": np.flipud(np.rot90(pref.reshape(W, H), k=1)),
    "reshape(n_row,n_col)":              pref.reshape(H, W).astype(float),
    "coords scatter":                    place_on_grid(pref, coords[:, ::-1]).astype(float),
}
print("\nplacement                            vert r   horiz r")
for k, m in cands.items():
    v, h = nbr_corr(m)
    print(f"{k:36s} {v:+.3f}   {h:+.3f}")

# tuning + per-depth means on the placement actually used
sheet = cands["reshape(n_col,n_row)+rot90 [used]"]
tun2d = np.flipud(np.rot90(tuned.reshape(W, H), k=1))
rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
aud = (rr < SHEET.encoder_rows) & (cc >= W // 2)
vis = (rr < SHEET.encoder_rows) & (cc < W // 2)
thk = rr >= SHEET.encoder_rows
for name, m in (("audio enc", aud), ("vision enc (undriven)", vis), ("thinker", thk)):
    print(f"tuned in {name:22s}: {tun2d[m].mean()*100:5.1f}%  "
          f"({int(tun2d[m].sum())}/{int(m.sum())})")

slot = SHEET.depth_slot(
    torch.from_numpy(np.stack([rr.ravel(), cc.ravel()], 1))).numpy().reshape(H, W)
print("\naudio encoder depth -> mean preferred freq (Hz), tuned units only")
sm_pref = np.where(tun2d, sheet, np.nan)
for l in range(SHEET.num_encoder_layers):
    m = aud & (slot == l)
    vals = sm_pref[m]
    hz = freqs_hz[int(round(np.nanmean(vals)))] if np.any(~np.isnan(vals)) else float("nan")
    print(f"  layer {l:2d}: {hz:7.0f} Hz   (n={int((~np.isnan(vals)).sum())})")
