"""Sheet maps of preferred SPATIAL FREQUENCY and preferred COLOUR.

The companion to ``plot_panel.py``, which draws the same map for orientation. Everything
about the geometry is shared with it -- the vision-band extraction, the layer selection,
the crop and the smoothing window are taken from the same helpers -- so the three maps are
directly comparable and can sit in one figure.

WHY THIS IS NOT JUST ``plot_panel --measure sf``
-------------------------------------------------
Two reasons, both about the quantity being averaged.

1. ``orientation_stats.npz`` has no SF or colour in it. Its ``tuning_curve`` is (8, n_units)
   over ORIENTATION -- spatial frequency, phase and colour are marginalised away before it
   is written. Both preferences are therefore recomputed here from the raw
   ``cortical_sheets.npy`` and the grating filenames, reusing ``smoothness_panels``'
   ``preferred_sf_index`` / ``preferred_color`` so the definitions match the bar figure.

2. ``plot_panel`` smooths with a CIRCULAR mean (``exp(2j*theta)``), which is right for an
   axis that wraps at 180 deg and wrong for everything else. Spatial frequency is linear
   and colour is binary, so applying it would fold the two ends of the SF range onto each
   other. This module uses a plain box mean over the same window instead.

Usage:
    python -m src.experiments.orientation.plot_map_sf_color \\
        --sheets results/<run>/orientation_unbiased/cortical_sheets.npy \\
        --stats  results/<run>/orientation_unbiased/orientation_stats.npz \\
        --outdir results/<run>/orientation_unbiased --layers all
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.experiments.orientation.run import parse_stimuli, GRATING_BANKS
from src.experiments.orientation.metrics import (
    band_grid, layer_band_mask, column_spacing, N_VISION_LAYERS,
)
from src.experiments.orientation.plot_panel import parse_layers
from src.experiments.orientation.smoothness_panels import (
    preferred_sf_index, preferred_color,
)
from src.utils.smoothing import NeuronSmoothingConv

from dotenv import load_dotenv
load_dotenv()
import os
STIMULI_DIR = os.getenv("STIMULI_DIR")


def box_mean_same(field2d, valid, width):
    """Linear counterpart of plot_panel.smoothed_orientation_map_same.

    Same window and same 'keep every row' behaviour, but a plain mean over the valid units
    in each window rather than a circular one -- SF is linear and colour is binary, so a
    circular mean would wrap their two extremes onto each other.
    """
    from scipy.ndimage import uniform_filter
    v = valid.astype(float)
    num = uniform_filter(np.where(valid, field2d, 0.0), size=width, mode="nearest")
    den = uniform_filter(v, size=width, mode="nearest")
    out = np.full_like(field2d, np.nan, dtype=float)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out, den * (width ** 2)


def draw(field, title, cbar_label, cmap, out, vmin=None, vmax=None,
         ticks=None, ticklabels=None, figsize=None):
    h, w = field.shape
    figsize = figsize or (9.0, max(2.2, 9.0 * h / w))
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(np.ma.masked_invalid(field), cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest", origin="upper", aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.8)
    ax.set_title(title, fontsize=10, pad=6)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cb.set_label(cbar_label, fontsize=9)
    if ticks is not None:
        cb.set_ticks(ticks)
        if ticklabels is not None:
            cb.set_ticklabels(ticklabels)
    fig.savefig(out, dpi=400, bbox_inches="tight")
    fig.savefig(str(out).replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"> wrote {out} (and .pdf)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheets", required=True, help="raw cortical_sheets.npy for the run")
    ap.add_argument("--stats", required=True,
                    help="orientation_stats.npz from the SAME run -- only its reliability "
                         "gate and Lambda are used, not its tuning curve")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--coords", default="src/neighborhoods/"
                    "model=qwen2-5omni-3b-tiled32-4608-v7_radius=dynamic_coords=42/coords.npy")
    ap.add_argument("--layers", default="all")
    ap.add_argument("--bank", default="unbiased")
    ap.add_argument("--fwhm-mm", type=float, default=3.0,
                    help="unit-level smoothing applied to the raw sheets before computing "
                         "preferences; match the orientation run this sits beside.")
    ap.add_argument("--smooth-window-frac", type=float, default=0.15,
                    help="map-smoothing window as a fraction of Lambda, as in plot_panel.")
    ap.add_argument("--no-crop", dest="crop", action="store_false", default=True)
    ap.add_argument("--column-spacing", type=float, default=None)
    args = ap.parse_args()

    stimuli_root = Path(STIMULI_DIR) / GRATING_BANKS[args.bank][1]
    _, ori_deg, counts, meta = parse_stimuli(stimuli_root, color="all")
    meta = {k: np.asarray(v) for k, v in meta.items()}

    sheets = np.load(args.sheets)
    coords = np.load(args.coords)
    if args.fwhm_mm > 0:
        sheets = NeuronSmoothingConv(fwhm_mm=args.fwhm_mm, resolution_mm=1.0)(coords, sheets)
    print(f"> {args.sheets}: {sheets.shape}, fwhm {args.fwhm_mm} mm")

    pref_sf, _ = preferred_sf_index(sheets, meta["sf"])
    pref_col, _ = preferred_color(sheets, meta["colour"])
    sfs = np.unique(meta["sf"])

    d = np.load(args.stats, allow_pickle=True)
    gate = d["gate"] if "gate" in d.files else d["tuned"]
    gate_band = band_grid(gate.astype(float)) > 0.5

    # Lambda and the window come from the ORIENTATION map, exactly as plot_panel computes
    # them, so all three maps are smoothed at the same spatial scale.
    ori_band = band_grid(d["pref_ori"])
    lam = args.column_spacing
    if lam is None:
        lam, _, _ = column_spacing(ori_band, gate_band & np.isfinite(ori_band))
    width = max(3, int(round(args.smooth_window_frac * lam)))
    print(f"> Lambda = {lam:.2f} px, window {width} px")

    sel = parse_layers(args.layers)
    lay_mask = layer_band_mask(sel)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    tag = "all" if len(sel) >= N_VISION_LAYERS else f"layers{args.layers}"

    specs = [
        ("sf", pref_sf, "Preferred spatial frequency", "spatial frequency (cpd)",
         "viridis", np.arange(len(sfs)), [f"{s:.0f}" for s in sfs]),
        ("color", pref_col, "Preferred colour", "",
         matplotlib.colors.ListedColormap(["#3A3A3A", "#C1272D"]), [0, 1], ["b/w", "r/g"]),
    ]
    for name, pref, title, cbar_label, cmap, ticks, ticklabels in specs:
        band = band_grid(pref)
        valid = gate_band & np.isfinite(band) & lay_mask
        b, v = band.copy(), valid.copy()
        if args.crop and len(sel) < N_VISION_LAYERS:
            rows = np.nonzero(lay_mask.any(axis=1))[0]
            b, v = b[rows.min():rows.max() + 1], v[rows.min():rows.max() + 1]
        sm, _ = box_mean_same(b, v, width)
        sm[~v] = np.nan
        lo, hi = (0, len(sfs) - 1) if name == "sf" else (0, 1)
        draw(sm, f"{title} — {'all layers' if tag == 'all' else 'vision ' + args.layers}",
             cbar_label, cmap, outdir / f"{name}_map_{tag}.png",
             vmin=lo, vmax=hi, ticks=ticks, ticklabels=ticklabels)


if __name__ == "__main__":
    main()
