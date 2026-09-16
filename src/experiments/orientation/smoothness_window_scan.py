"""Sliding window over the vision encoder: which depth range best matches macaque V1
orientation smoothness?

SELECTED ON THE OUTCOME. Scanning many windows and keeping the closest to 0.684 will find
a match by chance -- smoothness has a positive floor of ~0.04-0.09 from the max-over-bins
statistic, and noisy windows scatter around it. So every window is reported with its own
DISTANCE-PERMUTED null (same units, same values, distance relation destroyed), and the
column that matters is `real - null`, not `real`. A window whose real value lands on 0.684
while its null is also high has matched macaque on noise.

fwhm=0: the '-nosmooth' stats are used directly, no re-smoothing anywhere.
"""
import argparse
import numpy as np

from src.experiments.orientation.metrics import (
    band_grid, layer_band_mask, parse_layers, metric_difference_over_distance,
    smoothness, N_VISION_LAYERS)
from src.experiments.orientation.macaque_map import tdann_bin_edges

MACAQUE_ORI = 0.684

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--lam", type=float, default=11.0, help="pinned, as the panels do")
ap.add_argument("--sizes", default="3,4,6,8,12,16,24,33")
ap.add_argument("--num-samples", type=int, default=10)
ap.add_argument("--nbins", type=int, default=9)
ap.add_argument("--min-units", type=int, default=2000)
ap.add_argument("--min-slot", type=int, default=0,
                help="lowest slot a window may start at. Use 1 to exclude the conv "
                     "front-end: slot 0 holds only ~388 gated units against ~2,000 "
                     "elsewhere and scores WORSE than its own shuffle (47.1 deg vs a "
                     "45 deg null), so any window containing it is dragged down and the "
                     "early-layer windows lose for a reason that has nothing to do with "
                     "the early layers.")
ap.add_argument("--fwhm", type=float, default=0.0,
                help="re-smooth the RAW sheets at this fwhm and recompute preference, "
                     "exactly as smoothness_panels.topo_omni_curves does, so the scan and "
                     "the final panel sit on the same map. The GATE still comes from the "
                     "fwhm=0 stats -- that is what the panels do too, and keeping it fixed "
                     "means the unit set does not shift between fwhm settings.")
ap.add_argument("--coords", default="src/neighborhoods/model=qwen2-5omni-3b-strip-4096-v1"
                                    "_radius=5_neighborhoods=20_coords=42/coords.npy")
args = ap.parse_args()

d = np.load(f"results/{args.run}/orientation_unbiased/orientation_stats.npz")
gate = d["gate"] & d["vision_band"]
if args.fwhm > 0:
    import os
    from pathlib import Path
    from src.utils.smoothing import NeuronSmoothingConv
    from src.experiments.orientation.run import GRATING_BANKS
    from src.experiments.orientation.smoothness_panels import preferred_orientation, parse_stimuli
    root = Path(os.environ["STIMULI_DIR"]) / GRATING_BANKS["unbiased"][1]
    _, ori_deg, counts, _ = parse_stimuli(root, color="all")
    keys = np.repeat(ori_deg, counts)
    raw = np.load(f"results/{args.run}/orientation_unbiased/cortical_sheets.npy")
    sm = NeuronSmoothingConv(fwhm_mm=args.fwhm, resolution_mm=1.0)(np.load(args.coords), raw)
    pref, _ = preferred_orientation(sm, keys)
    print(f"re-smoothed at fwhm={args.fwhm} (sigma={args.fwhm/2.3548:.3f} px)")
else:
    pref = d["pref_ori"]
ori = band_grid(pref)
gated = (band_grid(gate.astype(float)) > 0.5) & np.isfinite(ori)
edges = tdann_bin_edges(args.lam, n_bins=args.nbins)

def score(mask, shuffle, seed=0):
    yy, xx = np.nonzero(mask)
    pos = np.stack([xx, yy], 1).astype(float)
    _, curves = metric_difference_over_distance(
        ori[mask], pos, distance_cutoff=edges[-1], num_samples=args.num_samples,
        sample_size=1000, num_bins=args.nbins, shuffle=shuffle, seed=seed,
        bin_edges=edges)
    return float(np.mean([smoothness(c) for c in curves]))

rows = []
for size in (int(x) for x in args.sizes.split(",")):
    for start in range(args.min_slot, N_VISION_LAYERS - size + 1):
        m = gated & layer_band_mask(parse_layers(f"{start}-{start+size-1}"))
        n = int(m.sum())
        if n < args.min_units:
            continue
        r = score(m, False)
        nl = score(m, True)
        rows.append((size, start, start + size - 1, n, r, nl, r - nl))

print(f"{args.run}  (fwhm=0, Lambda pinned {args.lam} px, macaque={MACAQUE_ORI})")
print(f"scanned {len(rows)} windows\n")
hdr = "%-4s %-7s %8s %8s %8s %8s %9s"
print(hdr % ("size", "slots", "units", "real", "null", "real-null", "|d macaque|"))
print("--- closest REAL value to macaque (what was asked, selection-biased) ---")
for w in sorted(rows, key=lambda t: abs(t[4] - MACAQUE_ORI))[:6]:
    print(hdr % (w[0], f"{w[1]}-{w[2]}", f"{w[3]:,}", f"{w[4]:.3f}", f"{w[5]:.3f}",
                 f"{w[6]:+.3f}", f"{abs(w[4]-MACAQUE_ORI):.3f}"))
print("\n--- largest real-null (strongest genuine structure) ---")
for w in sorted(rows, key=lambda t: -t[6])[:6]:
    print(hdr % (w[0], f"{w[1]}-{w[2]}", f"{w[3]:,}", f"{w[4]:.3f}", f"{w[5]:.3f}",
                 f"{w[6]:+.3f}", f"{abs(w[4]-MACAQUE_ORI):.3f}"))
r_ = np.array([w[4] for w in rows]); n_ = np.array([w[5] for w in rows])
print(f"\nacross all windows: real {r_.min():.3f}-{r_.max():.3f}, "
      f"null {n_.min():.3f}-{n_.max():.3f}")
print(f"no window comes close to macaque {MACAQUE_ORI}" if r_.max() < 0.5 else "")
