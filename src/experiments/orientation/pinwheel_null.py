"""Pinwheel count against a MULTI-DRAW shuffle null. No figures, so it is fast.

A single shuffle draw is one sample from the null distribution. On counts this small
(tens) that cannot distinguish 35-vs-28 from chance. This runs the control many times and
reports where the real count falls in the resulting distribution.
"""
import argparse
import numpy as np

from src.models.spatial_utils import DEFAULT_SHEET_LAYOUT as S
from src.experiments.orientation.nn_delta import slot_window
from src.experiments.orientation.plot_tiled import analyse

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--window", type=int, default=2)
ap.add_argument("--width", type=int, default=5)
ap.add_argument("--reps", type=int, default=20)
ap.add_argument("--slots", default="all")
args = ap.parse_args()

n_enc = S.num_encoder_layers
if args.slots == "all":
    keep = list(range(n_enc))
elif args.slots.startswith("p"):
    lo, hi = (float(x) for x in args.slots[1:].split("-"))
    keep = list(range(int(round(lo / 100 * (n_enc - 1))),
                      int(round(hi / 100 * (n_enc - 1))) + 1))
else:
    lo, hi = (int(x) for x in args.slots.split("-"))
    keep = list(range(lo, hi + 1))

print(f"window={args.window} box={args.width} slots={args.slots} reps={args.reps}")
print(f"\n{'run':<40} {'real':>6} {'null mean':>10} {'null sd':>8} {'z':>7} {'p':>7}")
for run in args.runs:
    d = np.load(f"results/{run}/orientation_unbiased/orientation_stats.npz")
    pref, gate = d["pref_ori"], d["gate"] & d["vision_band"]
    tiles, real = [], 0
    for s in keep:
        th = slot_window(pref, S, s, args.window)
        if th is None:
            continue
        ok = (slot_window(gate.astype(float), S, s, args.window) > 0.5) & np.isfinite(th)
        if ok.sum() < 200:
            continue
        tiles.append((th, ok))
        real += analyse(th, ok, args.width, 0.3, 30.0, 10.0)[2]
    null = []
    for rep in range(args.reps):
        rng = np.random.default_rng(1000 + rep)
        tot = 0
        for th, ok in tiles:
            sh = th.copy()
            sh[ok] = rng.permutation(th[ok])
            tot += analyse(sh, ok, args.width, 0.3, 30.0, 10.0)[2]
        null.append(tot)
    null = np.array(null, dtype=float)
    sd = null.std(ddof=1)
    z = (real - null.mean()) / sd if sd > 0 else float("nan")
    p = (float((null >= real).sum()) + 1.0) / (len(null) + 1.0)
    print(f"{run:<40} {real:>6} {null.mean():>10.1f} {sd:>8.1f} {z:>7.2f} {p:>7.3f}")
print(f"\np = fraction of {args.reps} null draws reaching the real count (add-one "
      f"corrected). Treat p > 0.05 as no detectable effect.")
