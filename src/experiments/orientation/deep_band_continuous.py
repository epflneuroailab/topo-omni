"""Zoom the DEEP vision band and test it as a CONTINUOUS map, with a matched control.

I previously analysed each 16x256 slot in isolation and called the full-band count
(plot_pinwheels_panel: 1043 pinwheels) a depth-contamination artifact. That may have been
too quick. The spatial loss uses radius-5 discs against 16-row layers, so a neighbourhood
centred within 5 rows of a boundary already spans two layers -- the objective organises
the band CONTINUOUSLY, and slicing it into 16-row strips can only destroy structure that
crosses a seam.

So: analyse the whole vision band as one map, and control it by shuffling positions
WITHIN EACH LAYER. That destroys tangential arrangement while preserving the depth
ordering, so anything the depth gradient alone can produce survives in the control.
"""
import argparse, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.models.spatial_utils import DEFAULT_SHEET_LAYOUT as S
from src.experiments.orientation.metrics import count_pinwheels, winding_numbers
from src.experiments.orientation.plot_panel import smoothed_orientation_map_same

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--width", type=int, default=5)
ap.add_argument("--sd-thresh", type=float, default=30.0)
ap.add_argument("--reps", type=int, default=20)
ap.add_argument("--deep-slots", default="24-32")
args = ap.parse_args()

d = np.load(f"results/{args.run}/orientation_unbiased/orientation_stats.npz")
pref, gate = d["pref_ori"], d["gate"] & d["vision_band"]
half, rpl = S.n_col // 2, S.rows_per_encoder_layer

# (n_col, n_row) -> vision band as [sheet_row, sheet_col]; NO rot90, so row 0 = slot 0
g_pref = pref.reshape(S.n_col, S.n_row)[0:half, 0:S.encoder_rows].T
g_gate = gate.reshape(S.n_col, S.n_row)[0:half, 0:S.encoder_rows].T > 0.5
print(f"vision band {g_pref.shape} = {S.num_encoder_layers} slots x {rpl} rows x {half} cols")

def shuffled_band(seed):
    """Permute positions WITHIN each layer: kills tangential order, keeps depth order."""
    rng = np.random.default_rng(seed)
    out = g_pref.copy()
    for l in range(S.num_encoder_layers):
        sl = slice(l * rpl, (l + 1) * rpl)
        blk, m = out[sl], g_gate[sl]
        v = blk[m]
        blk[m] = rng.permutation(v)
        out[sl] = blk
    return out

def count(band, mask):
    ori, sd, _ = smoothed_orientation_map_same(band, mask, args.width)
    wn = winding_numbers(ori)
    c, ctr = count_pinwheels(wn, sd, thresh=0.3, sd_thresh_deg=args.sd_thresh)
    return int(sum(c)), ori, ctr

lo, hi = (int(x) for x in args.deep_slots.split("-"))
sl = slice(lo * rpl, (hi + 1) * rpl)

for name, band in (("FULL vision band", (g_pref, g_gate, slice(None))),
                   (f"DEEP slots {args.deep_slots}", (g_pref, g_gate, sl))):
    p_, m_, s_ = band
    real, ori, ctr = count(p_[s_], m_[s_])
    null = np.array([count(shuffled_band(700 + r)[s_], m_[s_])[0] for r in range(args.reps)],
                    dtype=float)
    z = (real - null.mean()) / null.std(ddof=1) if null.std(ddof=1) > 0 else float("nan")
    pv = (float((null >= real).sum()) + 1) / (len(null) + 1)
    print(f"{name:<26} shape {p_[s_].shape}  real {real:>5}  "
          f"null {null.mean():>7.1f}+-{null.std(ddof=1):.1f}  z={z:>6.2f}  p={pv:.3f}")

# ---- picture: deep band, real vs within-layer-shuffled, same smoothing ----
real_n, ori_r, ctr_r = count(g_pref[sl], g_gate[sl])
sh = shuffled_band(700)
null_n, ori_s, ctr_s = count(sh[sl], g_gate[sl])
fig, axes = plt.subplots(2, 1, figsize=(14, 9))
for ax, om, ct, lab, n in ((axes[0], ori_r, ctr_r, "REAL", real_n),
                           (axes[1], ori_s, ctr_s, "within-layer position shuffle", null_n)):
    ax.imshow(om, cmap="hsv", vmin=0, vmax=180, interpolation="nearest", aspect="auto")
    pos = ct[0]; neg = ct[1] if len(ct) > 1 else np.empty((0, 2))
    for c, cl, ec in ((pos, "k", "none"), (neg, "w", "k")):
        if c.size:
            ax.scatter(c[:, 1], c[:, 0], s=16, c=cl, edgecolors=ec, linewidths=0.5)
    ax.set_title(f"{lab} — deep vision slots {args.deep_slots}, box {args.width}px — "
                 f"{n} pinwheels", fontsize=11)
    ax.set_xlabel("sheet column"); ax.set_ylabel("sheet row (depth)")
fig.suptitle(f"{args.run}: deep vision band analysed as ONE continuous map", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = f"results/{args.run}/orientation_unbiased/deep_band_continuous_boxw{args.width}.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nwrote {out}")
