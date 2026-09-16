"""Scan every contiguous slot range for a pinwheel excess, and price the search.

WHY THIS EXISTS. Picking a slot range after seeing the data is a garden-of-forking-paths:
the null sd is ~5 counts, so scanning enough ranges guarantees one clears p<0.05. The fix
is not to stop looking -- it is to look at ALL of them and compare the best result against
what the best result would be under the null anyway.

Per-tile counts are computed once (real, and once per null draw), so every range is then a
cheap sum. The same null draws are reused across ranges, which is what makes the max-z
null valid: for each draw we recompute the best range, giving the distribution of
"best range z" under no effect.
"""
import argparse
import numpy as np

from src.models.spatial_utils import DEFAULT_SHEET_LAYOUT as S
from src.experiments.orientation.nn_delta import slot_window
from src.experiments.orientation.plot_tiled import analyse

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--window", type=int, default=2)
ap.add_argument("--width", type=int, default=5)
ap.add_argument("--reps", type=int, default=40)
ap.add_argument("--min-span", type=int, default=6, help="shortest range to consider")
ap.add_argument("--report", default="18-28", help="also print this specific range")
args = ap.parse_args()

d = np.load(f"results/{args.run}/orientation_unbiased/orientation_stats.npz")
pref, gate = d["pref_ori"], d["gate"] & d["vision_band"]

slots, real_i, tiles = [], [], []
for s in range(S.num_encoder_layers):
    th = slot_window(pref, S, s, args.window)
    if th is None:
        continue
    ok = (slot_window(gate.astype(float), S, s, args.window) > 0.5) & np.isfinite(th)
    if ok.sum() < 200:
        continue
    slots.append(s); tiles.append((th, ok))
    real_i.append(analyse(th, ok, args.width, 0.3, 30.0, 10.0)[2])
real_i = np.array(real_i, dtype=float)

null_i = np.zeros((args.reps, len(tiles)))
for rep in range(args.reps):
    rng = np.random.default_rng(5000 + rep)
    for j, (th, ok) in enumerate(tiles):
        sh = th.copy(); sh[ok] = rng.permutation(th[ok])
        null_i[rep, j] = analyse(sh, ok, args.width, 0.3, 30.0, 10.0)[2]

def z_of(mask, real_vec, null_mat):
    r = real_vec[mask].sum()
    n = null_mat[:, mask].sum(1)
    sd = n.std(ddof=1)
    return (r - n.mean()) / sd if sd > 0 else -np.inf, r, n.mean(), sd

ranges = [(a, b) for a in range(len(slots)) for b in range(a + args.min_span, len(slots))]
print(f"{args.run}\nwindow={args.window} reps={args.reps}  "
      f"{len(slots)} usable tiles, {len(ranges)} contiguous ranges scanned\n")

lo, hi = (int(x) for x in args.report.split("-"))
m = np.array([lo <= s <= hi for s in slots])
z, r, nm, nsd = z_of(m, real_i, null_i)
print(f"REQUESTED slots {args.report}: real {r:.0f}  null {nm:.1f}+-{nsd:.1f}  z={z:.2f}")

best = max(ranges, key=lambda ab: z_of(
    np.array([slots[ab[0]] <= s <= slots[ab[1]] for s in slots]), real_i, null_i)[0])
mb = np.array([slots[best[0]] <= s <= slots[best[1]] for s in slots])
zb, rb, nmb, nsdb = z_of(mb, real_i, null_i)
print(f"BEST range slots {slots[best[0]]}-{slots[best[1]]}: real {rb:.0f}  "
      f"null {nmb:.1f}+-{nsdb:.1f}  z={zb:.2f}")

# what does the best range look like when there is no effect at all?
maxz_null = []
for rep in range(args.reps):
    held = null_i[rep]
    rest = np.delete(null_i, rep, axis=0)
    maxz_null.append(max(z_of(np.array([slots[a] <= s <= slots[b] for s in slots]),
                              held, rest)[0] for a, b in ranges))
maxz_null = np.array(maxz_null)
p = (float((maxz_null >= zb).sum()) + 1.0) / (len(maxz_null) + 1.0)
print(f"\nNULL best-range z: mean {maxz_null.mean():.2f}, 95th pct "
      f"{np.percentile(maxz_null, 95):.2f}, max {maxz_null.max():.2f}")
print(f"search-corrected p for the best range = {p:.3f}")
print("\nEven with no effect, scanning this many ranges yields a best z of about "
      f"{maxz_null.mean():.1f}. Compare any single range's z against THAT, not against 0.")
