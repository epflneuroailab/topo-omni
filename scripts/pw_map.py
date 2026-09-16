"""Pinwheel (orientation-preference) MAP on the unified sheet, with category clusters.

The map is built per TILE and then placed by flat unit index through the same
rot90(reshape(n_col, n_row)) the category maps use, so the two overlays cannot drift apart.
Only the requested vision-encoder slots are painted; everything else stays grey.

Clusters are drawn as OUTLINES, not fills -- a filled cluster would hide the orientation map
underneath, which is the thing being shown.
"""
import os, argparse, pickle
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb
from matplotlib.cm import ScalarMappable
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import binary_opening
from skimage import measure

from src.models.spatial_utils import DEFAULT_SHEET_LAYOUT as S
from src.experiments.orientation.nn_delta import slot_window, tile_grid_pos
from src.experiments.orientation.plot_tiled import analyse
from src.utils.connected_components import label_islands, island_stats, keep_only_id

ap = argparse.ArgumentParser()
ap.add_argument("--slots", default="0-7", help="vision-encoder slot range to paint, e.g. '0-7' or 'all'")
ap.add_argument("--width", type=int, default=5)
ap.add_argument("--thresh", type=float, default=0.3)
ap.add_argument("--sd-thresh", type=float, default=30.0)
ap.add_argument("--lam", type=float, default=10.0)
ap.add_argument("--top-k-pct", type=float, default=10.0)
ap.add_argument("--mark-centres", action="store_true", help="also dot the detected centres")
ap.add_argument("--cmap", default="hsv", help="cyclic base colormap (hsv, twilight, ...)")
ap.add_argument("--mix", type=float, default=0.5,
                help="0=pure background grey, 1=full-saturation base colormap. The hue "
                     "still carries orientation, it is just blended toward the page.")
ap.add_argument("--alpha", type=float, default=0.75, help="peak marker alpha")
ap.add_argument("--fade-px", type=float, default=14.0,
                help="width in sheet px over which markers fade out approaching the edge "
                     "of the painted region (0 disables)")
ap.add_argument("--marker-size", type=float, default=2.2, help="scatter marker area (pt^2)")
ap.add_argument("--ori-run", default="tiled32-4608-snap438-nosmooth")
ap.add_argument("--sel-run", default="topo-omni-tiled32-4608-snap438")
a = ap.parse_args()

H, W = S.n_row, S.n_col
THK, MID = S.thinker_rows, S.n_col // 2
eth, etw, _, enc_n = S.enc_tile
# (name, modality, colour, legend label). Colours are selectivity.py's palette, indexed by
# the label each category maps to there: Faces 0, Bodies 1, Scenes 2, Objects 3, Words 4,
# False Belief 5, False Photo 6, Nonwords 7, Quilted Speech 8, Math 9.
CATS = [("faces", "vision", "#E64B35", "Faces"),
        ("bodies", "vision", "#F39B2F", "Bodies"),
        ("scenes", "vision", "#F1C232", "Scenes"),
        ("objects", "vision", "#EFC94C", "Objects"),
        ("vwfa", "vision", "#D4AC0D", "Words"),
        ("speech", "audio", "#3B5B92", "Speech"),
        ("vocals", "audio", "#4C9F9B", "Vocals"),
        ("theory_of_mind_text_ALL", "cognitive", "#5AB4AC", "Theory of Mind"),
        ("language_text_ALL", "language", "#3F8F9C", "Language"),
        ("multiple_demand_text_ALL", "cognitive", "#2F3E75", "Multiple Demand")]

def place(flat):
    return np.rot90(np.asarray(flat, float).reshape(W, H), k=1)

keep = list(range(S.num_encoder_layers)) if a.slots == "all" else \
       list(range(int(a.slots.split("-")[0]), int(a.slots.split("-")[1]) + 1))
print(f"painting slots {keep[0]}..{keep[-1]} ({len(keep)} of {S.num_encoder_layers})")

d = np.load(f"results/{a.ori_run}/orientation_unbiased/orientation_stats.npz")
pref, gate = d["pref_ori"], d["gate"] & d["vision_band"]

ori_flat = np.full(S.num_units, np.nan)
ctr_pos, ctr_neg = [], []
n_pw = 0
for s in keep:
    theta = slot_window(pref, S, s, 1)
    if theta is None: continue
    valid = (slot_window(gate.astype(float), S, s, 1) > 0.5) & np.isfinite(theta)
    if valid.sum() < 200:
        print(f"  slot {s}: only {int(valid.sum())} valid units, skipped"); continue
    ori, ctr, n, _ = analyse(theta, valid, a.width, a.thresh, a.sd_thresh, a.lam)
    n_pw += int(n)
    tr, tc = tile_grid_pos(s, enc_n, S.tile_order == "serpentine")
    r0, c0 = tr * eth, tc * etw
    for i in range(ori.shape[0]):
        for j in range(ori.shape[1]):
            v = ori[i, j]
            if np.isfinite(v):
                ori_flat[(c0 + j) * H + (r0 + i)] = v
    for arr, sink in ((ctr[0], ctr_pos), (ctr[1] if len(ctr) > 1 else np.empty((0, 2)), ctr_neg)):
        for i, j in np.asarray(arr).reshape(-1, 2):
            sink.append((799 - (r0 + i), c0 + j))
print(f"pinwheels in painted slots: {n_pw}")

ori_g = place(ori_flat)

def largest_island(cat, modality):
    st = pickle.load(open(f"results/{a.sel_run}/{cat}/{cat}_fwhm3.0_selectivity_stats.pkl", "rb"))
    t = np.asarray(st["t"], float)
    # The three regions partition the sheet: thinker band on top (all columns), then the
    # vision and audio encoder halves below. language/cognitive localizers live in the
    # thinker band, so their denominator is THK*n_col, not encoder_rows*MID.
    band = np.zeros((H, W), bool)
    if modality in ("language", "cognitive"):
        band[:THK, :] = True
        denom = THK * W
    elif modality == "audio":
        band[THK:, MID:] = True
        denom = S.encoder_rows * MID
    elif modality == "vision":
        band[THK:, :MID] = True
        denom = S.encoder_rows * MID
    else:
        raise SystemExit(f"no band for modality {modality!r}")
    idx = place(np.arange(t.size, dtype=float)); g = idx[band]
    allowed = np.zeros(t.size, bool); allowed[g[np.isfinite(g)].astype(np.int64)] = True
    rank = np.where(np.isfinite(t) & allowed, t, -np.inf)
    m = np.zeros(t.size, bool)
    m[np.argsort(rank)[::-1][:int(a.top_k_pct / 100 * denom)]] = True
    clean = binary_opening(place(m) > 0.5, structure=np.ones((3, 3)))
    lab = measure.label(clean); out = np.zeros_like(clean)
    for reg in measure.regionprops(lab):
        if reg.area >= 20: out[lab == reg.label] = 1
    ml, ni = label_islands(out.astype(np.uint8), connectivity=8)
    sts = island_stats(ml, ni, place(t))
    return (keep_only_id(ml, keep_id=sts[0]["id"]).astype(bool), sts[0]["size"]) if sts else (None, 0)

fig, ax = plt.subplots(figsize=(9.0, 9.0 * H / W), dpi=300)
ax.imshow(np.ones((H, W)), cmap=ListedColormap(["#E9E9E9"]), vmin=0, vmax=1, interpolation="nearest")
# Muted cyclic colormap: blend the base map toward the page background so the hues stay
# distinguishable (they encode orientation) without the poster-paint look of raw hsv.
_bg = np.array(to_rgb("#E9E9E9"))
_base = plt.get_cmap(a.cmap)(np.linspace(0, 1, 256))[:, :3]
muted = ListedColormap(np.clip(a.mix * _base + (1.0 - a.mix) * _bg, 0, 1))
norm = Normalize(0, 180)

# Scatter, one marker per painted unit, with a per-point alpha that fades toward the edge
# of the painted region -- so the block does not end in a hard rectangle.
_m = np.isfinite(ori_g)
if a.fade_px > 0:
    _fade = np.clip(distance_transform_edt(_m) / a.fade_px, 0.0, 1.0)
else:
    _fade = np.ones_like(ori_g, dtype=float)
_ys, _xs = np.nonzero(_m)
_rgba = muted(norm(ori_g[_m]))
_rgba[:, 3] = a.alpha * _fade[_m]
ax.scatter(_xs, _ys, c=_rgba, s=a.marker_size, marker="o", linewidths=0, zorder=2)
im = ScalarMappable(norm=norm, cmap=muted)

handles = []
for cat, modality, col, lbl in CATS:
    try:
        isl, sz = largest_island(cat, modality)
    except FileNotFoundError:
        print(f"  {cat}: no stats file, skipped"); continue
    if isl is None:
        print(f"  {cat}: no island survived"); continue
    print(f"  {cat:26s} {modality:10s} {sz:5d} units")
    ax.contour(isl.astype(float), levels=[0.5], colors=[col], linewidths=1.6)
    handles.append(Line2D([0], [0], color=col, lw=2, label=lbl))

if a.mark_centres:
    for pts, fc, ec in ((ctr_pos, "k", "none"), (ctr_neg, "w", "k")):
        if pts:
            ys, xs = zip(*pts)
            ax.scatter(xs, ys, s=26, facecolors=fc, edgecolors=ec, linewidths=0.6,
                       alpha=0.65, zorder=6)

ax.axhline(y=THK, color="gray", ls="--", lw=1)
ax.vlines(x=MID, ymin=THK, ymax=H, color="gray", ls="--", lw=1)
ax.set_xticks([]); ax.set_yticks([])
if handles:
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.10, 1.0),
              frameon=False, fontsize=8)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, ticks=[0, 45, 90, 135, 180])
cb.set_label("preferred orientation (deg)")
tag = a.slots.replace("-", "to")
out = f"results/{a.sel_run}/unified_map_pinwheels/unified_orientation_map_slots{tag}.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=300, bbox_inches="tight")
print("saved", out)
