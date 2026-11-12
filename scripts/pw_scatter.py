"""TDANN-style orientation scatter on the unified sheet, with category-cluster outlines.

Style follows plot_pinwheels_panel.plot_panel_scatter: one CIRCLE per gated unit at its
sheet position, hue = preferred orientation, circles sized to OVERLAP (marker_scale > 1)
which is what gives those figures their cobblestone texture. Ungated units are left blank
rather than painted, and draw order is shuffled so no orientation sits systematically on top.

Marker size is derived from the axes geometry after layout, not hardcoded: at a fixed `s`
the circles only overlap at one particular figure size, and on the full 800x576 sheet a
hardcoded small `s` collapses the scatter back into a raster.
"""
import os, argparse, pickle
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize, to_rgb
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from scipy.ndimage import binary_opening, distance_transform_edt
from skimage import measure

from src.models.spatial_utils import DEFAULT_SHEET_LAYOUT as S
from src.experiments.orientation.nn_delta import slot_window, tile_grid_pos
from src.experiments.orientation.plot_tiled import analyse
from src.utils.connected_components import label_islands, island_stats, keep_only_id

ap = argparse.ArgumentParser()
ap.add_argument("--slots", default="0-7")
ap.add_argument("--width", type=int, default=5)
ap.add_argument("--thresh", type=float, default=0.3)
ap.add_argument("--sd-thresh", type=float, default=30.0)
ap.add_argument("--lam", type=float, default=10.0)
ap.add_argument("--top-k-pct", type=float, default=10.0)
ap.add_argument("--mark-centres", action="store_true")
ap.add_argument("--cmap", default="hsv")
ap.add_argument("--mix", type=float, default=0.72,
                help="0=page grey, 1=full-saturation base colormap")
ap.add_argument("--alpha", type=float, default=0.85)
ap.add_argument("--marker-scale", type=float, default=1.9,
                help=">1 makes circles overlap (the cobblestone look). Size is computed "
                     "from the axes width, so this is scale-independent.")
ap.add_argument("--jitter", type=float, default=0.0,
                help="display-only uniform jitter in sheet px, to break the integer lattice")
ap.add_argument("--fade-px", type=float, default=0.0,
                help="fade markers within this many px of the painted region's edge")
ap.add_argument("--zoom", default=None, help="crop as r0,r1,c0,c1 in DISPLAY coords")
ap.add_argument("--figw", type=float, default=9.0)
ap.add_argument("--ori-run", default="tiled32-4608-snap438-nosmooth")
ap.add_argument("--sel-run", default="topo-omni-tiled32-4608-snap438")
ap.add_argument("--tag", default="")
a = ap.parse_args()

H, W = S.n_row, S.n_col
THK, MID = S.thinker_rows, S.n_col // 2
eth, etw, _, enc_n = S.enc_tile
CATS = [("faces", "vision", "#A04D3B", "Faces"), ("bodies", "vision", "#BF8649", "Bodies"),
        ("scenes", "vision", "#CDB34C", "Scenes"), ("objects", "vision", "#E7DFA1", "Objects"),
        ("vwfa", "vision", "#877C26", "Words"), ("speech", "audio", "#485682", "Speech"),
        ("vocals", "audio", "#61A6A4", "Vocals"),
        ("theory_of_mind_text_ALL", "cognitive", "#92CBC3", "Theory of Mind"),
        ("language_text_ALL", "language", "#478190", "Language"),
        ("multiple_demand_text_ALL", "cognitive", "#303557", "Multiple Demand")]

def place(flat):
    return np.rot90(np.asarray(flat, float).reshape(W, H), k=1)

keep = list(range(S.num_encoder_layers)) if a.slots == "all" else \
       list(range(int(a.slots.split("-")[0]), int(a.slots.split("-")[1]) + 1))
d = np.load(f"results/{a.ori_run}/orientation_unbiased/orientation_stats.npz")
pref, gate = d["pref_ori"], d["gate"] & d["vision_band"]

ori_flat = np.full(S.num_units, np.nan)
ctr_pos, ctr_neg = [], []
for s in keep:
    theta = slot_window(pref, S, s, 1)
    if theta is None: continue
    valid = (slot_window(gate.astype(float), S, s, 1) > 0.5) & np.isfinite(theta)
    if valid.sum() < 200: continue
    ori, ctr, n, _ = analyse(theta, valid, a.width, a.thresh, a.sd_thresh, a.lam)
    ori = np.where(valid, ori, np.nan)          # leave ungated units BLANK, as in the reference
    tr, tc = tile_grid_pos(s, enc_n, S.tile_order == "serpentine")
    r0, c0 = tr * eth, tc * etw
    ii, jj = np.nonzero(np.isfinite(ori))
    ori_flat[(c0 + jj) * H + (r0 + ii)] = ori[ii, jj]
    for arr, sink in ((ctr[0], ctr_pos), (ctr[1] if len(ctr) > 1 else np.empty((0, 2)), ctr_neg)):
        for i, j in np.asarray(arr).reshape(-1, 2):
            sink.append((799 - (r0 + i), c0 + j))
ori_g = place(ori_flat)
print(f"slots {keep[0]}..{keep[-1]}: {int(np.isfinite(ori_g).sum())} gated units, "
      f"{len(ctr_pos)} (+) and {len(ctr_neg)} (-) pinwheels")

def largest_island(cat, modality):
    st = pickle.load(open(f"results/{a.sel_run}/{cat}/{cat}_fwhm3.0_selectivity_stats.pkl", "rb"))
    t = np.asarray(st["t"], float)
    band = np.zeros((H, W), bool)
    if modality in ("language", "cognitive"):
        band[:THK, :] = True; denom = THK * W
    elif modality == "audio":
        band[THK:, MID:] = True; denom = S.encoder_rows * MID
    else:
        band[THK:, :MID] = True; denom = S.encoder_rows * MID
    idx = place(np.arange(t.size, dtype=float)); g = idx[band]
    allowed = np.zeros(t.size, bool); allowed[g[np.isfinite(g)].astype(np.int64)] = True
    rank = np.where(np.isfinite(t) & allowed, t, -np.inf)
    m = np.zeros(t.size, bool); m[np.argsort(rank)[::-1][:int(a.top_k_pct / 100 * denom)]] = True
    clean = binary_opening(place(m) > 0.5, structure=np.ones((3, 3)))
    lab = measure.label(clean); out = np.zeros_like(clean)
    for reg in measure.regionprops(lab):
        if reg.area >= 20: out[lab == reg.label] = 1
    ml, ni = label_islands(out.astype(np.uint8), connectivity=8)
    sts = island_stats(ml, ni, place(t))
    return (keep_only_id(ml, keep_id=sts[0]["id"]).astype(bool), sts[0]["size"]) if sts else (None, 0)

r0v, r1v, c0v, c1v = (0, H, 0, W) if not a.zoom else tuple(int(x) for x in a.zoom.split(","))
aspect = (r1v - r0v) / max(c1v - c0v, 1)
fig, ax = plt.subplots(figsize=(a.figw, a.figw * aspect), dpi=300)
ax.set_facecolor("white")

_bg = np.array(to_rgb("#E9E9E9"))
_base = plt.get_cmap(a.cmap)(np.linspace(0, 1, 256))[:, :3]
muted = ListedColormap(np.clip(a.mix * _base + (1.0 - a.mix) * _bg, 0, 1))
norm = Normalize(0, 180)

m = np.isfinite(ori_g)
ys, xs = np.nonzero(m)
vals = ori_g[m]
if a.fade_px > 0:
    fade = np.clip(distance_transform_edt(np.pad(m, 1)) [1:-1, 1:-1] / a.fade_px, 0, 1)[m]
else:
    fade = np.ones(vals.size)
rng = np.random.default_rng(0)
x = xs.astype(float); y = ys.astype(float)
if a.jitter:
    x += rng.uniform(-a.jitter, a.jitter, x.size); y += rng.uniform(-a.jitter, a.jitter, y.size)
order = np.random.default_rng(1).permutation(vals.size)   # no orientation systematically on top
rgba = muted(norm(vals[order])); rgba[:, 3] = a.alpha * fade[order]
sc = ax.scatter(x[order], y[order], c=rgba, marker="o", s=4, linewidths=0, zorder=2)

handles = []
for cat, modality, col, lbl in CATS:
    isl, sz = largest_island(cat, modality)
    if isl is None:
        print(f"  {cat}: no island"); continue
    print(f"  {cat:26s} {modality:10s} {sz:5d} units")
    ax.contour(isl.astype(float), levels=[0.5], colors=[col], linewidths=1.6)
    handles.append(Line2D([0], [0], color=col, lw=2, label=lbl))

if a.mark_centres:
    for pts, fc, ec, lb in ((ctr_pos, "k", "none", f"+{len(ctr_pos)}"),
                            (ctr_neg, "w", "k", f"−{len(ctr_neg)}")):
        if pts:
            py, px = zip(*pts)
            ax.scatter(px, py, s=18, facecolors=fc, edgecolors=ec, linewidths=0.5,
                       alpha=0.8, zorder=6, label=lb)

ax.axhline(y=THK, color="gray", ls="--", lw=1)
ax.vlines(x=MID, ymin=THK, ymax=H, color="gray", ls="--", lw=1)
ax.set_xlim(c0v - 0.5, c1v - 0.5); ax.set_ylim(r1v - 0.5, r0v - 0.5)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

# Size the circles AFTER layout: s is in pt^2, so overlap depends on how many points one
# sheet column occupies on screen. Hardcoding s only overlaps at one figure size.
fig.canvas.draw()
bb = ax.get_window_extent(renderer=fig.canvas.get_renderer())
pts_per_col = (bb.width * 72.0 / fig.dpi) / max(c1v - c0v, 1)
sc.set_sizes(np.full(vals.size, (a.marker_scale * pts_per_col) ** 2))
print(f"marker: {pts_per_col:.3f} pt per sheet column -> s={(a.marker_scale * pts_per_col) ** 2:.2f} pt^2")

if handles:
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.12, 1.0), frameon=False, fontsize=8)
cb = fig.colorbar(ScalarMappable(norm=norm, cmap=muted), ax=ax, fraction=0.03, pad=0.02,
                  ticks=[0, 45, 90, 135, 180])
cb.set_label("preferred orientation (deg)")
tag = a.slots.replace("-", "to") + (f"_{a.tag}" if a.tag else "")
out = f"results/{a.sel_run}/unified_map_pinwheels/unified_ori_scatter_slots{tag}.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=300, bbox_inches="tight")
print("saved", out)
