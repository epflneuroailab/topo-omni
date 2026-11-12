"""Pinwheel centres and category-selective clusters on ONE unified sheet.

Placement: pinwheel centres are detected per TILE (a tiled layer is a 64x72 tile; a contour
walked across a tile border crosses a depth discontinuity, so cross-tile winding numbers are
layout artefacts). Each tile-local centre is converted to a FLAT unit index and then placed
through the identical rot90(reshape(n_col, n_row)) used for the category maps -- so the two
overlays cannot drift apart through an orientation mistake.
"""
import numpy as np, pickle, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import binary_opening
from skimage import measure

from src.models.spatial_utils import DEFAULT_SHEET_LAYOUT as S
from src.experiments.orientation.nn_delta import slot_window
from src.experiments.orientation.plot_tiled import analyse
from src.experiments.orientation.nn_delta import tile_grid_pos
from src.utils.connected_components import label_islands, island_stats, keep_only_id

H, W = S.n_row, S.n_col
THK, MID = S.thinker_rows, S.n_col // 2
RUN_ORI = "tiled32-4608-snap438-nosmooth"
RUN_SEL = "topo-omni-tiled32-4608-snap438"
WIDTH, THRESH, SD_THRESH, LAM = 5, 0.3, 30.0, 10.0
TOPK, PTHR = 10.0, 0.001

CATS = [("faces", "vision", "#A04D3B"), ("bodies", "vision", "#BF8649"),
        ("scenes", "vision", "#CDB34C"), ("objects", "vision", "#E7DFA1"),
        ("vwfa", "vision", "#877C26"), ("speech", "audio", "#485682"),
        ("vocals", "audio", "#61A6A4")]

def place(flat):
    return np.rot90(np.asarray(flat, float).reshape(W, H), k=1)

# ---- pinwheel centres -> flat indices -------------------------------------------------
d = np.load(f"results/{RUN_ORI}/orientation_unbiased/orientation_stats.npz")
pref, gate = d["pref_ori"], d["gate"] & d["vision_band"]
eth, etw, _, enc_n = S.enc_tile
pos_flat = np.zeros(S.num_units, bool)
neg_flat = np.zeros(S.num_units, bool)
n_pos = n_neg = 0
for s in range(S.num_encoder_layers):
    theta = slot_window(pref, S, s, 1)
    if theta is None:
        continue
    valid = (slot_window(gate.astype(float), S, s, 1) > 0.5) & np.isfinite(theta)
    if valid.sum() < 200:
        continue
    _, ctr, n, _ = analyse(theta, valid, WIDTH, THRESH, SD_THRESH, LAM)
    tr, tc = tile_grid_pos(s, enc_n, S.tile_order == "serpentine")
    r0, c0 = tr * eth, tc * etw
    for arr, sink in ((ctr[0], pos_flat), (ctr[1] if len(ctr) > 1 else np.empty((0, 2)), neg_flat)):
        for i, j in np.asarray(arr).reshape(-1, 2):
            r, c = int(round(r0 + i)), int(round(c0 + j))
            if 0 <= r < H and 0 <= c < W:
                sink[c * H + r] = True          # grid = flat.reshape(n_col, n_row) -> [col, row]
    n_pos += len(ctr[0]); n_neg += len(ctr[1]) if len(ctr) > 1 else 0
print(f"pinwheels placed: {n_pos} positive, {n_neg} negative ({n_pos + n_neg} total)")
pos_g, neg_g = place(pos_flat) > 0.5, place(neg_flat) > 0.5

# ---- category clusters (same recipe as the unified map) --------------------------------
def largest_island(cat, modality):
    st = pickle.load(open(f"results/{RUN_SEL}/{cat}/{cat}_fwhm3.0_selectivity_stats.pkl", "rb"))
    t_flat = np.asarray(st["t"], float)
    band = np.zeros((H, W), bool)
    if modality == "vision":
        band[THK:, :MID] = True
    else:
        band[THK:, MID:] = True
    idx = place(np.arange(t_flat.size, dtype=float))
    allowed = np.zeros(t_flat.size, bool)
    g = idx[band]
    allowed[g[np.isfinite(g)].astype(np.int64)] = True
    rank = np.where(np.isfinite(t_flat) & allowed, t_flat, -np.inf)
    n_keep = int(TOPK / 100 * S.encoder_rows * MID)
    m = np.zeros(t_flat.size, bool); m[np.argsort(rank)[::-1][:n_keep]] = True
    sel = place(m) > 0.5
    clean = binary_opening(sel, structure=np.ones((3, 3)))
    lab = measure.label(clean); out = np.zeros_like(clean)
    for reg in measure.regionprops(lab):
        if reg.area >= 20:
            out[lab == reg.label] = 1
    ml, ni = label_islands(out.astype(np.uint8), connectivity=8)
    sts = island_stats(ml, ni, place(t_flat))
    if not sts:
        return None, 0
    return keep_only_id(ml, keep_id=sts[0]["id"]).astype(bool), sts[0]["size"]

fig, ax = plt.subplots(figsize=(9.0, 9.0 * H / W), dpi=300)
ax.imshow(np.ones((H, W)), cmap=matplotlib.colors.ListedColormap(["#E7E7E7"]),
          vmin=0, vmax=1, interpolation="nearest")
for cat, modality, col in CATS:
    isl, sz = largest_island(cat, modality)
    if isl is None:
        print(f"  {cat}: no island"); continue
    print(f"  {cat}: {sz} units")
    ax.imshow(np.ma.masked_where(~isl, np.ones((H, W))),
              cmap=matplotlib.colors.ListedColormap([col]), vmin=0, vmax=1,
              interpolation="nearest")

for g, fc, ec, lbl in ((pos_g, "k", "none", "+1"), (neg_g, "w", "k", "-1")):
    ys, xs = np.nonzero(g)
    if len(ys):
        ax.scatter(xs, ys, s=55, c=fc, edgecolors=ec, linewidths=0.8, zorder=5, label=f"pinwheel {lbl}")

ax.axhline(y=THK, color="gray", ls="--", lw=1)
ax.vlines(x=MID, ymin=THK, ymax=H, color="gray", ls="--", lw=1)
ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(H - 0.5, -0.5)
if n_pos + n_neg:
    ax.legend(loc="upper right", frameon=True, fontsize=8)
out = f"results/{RUN_SEL}/unified_map_pinwheels/unified_clusters_pinwheels.png"
import os; os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=300, bbox_inches="tight")
print("saved", out)
