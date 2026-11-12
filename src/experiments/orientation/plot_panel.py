"""Paper-panel version of ``orientation_metrics.plot_pinwheels``: a smoothed orientation
map with pinwheel centres overlaid, plus a cyclic colour bar, optionally restricted to a
contiguous set of vision-encoder layers.

``orientation_metrics.py`` deliberately keeps the map and the pinwheel counts on the FULL
165-row vision band even when ``--layers`` is given, because a 4-slot strip is 20 sheet
rows -- well under one column spacing (Lambda ~ 32 px) -- so a 2D defect count from it is
not meaningful. This script exists for the figure panel: it will crop the map to the
selected layers when asked, and prints how many hypercolumns tall that leaves.

Usage (from the repo root, with PYTHONPATH=.):
    python -m src.experiments.orientation.plot_panel \
        --stats results/topo-omni/orientation_unbiased/orientation_stats.npz \
        --layers 0-3 --out results/topo-omni/orientation_unbiased/panels/panel_pinwheels.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, Normalize, hsv_to_rgb, rgb_to_hsv
from matplotlib.transforms import offset_copy

from src.experiments.orientation.metrics import (
    N_VISION_LAYERS, ROWS_PER_VISION_LAYER, band_grid, column_spacing,
    count_pinwheels, layer_band_mask, parse_layers, pinwheel_density,
    smoothed_orientation_map, winding_numbers,
)


def smoothed_orientation_map_same(pref_ori_2d, valid, width):
    """Same as ``smoothed_orientation_map`` but stride 1 with centred, edge-clipped windows,
    so the output has exactly the input shape.

    ``smoothed_orientation_map`` samples windows at ``stride`` and only where a full window
    fits, which turns a 20-row layer strip into 8 rows. For a figure panel we want every
    sheet row kept, so here each pixel gets the circular mean over the ``width x width``
    window centred on it, truncated at the border (like a 'same' convolution).
    """
    theta = np.asarray(pref_ori_2d, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    z = np.where(valid, np.exp(2j * np.deg2rad(np.nan_to_num(theta))), 0.0)
    ny, nx = theta.shape
    r = width // 2

    cz = np.pad(np.cumsum(np.cumsum(z, axis=0), axis=1), ((1, 0), (1, 0)))
    cv = np.pad(np.cumsum(np.cumsum(valid.astype(float), axis=0), axis=1), ((1, 0), (1, 0)))

    i = np.arange(ny)
    j = np.arange(nx)
    r0, r1 = np.clip(i - r, 0, ny), np.clip(i + r + 1, 0, ny)
    c0, c1 = np.clip(j - r, 0, nx), np.clip(j + r + 1, 0, nx)

    def box(cum):
        return (cum[np.ix_(r1, c1)] - cum[np.ix_(r0, c1)]
                - cum[np.ix_(r1, c0)] + cum[np.ix_(r0, c0)])

    cnt = box(cv)
    ori = np.full((ny, nx), np.nan)
    sd = np.full((ny, nx), np.nan)
    ok = cnt >= 1
    zbar = np.zeros((ny, nx), dtype=complex)
    zbar[ok] = box(cz)[ok] / cnt[ok]
    ori[ok] = (np.rad2deg(np.angle(zbar[ok])) / 2.0) % 180.0
    rlen = np.minimum(np.abs(zbar), 1.0)
    good = ok & (rlen > 1e-12)
    sd[good] = np.rad2deg(np.sqrt(-2.0 * np.log(rlen[good]))) / 2.0
    sd[ok & ~good] = 180.0
    return ori, sd, np.rint(cnt).astype(int)


def muted_cyclic(sat=0.70, val=0.90, n=256, name="hsv_muted"):
    """``hsv`` pulled toward pastel, so a Topo-Omni panel sits next to TDANN's without
    shouting. TDANN's orientation maps use a desaturated cyclic palette; ``hsv`` is fully
    saturated, which reads as much louder at panel size. Stays cyclic and keeps hue order,
    so 0 and 180 deg remain the same colour.
    """
    rgb = plt.get_cmap("hsv")(np.linspace(0.0, 1.0, n))[:, :3]
    h = rgb_to_hsv(rgb)
    h[:, 1] *= sat
    h[:, 2] *= val
    return ListedColormap(hsv_to_rgb(h), name=name)


CMAPS = {"hsv": lambda: plt.get_cmap("hsv"), "muted": muted_cyclic}


def _sizes_in_points(ax, fig, scale=1.0):
    """Marker area (pt^2) that makes one data unit one marker wide, at the current axes size."""
    p0 = ax.transData.transform((0.0, 0.0))
    p1 = ax.transData.transform((1.0, 0.0))
    px_per_unit = abs(p1[0] - p0[0])
    pt_per_unit = px_per_unit * 72.0 / fig.dpi
    return max(0.5, (pt_per_unit * scale) ** 2)


def plot_panel_scatter(ori_map, valid, centres, savepath, title=None, figsize=None,
                       dpi=400, marker="o", marker_scale=1.9, show_counts=True,
                       pinwheel_size=16, cbar=True, jitter=0.0, alpha=1.0,
                       edge=0.0, rng_seed=0, cbar_orientation=None,
                       scalebar_px=None, scalebar_label=r"$\Lambda$",
                       cbar_label="Orientation (deg)", cmap="hsv"):
    """TDANN-style scatter: one circle per unit at its sheet position, hue = preferred
    orientation. Unlike the heatmap this leaves ungated units as blank background rather
    than white pixels inside the image, and shows per-unit preferences directly.

    ``marker_scale`` > 1 makes the circles overlap, which is what gives TDANN's figures
    their cobblestone look; their units sit at continuous positions, ours on the integer
    sheet lattice, so ``jitter`` (in sheet px, uniform) can be used to break the visible
    grid. Jitter is display-only -- it moves nothing that any metric is computed on -- and
    defaults to 0.
    """
    ny, nx = ori_map.shape
    rr, cc = np.nonzero(valid & np.isfinite(ori_map))
    vals = ori_map[rr, cc]

    if figsize is None:
        aspect = nx / max(ny, 1)
        width_in = min(7.0, max(3.2, 0.055 * nx))
        figsize = (width_in, max(1.15, width_in / aspect) + 0.55)

    x = cc.astype(float)
    y = rr.astype(float)
    if jitter:
        g = np.random.default_rng(rng_seed)
        x = x + g.uniform(-jitter, jitter, x.size)
        y = y + g.uniform(-jitter, jitter, y.size)

    fig, ax = plt.subplots(figsize=figsize)
    # draw in a fixed shuffled order so no orientation is systematically on top
    order = np.random.default_rng(rng_seed + 1).permutation(vals.size)
    sc = ax.scatter(x[order], y[order], c=vals[order], cmap=cmap, norm=Normalize(0, 180),
                    marker=marker, s=4, alpha=alpha,
                    linewidths=edge, edgecolors="none" if edge <= 0 else "0.25")
    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(ny - 0.5, -0.5)                 # match imshow's origin='upper'
    ax.set_aspect("equal")

    if len(centres) and centres[0].size:
        ax.scatter(centres[0][:, 1], centres[0][:, 0], c="k", s=pinwheel_size,
                   linewidths=0, zorder=3, label=f"+{len(centres[0])}")
    if len(centres) > 1 and centres[1].size:
        ax.scatter(centres[1][:, 1], centres[1][:, 0], c="w", s=pinwheel_size,
                   edgecolors="k", linewidths=0.45, zorder=3,
                   label=f"\u2212{len(centres[1])}")

    n = sum(c.shape[0] for c in centres)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
    ax.set_title(title if title is not None
                 else f"Smoothed orientation map \u2014 {n} pinwheels", fontsize=10, pad=6)
    if show_counts and n:
        ax.legend(frameon=False, fontsize=8, loc="upper right", handletextpad=0.15,
                  labelspacing=0.25, borderaxespad=0.25)

    if cbar:
        cb = fig.colorbar(sc, ax=ax,
                          orientation=cbar_orientation or
                          ("horizontal" if nx / max(ny, 1) > 3 else "vertical"),
                          fraction=0.045, pad=0.03, shrink=0.8, aspect=22,
                          ticks=[0, 45, 90, 135, 180])
        cb.set_label(cbar_label, fontsize=9)
        cb.ax.tick_params(labelsize=8, length=2.5, width=0.7)
        cb.outline.set_linewidth(0.7)

    if scalebar_px:
        add_lambda_scalebar(ax, scalebar_px, label=scalebar_label)

    fig.tight_layout()
    fig.canvas.draw()
    sc.set_sizes(np.full(vals.size, _sizes_in_points(ax, fig, marker_scale)))

    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
        fig.savefig(str(savepath).replace(".png", ".pdf"), bbox_inches="tight")
        print(f"> wrote {savepath} (and .pdf)")
    return fig, ax


def apply_zoom(field, valid, centres, box):
    """Crop the display window to ``box`` = (r0, r1, c0, c1) and rebase pinwheel centres.

    The smoothing and the pinwheel detection run on the full selected region first, so
    windows near the edge of the zoom still see their real neighbours -- only the display
    is cropped, as in TDANN's zoomed panel.
    """
    r0, r1, c0, c1 = box
    out = []
    for c in centres:
        if c.size == 0:
            out.append(c.reshape(-1, 2))
            continue
        keep = ((c[:, 0] >= r0) & (c[:, 0] < r1) & (c[:, 1] >= c0) & (c[:, 1] < c1))
        out.append((c[keep] - np.array([r0, c0])).reshape(-1, 2))
    return field[r0:r1, c0:c1], valid[r0:r1, c0:c1], out


def add_lambda_scalebar(ax, lam_px, label=r"$\Lambda$", pad_pt=7.0, color="k", fontsize=9):
    """Horizontal bar one column spacing long, TDANN's '1mm' equivalent for a model sheet.

    Offsets are in POINTS, not data units: a 20-row slot strip and a 165-row band have
    wildly different data heights, and a padding expressed as a fraction of that would put
    the label on top of the bar in the short one.
    """
    x0, x1 = ax.get_xlim()
    y0, _ = ax.get_ylim()                       # image coords: y0 is the BOTTOM edge
    xs = x1 - 0.02 * abs(x1 - x0) - lam_px
    tr_bar = offset_copy(ax.transData, fig=ax.figure, y=-pad_pt, units="points")
    tr_txt = offset_copy(ax.transData, fig=ax.figure, y=-(pad_pt + 2.5), units="points")
    ax.plot([xs, xs + lam_px], [y0, y0], color=color, lw=2.4, solid_capstyle="butt",
            transform=tr_bar, clip_on=False, zorder=5)
    ax.text(xs + lam_px / 2.0, y0, label, ha="center", va="top", transform=tr_txt,
            fontsize=fontsize, color=color, clip_on=False)


def parse_box(spec, shape):
    """``'r0,r1,c0,c1'`` or ``'squareN'`` / ``'sqN'`` for a centred N x N window."""
    ny, nx = shape
    spec = str(spec).strip().lower().replace(" ", "")
    if spec.startswith(("square", "sq")):
        n = int(spec.lstrip("squarei"))
        n = min(n, ny, nx)
        r0 = (ny - n) // 2
        c0 = (nx - n) // 2
        return r0, r0 + n, c0, c0 + n
    r0, r1, c0, c1 = (int(v) for v in spec.split(","))
    return max(0, r0), min(ny, r1), max(0, c0), min(nx, c1)


def plot_panel(ori_map, centres, savepath, lam_px=None, stride=None,
               crop_rows=None, title=None, figsize=None, dpi=400,
               show_counts=True, marker_size=16, cbar_orientation=None,
               scalebar_px=None, scalebar_label=r"$\Lambda$",
               cbar_label="Orientation (deg)", cmap="hsv"):
    """Smoothed orientation map + pinwheel centres + a cyclic 0-180 deg colour bar."""
    h, w = ori_map.shape
    if figsize is None:
        # keep pixels square, cap the width so a thin strip stays legible
        aspect = w / max(h, 1)
        width_in = min(7.0, max(3.2, 0.055 * w))
        figsize = (width_in, max(1.15, width_in / aspect) + 0.55)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(ori_map, cmap=cmap, vmin=0, vmax=180,
                   interpolation="nearest", origin="upper")

    if len(centres) and centres[0].size:
        ax.scatter(centres[0][:, 1], centres[0][:, 0], c="k", s=marker_size,
                   linewidths=0, zorder=3, label=f"+{len(centres[0])}")
    if len(centres) > 1 and centres[1].size:
        ax.scatter(centres[1][:, 1], centres[1][:, 0], c="w", s=marker_size,
                   edgecolors="k", linewidths=0.45, zorder=3,
                   label=f"−{len(centres[1])}")

    n = sum(c.shape[0] for c in centres)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.8)
    if title is None:
        title = f"Smoothed orientation map — {n} pinwheels"
    ax.set_title(title, fontsize=10, pad=6)
    if show_counts and n:
        ax.legend(frameon=False, fontsize=8, loc="upper right",
                  handletextpad=0.15, labelspacing=0.25,
                  borderaxespad=0.25, labelcolor="k")

    # cyclic colour bar: 0 and 180 deg are the same orientation, so label both ends
    cb = fig.colorbar(im, ax=ax,
                      orientation=cbar_orientation or
                      ("horizontal" if w / max(h, 1) > 3 else "vertical"),
                      fraction=0.045, pad=0.03, shrink=0.8, aspect=22,
                      ticks=[0, 45, 90, 135, 180])
    cb.set_label(cbar_label, fontsize=9)
    cb.ax.set_yticklabels(["0", "45", "90", "135", "180"]) if cb.orientation == "vertical" \
        else cb.ax.set_xticklabels(["0", "45", "90", "135", "180"])
    cb.ax.tick_params(labelsize=8, length=2.5, width=0.7)
    cb.outline.set_linewidth(0.7)

    if scalebar_px:
        add_lambda_scalebar(ax, scalebar_px, label=scalebar_label)

    fig.tight_layout()
    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
        fig.savefig(str(savepath).replace(".png", ".pdf"), bbox_inches="tight")
        print(f"> wrote {savepath} (and .pdf)")
    return fig, ax


def _bar_px(args, lam):
    """Bar length in pixels: millimetres if asked for, else one Lambda."""
    if getattr(args, "scalebar_mm", None):
        from src.models.spatial_utils import MACAQUE_LAMBDA_MM
        return args.scalebar_mm * lam / MACAQUE_LAMBDA_MM
    return lam if args.scalebar else None


def _bar_label(args, lam):
    if getattr(args, "scalebar_mm", None):
        from src.models.spatial_utils import MACAQUE_LAMBDA_MM
        mm = args.scalebar_mm
        return (f"{mm:g} mm  ($\\Lambda$={lam:.1f} px $\\equiv$ "
                f"{MACAQUE_LAMBDA_MM:g} mm)")
    return r"$\Lambda$"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stats", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--layers", type=str, default="0-3",
                    help="vision-encoder layers to keep, e.g. 'all', '0-3', '0,3,5'")
    ap.add_argument("--crop", action="store_true", default=True,
                    help="crop the map to the selected layers (default)")
    ap.add_argument("--no-crop", dest="crop", action="store_false",
                    help="keep the full 165-row band, gate units by slot only")
    ap.add_argument("--smooth-window-frac", type=float, default=0.15)
    ap.add_argument("--column-spacing", type=float, default=None,
                    help="override Lambda (px). Default: estimated on the FULL band, so "
                         "the smoothing matches the full-band figure")
    ap.add_argument("--winding-thresh", type=float, default=0.3)
    ap.add_argument("--sd-thresh-deg", type=float, default=30.0)
    ap.add_argument("--open-loop", action="store_true")
    ap.add_argument("--downsample", action="store_true",
                    help="sample smoothing windows at stride 0.3*width and drop partial\n                         windows, as orientation_metrics.py does. Default keeps every\n                         sheet row (stride 1, centred windows), so a 4-layer strip stays\n                         20 px tall instead of collapsing to 8")
    ap.add_argument("--style", choices=["heatmap", "scatter", "both"],
                    default="heatmap",
                    help="heatmap: imshow of the smoothed field (default). scatter: one\n                         marker per unit at its sheet position, TDANN-style. both: write\n                         each to <out> with a _heatmap/_scatter suffix")
    ap.add_argument("--raw", action="store_true",
                    help="scatter the UNsmoothed per-unit preferred orientation instead\n                         of the smoothed field (pinwheel markers still come from the\n                         smoothed map). Only affects --style scatter")
    ap.add_argument("--marker", type=str, default="o",
                    help="scatter marker; 'o' for TDANN-style circles, 's' to tile exactly")
    ap.add_argument("--cbar", choices=["auto", "horizontal", "vertical"],
                    default="auto", help="colour-bar orientation")
    ap.add_argument("--marker-scale", type=float, default=1.9,
                    help="marker width in sheet px; >1 makes the circles overlap")
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="display-only uniform jitter (sheet px) to break the grid lattice")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--edge", type=float, default=0.0,
                    help="marker edge linewidth (pt); 0 for none")
    ap.add_argument("--all-units", action="store_true",
                    help="scatter every unit in the selected layers, not just the ones that "
                         "passed the tuning gate. Denser, TDANN-like, but includes units "
                         "whose preferred orientation is not significant")
    ap.add_argument("--zoom", type=str, default=None,
                    help="display window inside the selected region: "
                         "'r0,r1,c0,c1' in band px, or 'sq80' for a centred "
                         "80x80 square. Smoothing and pinwheel detection still "
                         "run on the full region, so edge windows keep their "
                         "real neighbours -- only the display is cropped")
    ap.add_argument("--cmap", choices=sorted(CMAPS), default="muted",
                    help="'muted' is a desaturated cyclic map matching TDANN's "
                         "panels; 'hsv' is the fully saturated map the rest of "
                         "the repo's orientation figures use")
    ap.add_argument("--cbar-label", type=str, default="Orientation (deg)")
    ap.add_argument("--no-pinwheels", dest="pinwheels", action="store_false",
                    default=True,
                    help="draw the map only, with no pinwheel markers or count "
                         "legend. Use this on an UNSMOOTHED map (stats.fwhm_mm=0): "
                         "count_pinwheels requires locally homogeneous windows "
                         "(sd < sd_thresh_deg), and without smoothing the median "
                         "circular sd is ~46 deg, so only ~5% of windows qualify "
                         "and the counts are a threshold artifact, not a measure")
    ap.add_argument("--scalebar-mm", type=float, default=None,
                    help="draw a black bar this many MILLIMETRES long instead of one "
                         "Lambda. The conversion is mm_per_px = MACAQUE_LAMBDA_MM / "
                         "Lambda_measured, i.e. it ASSUMES this sheet's column spacing "
                         "corresponds to macaque V1's 0.75 mm. That assumption is the "
                         "whole calibration -- it is not an independent measurement of "
                         "cortical distance, and it is only as good as the Lambda "
                         "estimate, which is quantised to the radial bin grid "
                         "(~1 bin = ~12%% at Lambda~12 px). The label carries Lambda so "
                         "the provenance travels with the figure.")
    ap.add_argument("--scalebar", action="store_true",
                    help="draw a bar one column spacing (Lambda) long")
    ap.add_argument("--title", type=str, default=None)
    ap.add_argument("--no-markers", action="store_true",
                    help="draw the orientation map alone: no pinwheel dots, no +/- count "
                         "legend, and no count in the title. The count is not evidence on "
                         "this sheet -- at matched Lambda the UNTRAINED step-0 model "
                         "scores 298 against the trained model's 254 -- so a figure that "
                         "annotates it invites a conclusion the number does not support.")
    ap.add_argument("--no-counts", action="store_true",
                    help="keep the pinwheel dots but drop every COUNT: none in the title, "
                         "no +/- tally legend. Distinct from --no-markers, which also "
                         "removes the dots. Use this when the singularities are worth "
                         "showing but the number is not -- see --no-markers on why the "
                         "count is not evidence on this sheet.")
    ap.add_argument("--figsize", type=float, nargs=2, default=None)
    args = ap.parse_args()

    d = np.load(args.stats, allow_pickle=True)
    pref_ori = d["pref_ori"]
    gate = d["gate"] if "gate" in d.files else d["tuned"]
    print(f"> Loaded {args.stats} ({pref_ori.size} units)")

    ori_band = band_grid(pref_ori)                       # (165, 256)
    valid = (band_grid(gate.astype(float)) > 0.5) & np.isfinite(ori_band)

    sel = parse_layers(args.layers)
    lay_mask = layer_band_mask(sel)
    valid = valid & lay_mask

    # Lambda from the full band (a 20-row strip cannot support its own 2D estimate)
    lam = args.column_spacing
    if lam is None:
        lam, _, _ = column_spacing(ori_band,
                                   (band_grid(gate.astype(float)) > 0.5)
                                   & np.isfinite(ori_band))
    print(f"> Lambda = {lam:.2f} sheet px (full band)")

    if args.crop and len(sel) < N_VISION_LAYERS:
        rows = np.nonzero(lay_mask.any(axis=1))[0]
        r0, r1 = rows.min(), rows.max() + 1
        ori_band, valid = ori_band[r0:r1], valid[r0:r1]
        print(f"> Cropped to band rows {r0}-{r1 - 1} "
              f"({r1 - r0} rows = {(r1 - r0) / lam:.2f} Lambda tall, "
              f"{ori_band.shape[1] / lam:.1f} Lambda wide)")
        if (r1 - r0) / lam < 1.0:
            print("  CAVEAT: under one column spacing tall -- pinwheel *density* from this "
                  "strip is descriptive only; report the full-band value instead.")

    width = max(3, int(round(args.smooth_window_frac * lam)))
    if args.downsample:
        stride = max(1, int(round(0.3 * width)))
        ori_sm, sd_sm, n_sm = smoothed_orientation_map(ori_band, valid, width, stride)
    else:
        stride = 1
        ori_sm, sd_sm, n_sm = smoothed_orientation_map_same(ori_band, valid, width)
    print(f"> Smoothed map {ori_sm.shape} (window {width} px, stride {stride} px), "
          f"{np.median(n_sm[n_sm > 0]):.0f} units/window (median)")

    wn = winding_numbers(ori_sm, closed_loop=not args.open_loop)
    counts, centres = count_pinwheels(wn, sd_sm, thresh=args.winding_thresh,
                                      sd_thresh_deg=args.sd_thresh_deg)
    total = sum(counts)
    dens = pinwheel_density(total, ori_sm.shape, stride, lam)
    print(f"> Pinwheels: {counts[0]} positive + {counts[1]} negative = {total}  "
          f"(density {dens:.2f} per Lambda^2)")

    layer_tag = ("all layers" if len(sel) == N_VISION_LAYERS
                 else f"vision layers {sel.min()}–{sel.max()}")
    if not args.pinwheels:
        centres = [np.zeros((0, 2)), np.zeros((0, 2))]
        total = 0

    disp_ori, disp_valid, disp_centres = ori_sm, valid, centres
    disp_raw = ori_band
    if args.zoom:
        box = parse_box(args.zoom, ori_sm.shape)
        disp_ori, disp_valid, disp_centres = apply_zoom(ori_sm, valid, centres, box)
        disp_raw, _, _ = apply_zoom(ori_band, valid, centres, box)
        shown_n = sum(c.shape[0] for c in disp_centres)
        print(f"> Zoom to band rows {box[0]}-{box[1] - 1}, cols {box[2]}-{box[3] - 1} "
              f"({(box[1] - box[0]) / lam:.2f} x {(box[3] - box[2]) / lam:.2f} Lambda): "
              f"{shown_n} of {total} pinwheels in view")
        total = shown_n

    # `or` would swallow an explicit empty title, and --no-markers must never leave a
    # pinwheel count in the title after the dots have been removed.
    if args.title is not None:
        title = args.title
    elif args.pinwheels and not args.no_markers and not args.no_counts:
        title = f"{layer_tag} \u2014 {total} pinwheels"
    else:
        title = f"{layer_tag}"
    figsize = tuple(args.figsize) if args.figsize else None
    cm = CMAPS[args.cmap]()
    out = Path(args.out)

    def _p(suffix):
        return out if args.style != "both" else out.with_name(out.stem + suffix + out.suffix)

    if args.style in ("heatmap", "both"):
        plot_panel(disp_ori, [] if args.no_markers else disp_centres, _p("_heatmap"),
                   lam_px=lam, stride=stride, show_counts=not (args.no_markers or args.no_counts),
                   title=title, figsize=figsize,
                   cbar_orientation=None if args.cbar == "auto" else args.cbar,
                   scalebar_px=_bar_px(args, lam),
                   scalebar_label=_bar_label(args, lam),
                   cbar_label=args.cbar_label, cmap=cm)
    if args.style in ("scatter", "both"):
        if stride != 1:
            raise SystemExit("--style scatter needs the same-size smoother; drop --downsample")
        field = disp_raw if args.raw else disp_ori
        shown = np.ones_like(disp_valid) if args.all_units else disp_valid
        plot_panel_scatter(field, shown, [] if args.no_markers else disp_centres,
                           _p("_scatter"), show_counts=not (args.no_markers or args.no_counts), title=title,
                           figsize=figsize, marker=args.marker,
                           marker_scale=args.marker_scale, jitter=args.jitter,
                           alpha=args.alpha, edge=args.edge,
                           cbar_orientation=None if args.cbar == "auto" else args.cbar,
                           scalebar_px=_bar_px(args, lam),
                           scalebar_label=_bar_label(args, lam),
                           cbar_label=args.cbar_label, cmap=cm)


if __name__ == "__main__":
    main()
