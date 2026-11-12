"""Pinwheel maps for a TILED (continuous) sheet, one panel per vision-encoder slot.

Why not `plot_pinwheels_panel`: that script treats the vision band as one continuous map,
which is right for the strip layout (a layer is a full-width band of rows). Under tiling a
layer is a 64x64 TILE, and adjacent tiles are different layers -- a contour walked across a
tile boundary crosses a depth discontinuity, so any winding number it picks up is an
artefact of the layout, not a pinwheel. Here each tile is smoothed and analysed alone.

ALWAYS PLOTS THE SHUFFLE CONTROL. Pinwheel detection is a local winding-number test; it
returns a non-zero count on ANY sufficiently smoothed field, including pure noise, because
smoothing a random axial field manufactures local continuity. The only interpretable
quantity is real-vs-shuffled on the identical pipeline. Read a count as evidence of
structure only if it clears the control.

SMOOTHING HERE IS NOT THE CONFIG'S fwhm_mm. `--width` is a box window in pixels applied
to the ORIENTATION field at analysis time, only so a winding contour has a continuous
field to walk. `fwhm_mm` is a Gaussian applied to the ACTIVATIONS at eval time, before
preference exists. Outputs are tagged `boxw<N>` to keep the two apart.

    python -m src.experiments.orientation.plot_tiled --run continuous-4096-ckpt138-nosmooth
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.models.spatial_utils import DEFAULT_SHEET_LAYOUT as S
from src.experiments.orientation.metrics import (count_pinwheels, pinwheel_density,
                                                   winding_numbers)
from src.experiments.orientation.plot_panel import smoothed_orientation_map_same


# One extractor for both layouts, so strip-4096 (16 x 256 per slot) and continuous
# (64 x 64 per slot) go through an identical analysis.
from src.experiments.orientation.nn_delta import slot_tile, slot_window  # noqa: E402


def analyse(theta, valid, width, thresh, sd_thresh, lam):
    ori, sd, _ = smoothed_orientation_map_same(theta, valid, width)
    wn = winding_numbers(ori)
    counts, centres = count_pinwheels(wn, sd, thresh=thresh, sd_thresh_deg=sd_thresh)
    n = int(sum(counts))
    dens = pinwheel_density(n, ori.shape, stride=1, lam_px=lam)
    return ori, centres, n, dens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--width", type=int, default=5,
                    help="side of the BOX window, in PIXELS, used to circular-mean the "
                         "preferred-orientation field before winding numbers are walked. "
                         "NOT the config's fwhm_mm: that is a GAUSSIAN applied to the "
                         "ACTIVATIONS at eval time (sigma_px = fwhm_mm / 2.355), before "
                         "preference is computed, and every run here sets it to 0 (the "
                         "'-nosmooth' in the run name). Output files are tagged 'boxw<N>' "
                         "rather than 'w<N>' so they cannot be read as the repo's "
                         "'fwhm<N>' convention -- which is a live trap, since a width-5 "
                         "box is roughly a fwhm 3.4 px Gaussian. The two compose: run "
                         "this on a '-fwhm3' results dir and you get both.")
    ap.add_argument("--thresh", type=float, default=0.3, help="|winding| threshold")
    ap.add_argument("--sd-thresh", type=float, default=30.0,
                    help="max circular sd (deg) for a window to count as homogeneous")
    ap.add_argument("--lam", type=float, default=10.0,
                    help="Lambda in px used for the DENSITY figure only. NOMINAL: the "
                         "measured Lambda on this sheet is not separable from a shuffle "
                         "control, so treat every per-Lambda^2 number as provisional.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window", type=int, default=1,
                    help="consecutive slots per tile, sliding by 1. A strip-4096 slot is "
                         "1.6 Lambda on its short axis, too thin to hold a pinwheel; "
                         "--window 2 makes it 3.2. See nn_delta.slot_window.")
    ap.add_argument("--slots", default="all",
                    help="'all', 'lo-hi', or 'pXX-100' as a percentage of encoder depth "
                         "with 100 the DEEPEST slot. Slot 0 is the conv front-end and it "
                         "dominates the raw count on a sparse sheet -- at ckpt-312 it held "
                         "17 of 22 pinwheels off 388 valid units, against 20 in its own "
                         "shuffle -- so restricting is often the honest thing to do.")
    ap.add_argument("--no-counts", action="store_true",
                    help="drop the per-slot pinwheel count from the titles; "
                         "the maps and the dots are unchanged.")
    args = ap.parse_args()

    probe = slot_window(np.zeros(S.num_units), S, 0, args.window).shape
    print(f"layout {S.mode} {S.n_row}x{S.n_col}; a vision slot is {probe[0]}x{probe[1]} px")
    if min(probe) < 4 * args.lam:
        print(f"  ! a slot spans only {min(probe) / args.lam:.1f} Lambda on its short axis "
              f"(at the nominal Lambda={args.lam:g} px). A pinwheel needs 2-D room: below "
              f"~4 Lambda the detector is mostly reading the boundary, and the count is "
              f"not comparable with a layout whose slots are square. This is the EXTENT "
              f"problem the continuous layout exists to fix.")

    d = np.load(f"results/{args.run}/orientation_unbiased/orientation_stats.npz")
    pref = d["pref_ori"]
    gate = d["gate"] & d["vision_band"]
    outdir = Path(f"results/{args.run}/orientation_unbiased/pinwheels_tiled")
    outdir.mkdir(parents=True, exist_ok=True)
    # The slot range goes in the FILENAME. Without it a --slots run silently overwrites
    # the all-slots figures with a different tile count under the same name, which is the
    # same class of mistake as tagging a box width as though it were an fwhm.
    stag = "" if args.slots == "all" else f"_{args.slots.replace('-', 'to')}"
    stag += "" if args.window == 1 else f"_win{args.window}"

    rng = np.random.default_rng(args.seed)
    n_enc = S.num_encoder_layers
    if args.slots == "all":
        keep = list(range(n_enc))
    elif args.slots.startswith("p"):
        lo_pct, hi_pct = (float(x) for x in args.slots[1:].split("-"))
        keep = list(range(int(round(lo_pct / 100.0 * (n_enc - 1))),
                          int(round(hi_pct / 100.0 * (n_enc - 1))) + 1))
    else:
        lo, hi = (int(x) for x in args.slots.split("-"))
        keep = list(range(lo, hi + 1))
    print(f"slots: {args.slots} -> {keep[0]}..{keep[-1]} ({len(keep)} of {n_enc})")
    rows, real_maps, real_ctrs, sh_maps, sh_ctrs = [], [], [], [], []
    for s in keep:
        theta = slot_window(pref, S, s, args.window)
        if theta is None:
            continue
        valid = (slot_window(gate.astype(float), S, s, args.window) > 0.5) & np.isfinite(theta)
        if valid.sum() < 200:
            rows.append((s, int(valid.sum()), np.nan, np.nan, np.nan))
            real_maps.append(None); real_ctrs.append(None)
            sh_maps.append(None); sh_ctrs.append(None); continue
        ori, ctr, n, dens = analyse(theta, valid, args.width, args.thresh,
                                    args.sd_thresh, args.lam)
        sh = theta.copy(); sh[valid] = rng.permutation(theta[valid])
        ori_sh, ctr_sh, n_sh, _ = analyse(sh, valid, args.width, args.thresh,
                                          args.sd_thresh, args.lam)
        rows.append((s, int(valid.sum()), n, dens, n_sh))
        real_maps.append(ori); real_ctrs.append(ctr)
        sh_maps.append(ori_sh); sh_ctrs.append(ctr_sh)

    print(f"{'slot':>4} {'n_valid':>8} {'pinwheels':>10} {'per_Lam^2':>10} {'SHUFFLED':>9}")
    for s, nv, n, dens, n_sh in rows:
        print(f"{s:>4} {nv:>8} {n:>10.0f} {dens:>10.3f} {n_sh:>9.0f}")
    ok = [(n, n_sh) for _, _, n, _, n_sh in rows if np.isfinite(n)]
    tot, tot_sh = sum(a for a, _ in ok), sum(b for _, b in ok)
    print(f"\nTOTAL over {len(ok)} tiles: real {tot:.0f}  shuffled {tot_sh:.0f}  "
          f"ratio {tot / max(tot_sh, 1e-9):.2f}")
    print(f"mean density {np.nanmean([r[3] for r in rows]):.3f} pinwheels/Lambda^2 "
          f"at NOMINAL Lambda={args.lam} px (macaque V1 ~ 3.1)")

    def draw_tile(ax, omap, ctr):
        ax.imshow(omap, cmap="hsv", vmin=0, vmax=180, interpolation="nearest")
        pos = ctr[0]
        neg = ctr[1] if len(ctr) > 1 else np.empty((0, 2))
        for c, col, ec in ((pos, "k", "none"), (neg, "w", "k")):
            if c.size:
                ax.scatter(c[:, 1], c[:, 0], s=7, c=col, edgecolors=ec, linewidths=0.3)

    def grid_fig(maps, ctrs, which, path):
        """One panel per slot. IDENTICAL layout, size and colour scale for the real and
        shuffled sets, so the two files can be flipped between directly."""
        ncol = 6
        nrow = int(np.ceil(len(rows) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(2.05 * ncol, 2.25 * nrow))
        for ax in axes.ravel():
            ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)
        for k, (sl, nv, n, dens, n_sh) in enumerate(rows):
            ax = axes.ravel()[k]
            if maps[k] is None:
                ax.text(.5, .5, f"slot {sl}\n(too few units)", ha="center",
                        va="center", fontsize=7, color="0.5"); continue
            draw_tile(ax, maps[k], ctrs[k])
            shown = n if which == "real" else n_sh
            ax.set_title(f"slot {sl}" if args.no_counts
                         else f"slot {sl}   {shown:.0f} pw", fontsize=7)
        tag = ("orientation preference per vision slot" if which == "real" else
               "SPATIALLY SHUFFLED control — same units, positions permuted within tile")
        fig.suptitle(f"{args.run} — {tag} (64x64 tile), smoothing {args.width}px\n"
                     f"black +1/2 winding, white -1/2   |   total "
                     f"{(tot if which == 'real' else tot_sh):.0f} pinwheels",
                     fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.965])
        fig.savefig(path, dpi=170, bbox_inches="tight"); plt.close(fig)
        return path

    p1 = grid_fig(real_maps, real_ctrs, "real",
                  outdir / f"pinwheel_maps_boxw{args.width}{stag}.png")
    p1b = grid_fig(sh_maps, sh_ctrs, "shuffled",
                   outdir / f"pinwheel_maps_SHUFFLED_boxw{args.width}{stag}.png")

    # ---- paired close-up: the visual claim, at a readable size ----
    pick = [k for k in range(len(rows)) if real_maps[k] is not None]
    pick = pick[:: max(1, len(pick) // 4)][:4]
    fig, axes = plt.subplots(2, len(pick), figsize=(3.0 * len(pick), 6.4))
    for col, k in enumerate(pick):
        for row, (maps, ctrs, lab) in enumerate(
                ((real_maps, real_ctrs, "real"), (sh_maps, sh_ctrs, "shuffled"))):
            ax = axes[row, col]
            draw_tile(ax, maps[k], ctrs[k])
            ax.set_xticks([]); ax.set_yticks([])
            npw = rows[k][2] if lab == "real" else rows[k][4]
            ax.set_title(f"slot {rows[k][0]} — {lab}" if args.no_counts
                         else f"slot {rows[k][0]} — {lab}   {npw:.0f} pw",
                         fontsize=9)
    fig.suptitle(f"{args.run}: real vs spatially shuffled, same tiles and same detector\n"
                 f"If the top row is structure, the bottom row must look different.",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p3 = outdir / f"pinwheel_paired_boxw{args.width}{stag}.png"
    fig.savefig(p3, dpi=170, bbox_inches="tight"); plt.close(fig)

    # ---- figure 2: the honest readout ----
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    sl = [r[0] for r in rows if np.isfinite(r[2])]
    ax.plot(sl, [r[2] for r in rows if np.isfinite(r[2])], "o-", ms=4, lw=1.4,
            color="#c0392b", label="real map")
    ax.plot(sl, [r[4] for r in rows if np.isfinite(r[2])], "s--", ms=4, lw=1.4,
            color="0.45", label="spatially shuffled (control)")
    ax.set_xlabel("vision encoder slot (0 = conv front-end)")
    ax.set_ylabel("pinwheels detected in tile")
    ax.set_title(f"{args.run}: pinwheel count vs shuffle control "
                 f"(total {tot:.0f} vs {tot_sh:.0f})", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p2 = outdir / f"pinwheel_vs_shuffle_boxw{args.width}{stag}.png"
    fig.savefig(p2, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"\nwrote {p1}\nwrote {p1b}\nwrote {p2}\nwrote {p3}")


if __name__ == "__main__":
    main()
