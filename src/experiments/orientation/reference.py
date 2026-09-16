"""
Compare the Topo-Omni orientation map against the reference data TDANN uses for its V1
figure (Margalit et al., Fig. 2).

The reference is MACAQUE V1, not human. TDANN's Fig. 2 compares against
``NauhausOrientationTissue`` (macaque optical-imaging maps) and Ringach et al. 2002
(single-unit tuning from 308 macaque V1 neurons). There is no human counterpart for these
particular metrics: pinwheel-resolution orientation maps require optical imaging or dense
electrode arrays, and human V1 orientation columns have never been mapped at that scale
(7T fMRI recovers coarse orientation biases, not pinwheels). So "how human-like is the map"
is not a question this metric family can answer -- it is macaque-like or not.

What is quantitatively comparable, and what is not
--------------------------------------------------
circular variance
    Directly comparable. ``data/ringach_2002_db.mat`` ships with TDANN and holds the real
    per-neuron circular variances (field ``orivar``), so this module runs an actual
    two-sample KS test rather than reading a number off a figure.
pinwheel density, map smoothness
    TDANN computes these from macaque map data that is *not* redistributed in their repo, so
    the reference values here are read off the published figure and are approximate. They
    are labelled as such in the output; do not quote them as published values.

A measurement caveat that matters for CV
----------------------------------------
Ringach's CVs come from single neurons, and TDANN's from single model units. The tuning
curve saved by ``run_orientation`` is computed on the *smoothed* sheet, which is a
neighbourhood average and not the analogue of a single unit. This module therefore reports
CV from the unsmoothed responses as the like-for-like comparison, with the smoothed value
alongside.

Usage
-----
python -m src.experiments.orientation.reference
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from scipy.stats import ks_2samp

from src.experiments.orientation.metrics import (
    band_grid, circular_variance, layer_band_mask, parse_layers,
    N_VISION_LAYERS, ROWS_PER_VISION_LAYER,
)
from src.experiments.orientation.run import orientation_tuning_curve

from dotenv import load_dotenv
load_dotenv()

# TDANN's thresholds (notebooks/F02_V1_Details): a unit counts as orientation selective
# when its circular variance falls below CV_THRESH.
CV_THRESH = 0.6

# Approximate values read off the published Fig. 2 (panels E and K). TDANN does not
# redistribute the macaque map data these come from, so they are figure-read, not exact.
FIGURE_REFERENCE = {
    "pinwheel_density": {
        "Macaque V1": 3.0, "TDANN": 2.1, "Hand-Crafted SOM": 4.0,
        "DNN-SOM": 0.4, "Task Only": 0.6, "Unoptimized": 0.1,
    },
    "map_smoothness": {
        "Macaque V1": 0.65, "TDANN": 0.72,
    },
}


def load_macaque_cv(path="data/ringach_2002_db.mat") -> np.ndarray:
    """Per-neuron circular variance for 308 macaque V1 neurons (Ringach et al. 2002)."""
    db = loadmat(str(path))["db"]
    cv = np.squeeze(db["orivar"][0, 0]).astype(float)
    return cv[np.isfinite(cv)]


def cv_bin(cv: np.ndarray, n_bins: int = 40) -> np.ndarray:
    """TDANN's ``cv_bin``: percentage histogram over 40 equal bins on [0, 1]."""
    counts, _ = np.histogram(cv, bins=np.linspace(0.0, 1.0, n_bins))
    return counts / max(np.sum(counts), 1) * 100.0


def compare_cv(model_cv: np.ndarray, macaque_cv: np.ndarray, label: str) -> dict:
    """KS comparison of two CV distributions, plus the summary stats TDANN reports."""
    model_cv = model_cv[np.isfinite(model_cv)]
    ks_raw, p_raw = ks_2samp(model_cv, macaque_cv)
    # TDANN runs ks_2samp on the *binned* histograms rather than the raw samples, which
    # compares bin-height distributions instead of the underlying CVs. Reported for
    # comparability with their panel J, but `ks_raw` is the statistically meaningful one.
    ks_binned, _ = ks_2samp(cv_bin(model_cv), cv_bin(macaque_cv))
    return {
        "label": label,
        "n": model_cv.size,
        "median": float(np.median(model_cv)),
        "pct_selective": float(100.0 * np.mean(model_cv < CV_THRESH)),
        "ks": float(ks_raw),
        "p": float(p_raw),
        "similarity": float(1.0 - ks_raw),
        "ks_binned_tdann": float(ks_binned),
    }


def plot_cv_comparison(variants, macaque_cv, savepath=None, figsize=(9.5, 3.8)):
    """Histogram + CDF overlay: the CDF panel is where the KS statistic is legible."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    bins = np.linspace(0, 1, 40)
    # One colour per variant, cycled if there are ever more: zipping against a short list
    # silently drops the extra curves.
    palette = ["#5B2C9E", "#8E7CC3", "#2E8B8B", "#C25E00"]
    colors = [palette[i % len(palette)] for i in range(len(variants))]

    ax1.hist(macaque_cv, bins=bins, density=True, color="0.55",
             alpha=0.75, label=f"Macaque V1 (n={macaque_cv.size})")
    for (name, cv), c in zip(variants, colors):
        ax1.hist(cv[np.isfinite(cv)], bins=bins, density=True, histtype="step",
                 lw=1.9, color=c, label=f"{name} (n={np.isfinite(cv).sum()})")
    ax1.set_xlabel("circular variance")
    ax1.set_ylabel("density")
    ax1.set_title("Orientation tuning strength")
    ax1.legend(frameon=False, fontsize=7.5)
    sns.despine(ax=ax1)

    def cdf(x):
        x = np.sort(x[np.isfinite(x)])
        return x, np.arange(1, x.size + 1) / x.size

    xm, ym = cdf(macaque_cv)
    ax2.plot(xm, ym, color="0.35", lw=2.0, label="Macaque V1")
    for (name, cv), c in zip(variants, colors):
        xv, yv = cdf(cv)
        ax2.plot(xv, yv, color=c, lw=1.9, label=name)
    ax2.axvline(CV_THRESH, color="crimson", ls=":", lw=1.2,
                label=f"selective cut ({CV_THRESH})")
    ax2.set_xlabel("circular variance")
    ax2.set_ylabel("cumulative fraction")
    ax2.set_title("CDF (vertical gap = KS)")
    ax2.legend(frameon=False, fontsize=7.5, loc="lower right")
    sns.despine(ax=ax2)

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig, (ax1, ax2)


def plot_cv_by_depth(profiles, macaque_cv, savepath=None, figsize=(9.5, 3.7)):
    """% orientation-selective and KS-to-macaque as a function of encoder depth."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    palette = ["#5B2C9E", "#2E8B8B"]
    mac_pct = 100.0 * np.mean(macaque_cv < CV_THRESH)

    for (name, prof), c in zip(profiles.items(), palette):
        if not prof:
            continue
        x = [0.5 * (r["lo"] + r["hi"]) for r in prof]
        short = name.split("(")[0].strip()
        ax1.plot(x, [r["pct"] for r in prof], "o-", color=c, lw=1.8, ms=4, label=short)
        ax2.plot(x, [r["ks"] for r in prof], "o-", color=c, lw=1.8, ms=4, label=short)

    ax1.axhline(mac_pct, color="#1B7F3B", ls="--", lw=1.4,
                label=f"Macaque V1 ({mac_pct:.0f}%)")
    ax1.set_xlabel("vision-encoder layer (0 = earliest)")
    ax1.set_ylabel(f"% units with CV < {CV_THRESH}")
    ax1.set_title("Orientation selectivity by depth")
    ax1.legend(frameon=False, fontsize=7.5)
    sns.despine(ax=ax1)

    ax2.set_xlabel("vision-encoder layer (0 = earliest)")
    ax2.set_ylabel("KS vs macaque (lower = closer)")
    ax2.set_title("Distance from the macaque CV distribution")
    ax2.legend(frameon=False, fontsize=7.5)
    sns.despine(ax=ax2)

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig, (ax1, ax2)


def plot_summary_table(mac, ours, label, extra_notes=None, savepath=None,
                       figsize=(8.6, 4.4)):
    """Two-column comparison table as a figure: macaque V1 against one model condition.

    ``mac`` and ``ours`` are dicts of {row label: formatted string}; rows are drawn in the
    order given. ``extra_notes`` are footnote lines, used to flag any row that could not be
    computed on the restricted unit set.
    """
    rows = list(mac.keys())
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    n = len(rows)
    y0, dy = 0.90, 0.072
    xm, xo = 0.52, 0.80
    mac_c, our_c = "#1B7F3B", "#5B2C9E"

    ax.text(0.02, y0 + 0.075, "metric", fontsize=10, weight="bold")
    ax.text(xm, y0 + 0.075, "Macaque V1", fontsize=10, weight="bold",
            color=mac_c, ha="center")
    ax.text(xo, y0 + 0.075, label, fontsize=10, weight="bold", color=our_c, ha="center")
    ax.plot([0.0, 1.0], [y0 + 0.045, y0 + 0.045], color="0.2", lw=1.2)

    for i, key in enumerate(rows):
        y = y0 - i * dy
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((0.0, y - 0.022), 1.0, dy * 0.92,
                                       color="0.95", zorder=0))
        ax.text(0.02, y, key, fontsize=9.5, va="center", zorder=2)
        ax.text(xm, y, mac.get(key, "—"), fontsize=9.5, va="center", ha="center",
                color=mac_c, zorder=2)
        ax.text(xo, y, ours.get(key, "—"), fontsize=9.5, va="center", ha="center",
                color=our_c, zorder=2)

    ax.plot([0.0, 1.0], [y0 - (n - 1) * dy - 0.030, y0 - (n - 1) * dy - 0.030],
            color="0.2", lw=1.0)
    if extra_notes:
        for j, note in enumerate(extra_notes):
            ax.text(0.0, y0 - (n - 1) * dy - 0.075 - j * 0.048, note,
                    fontsize=7.6, color="0.35", va="top")

    ax.set_xlim(0, 1)
    ax.set_ylim(y0 - (n - 1) * dy - 0.075 - 0.05 * (len(extra_notes or []) + 1), 1.02)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig, ax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", type=str,
                    default="results/topo-omni/orientation/orientation_stats.npz")
    ap.add_argument("--metrics", type=str,
                    default="results/topo-omni/orientation/metrics/orientation_metrics.npz")
    ap.add_argument("--sheets", type=str,
                    default="results/topo-omni/orientation/cortical_sheets.npy")
    ap.add_argument("--ringach", type=str, default="data/ringach_2002_db.mat")
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--layers", type=str, default="all",
                    help="which vision-encoder layers to compare (0 = earliest, 31 = "
                         "deepest): 'all', a range like '0-7', or a list like '0,1,2'. "
                         "Macaque V1 is an early visual area, so an early-layer subset is "
                         "the more defensible comparison than the whole band.")
    args = ap.parse_args()
    args.metrics_explicit = any(a.startswith("--metrics") for a in sys.argv[1:])

    stats_path = Path(args.stats)
    outdir = Path(args.outdir) if args.outdir else stats_path.parent / "metrics"
    outdir.mkdir(parents=True, exist_ok=True)

    # Follow --outdir unless --metrics was given explicitly. Otherwise a layer-restricted
    # run writes its metrics to --outdir but reads the *full-band* metrics back here, so the
    # map-level section would silently describe a different unit set from section 1.
    if not args.metrics_explicit:
        cand = outdir / "orientation_metrics.npz"
        if cand.exists():
            args.metrics = str(cand)

    d = np.load(stats_path, allow_pickle=True)
    counts, ori_deg = d["counts"], d["ori_deg"]
    gate = d["gate"] if "gate" in d.files else d["tuned"]
    band_gate_all = band_grid(gate.astype(float)) > 0.5

    sel_layers = parse_layers(args.layers)
    lay_mask = layer_band_mask(sel_layers)
    band_gate = band_gate_all & lay_mask
    layer_tag = ("all layers" if len(sel_layers) == N_VISION_LAYERS
                 else f"layers {sel_layers.min()}-{sel_layers.max()}")
    if len(sel_layers) < N_VISION_LAYERS:
        print(f"Restricted to vision-encoder layers {sel_layers.min()}-{sel_layers.max()} "
              f"({len(sel_layers)}/{N_VISION_LAYERS} layers, "
              f"{len(sel_layers) * ROWS_PER_VISION_LAYER} sheet rows): "
              f"{band_gate.sum()} of {band_gate_all.sum()} gated units\n")

    print("=" * 78)
    print("REFERENCE IS MACAQUE V1, NOT HUMAN")
    print("=" * 78)
    print("  TDANN Fig. 2 compares against macaque optical-imaging maps (Nauhaus) and")
    print("  macaque single units (Ringach 2002). Human V1 orientation columns have never")
    print("  been mapped at pinwheel resolution, so there is no human reference for these")
    print("  metrics -- the question is whether the map is macaque-like.")
    print("  (The repo's human fMRI data under fMRI/ is category selectivity, not")
    print("   orientation, so it cannot substitute here.)")

    macaque_cv = load_macaque_cv(args.ringach)

    # ---- CV: smoothed (what run_orientation saved) and unsmoothed (like-for-like).
    # Kept as 2D band grids so the per-layer depth profile below can slice by row.
    cv_sm_2d = band_grid(circular_variance(d["tuning_curve"], ori_deg))
    cv_smoothed = cv_sm_2d[band_gate]
    cv_unsm_2d = ptp_2d = None

    variants = []
    flat_frac = None
    if Path(args.sheets).exists():
        raw = np.load(args.sheets)
        curve_raw, _ = orientation_tuning_curve(raw, counts)
        cv_unsm_2d = band_grid(circular_variance(curve_raw, ori_deg))
        cv_unsmoothed = cv_unsm_2d[band_gate]
        # Half the units are *completely* unmodulated by these stimuli before smoothing
        # (identical response to every grating), which pins their CV at exactly 1. Quoting a
        # median over that mixture says more about how many units are dead than about how
        # sharply the live ones are tuned, so report the modulated subset separately.
        ptp_2d = band_grid(curve_raw.max(axis=0) - curve_raw.min(axis=0))
        ptp = ptp_2d[band_gate]
        flat_frac = float(np.mean(ptp <= 0))
        variants.append(("Topo-Omni unsmoothed, modulated", cv_unsmoothed[ptp > 0]))
        # variants.append(("Topo-Omni unsmoothed, all", cv_unsmoothed))
        # Same live-unit selection applied to the smoothed CV. This completes a 2x2 --
        # {unsmoothed, smoothed} x {all units, live units} -- so the gap between "smoothed,
        # all" and "unsmoothed, modulated" can be attributed to the smoothing or to the
        # selection rather than confounding the two. `ptp` comes from the *raw* responses on
        # purpose: selecting on the smoothed curve would be circular, since smoothing leaves
        # almost nothing exactly flat.
        # Must stay inside this block: `ptp` does not exist without --sheets.
        variants.append(("Topo-Omni smoothed, modulated", cv_smoothed[ptp > 0]))
    else:
        print(f"\n  ! {args.sheets} missing; unsmoothed CV and live-unit selection skipped")
    # variants.append(("Topo-Omni smoothed sheet", cv_smoothed))


    print("\n" + "=" * 78)
    print("1. CIRCULAR VARIANCE  — real quantitative comparison (Ringach 2002, n=308)")
    print("=" * 78)
    print(f"  {'':32s} {'n':>7} {'median':>7} {'%CV<0.6':>8} {'KS':>6} {'1-KS':>6}")
    mac = {"label": "Macaque V1 (Ringach 2002)", "n": macaque_cv.size,
           "median": float(np.median(macaque_cv)),
           "pct_selective": float(100.0 * np.mean(macaque_cv < CV_THRESH))}
    print(f"  {mac['label']:32s} {mac['n']:7d} {mac['median']:7.3f} "
          f"{mac['pct_selective']:7.1f}% {'—':>6} {'—':>6}")
    rows = []
    for name, cv in variants:
        r = compare_cv(cv, macaque_cv, name)
        rows.append(r)
        print(f"  {r['label']:32s} {r['n']:7d} {r['median']:7.3f} "
              f"{r['pct_selective']:7.1f}% {r['ks']:6.3f} {r['similarity']:6.3f}")
    if flat_frac is not None:
        print(f"\n  {100*flat_frac:.1f}% of gated vision-band units respond IDENTICALLY to all "
              f"510 gratings\n  before smoothing (flat tuning curve -> CV exactly 1). The "
              f"'modulated' row excludes them.")
    print(f"\n  KS is the max gap between CDFs (0 = identical). p-values are not reported as")
    print(f"  meaningful: with ~10^4 model units against 308 neurons, any real difference is")
    print(f"  'significant'. The KS statistic is the effect size that matters.")
    best = min(rows, key=lambda r: r["ks"])
    print(f"  Closest to macaque: {best['label']} (KS {best['ks']:.3f}, "
          f"similarity {best['similarity']:.3f})")
    print(f"\n  Read this the unfavourable way round: macaque V1 has {mac['pct_selective']:.0f}% "
          f"of neurons below the\n  selectivity cut, and no variant here comes close. Individual "
          f"units are much more\n  weakly orientation tuned than macaque V1 neurons, even though "
          f"the *map* they form\n  (see part 2) is macaque-like in its spatial organisation.")

    # ---- CV as a function of encoder depth
    # Macaque V1 is an early visual area, so the whole 32-layer band is the wrong comparison
    # set. Both statistics are profiled, because they disagree in an informative way: the
    # single-unit (unsmoothed) tuning strength is flat with depth, while the local-population
    # (smoothed) strength has a sharp early/deep split. The gradient people see in the OSI
    # map is therefore about spatial clustering of orientation preference, not about how
    # sharply individual units are tuned.
    print("\n" + "=" * 78)
    print("1b. CV BY ENCODER DEPTH  — which layers, if any, look like V1?")
    print("=" * 78)
    print(f"  macaque: median {np.median(macaque_cv):.3f}, "
          f"{100 * np.mean(macaque_cv < CV_THRESH):.1f}% selective   "
          f"(0 = earliest layer, 31 = deepest)")

    block = 4
    sources = []
    if cv_unsm_2d is not None:
        sources.append(("unsmoothed, modulated (single-unit analogue)", cv_unsm_2d, True))
    sources.append(("smoothed sheet (local-population analogue)", cv_sm_2d, False))

    profiles = {}
    for src_name, src_2d, live_only in sources:
        print(f"\n  {src_name}")
        print(f"  {'layers':>9} {'n':>7} {'median':>7} {'%CV<0.6':>8} {'KS':>6}")
        prof = []
        for lo in range(0, N_VISION_LAYERS, block):
            lays = np.arange(lo, min(lo + block, N_VISION_LAYERS))
            sel = band_gate_all & layer_band_mask(lays)
            if live_only and ptp_2d is not None:
                sel = sel & (ptp_2d > 0)
            vals = src_2d[sel]
            vals = vals[np.isfinite(vals)]
            if vals.size < 50:
                print(f"  {f'{lays.min()}-{lays.max()}':>9} {vals.size:>7}   (too few units)")
                continue
            ks, _ = ks_2samp(vals, macaque_cv)
            prof.append({"lo": int(lays.min()), "hi": int(lays.max()), "n": int(vals.size),
                         "median": float(np.median(vals)),
                         "pct": float(100 * np.mean(vals < CV_THRESH)), "ks": float(ks)})
            print(f"  {f'{lays.min()}-{lays.max()}':>9} {vals.size:>7} "
                  f"{prof[-1]['median']:7.3f} {prof[-1]['pct']:7.1f}% {ks:6.3f}")
        profiles[src_name] = prof
        if prof:
            best = min(prof, key=lambda r: r["ks"])
            spread = max(r["ks"] for r in prof) - min(r["ks"] for r in prof)
            print(f"    best: layers {best['lo']}-{best['hi']} (KS {best['ks']:.3f}, "
                  f"{best['pct']:.1f}% selective); KS spread across depth {spread:.3f}")
            if spread < 0.05:
                print(f"    -> essentially FLAT with depth: restricting to early layers "
                      f"does not improve this comparison")
            else:
                print(f"    -> strong depth dependence: early layers are much closer to V1")

    plot_cv_by_depth(profiles, macaque_cv, savepath=outdir / "cv_by_depth.png")

    # ---- map-level metrics vs figure-read reference
    print("\n" + "=" * 78)
    print("2. MAP-LEVEL METRICS — reference values READ OFF the published figure")
    print("=" * 78)
    if Path(args.metrics).exists():
        m = np.load(args.metrics, allow_pickle=True)
        ours_density = float(m["pinwheel_density"])
        ours_smooth = float(np.nanmean(m["smoothness"]))
        lam = float(m["column_spacing_px"])
        sweep_f = m["sweep_fracs"]; sweep_d = m["sweep_density_sd30"]
    else:
        raise FileNotFoundError(f"{args.metrics} not found; run orientation_metrics first")

    ref_d = FIGURE_REFERENCE["pinwheel_density"]
    ref_s = FIGURE_REFERENCE["map_smoothness"]
    print(f"  {'':22s} {'pinwheels/Λ²':>13} {'smoothness':>11}")
    print(f"  {'Topo-Omni (ours)':22s} {ours_density:13.2f} {ours_smooth:11.3f}")
    for k in ("Macaque V1", "TDANN", "Hand-Crafted SOM", "DNN-SOM", "Task Only",
              "Unoptimized"):
        s = f"{ref_s[k]:11.2f}" if k in ref_s else f"{'—':>11}"
        print(f"  {'~ ' + k:22s} {ref_d[k]:13.1f} {s}   (approx.)")

    print(f"\n  Ours sits {'above' if ours_density > ref_d['Macaque V1'] else 'below'} macaque "
          f"({ours_density:.2f} vs ~{ref_d['Macaque V1']:.1f}) and "
          f"{'above' if ours_density > ref_d['TDANN'] else 'below'} TDANN "
          f"(~{ref_d['TDANN']:.1f}).")
    print(f"  BUT the density depends heavily on the smoothing window — across the swept")
    print(f"  windows it ranges {np.nanmin(sweep_d):.2f}–{np.nanmax(sweep_d):.2f} per Λ²,")
    print(f"  which straddles every reference value above. Treat the ranking as unresolved")
    print(f"  unless the window is matched to TDANN's (window/Λ, not window in mm).")
    print(f"  Smoothness {ours_smooth:.3f} vs macaque ~{ref_s['Macaque V1']:.2f}: ours is "
          f"{'smoother' if ours_smooth > ref_s['Macaque V1'] else 'less smooth'}. Note this")
    print(f"  metric is NOT parameter-free, despite being bounded: it depends on the")
    print(f"  smoothing fwhm applied before the map was built (0.730 at fwhm=4 vs 0.466 at")
    print(f"  fwhm=2 on this sheet, straddling macaque) and on the distance binning, since")
    print(f"  (max - curve[0]) / max moves with the width of the first bin. Only compare")
    print(f"  values computed at the same fwhm AND the same bins -- see")
    print(f"  orientation_metrics, which now uses TDANN's binning, and macaque_map, which")
    print(f"  reconstructs their macaque curve so the two are directly overlaid.")
    print(f"  (Λ = {lam:.1f} sheet px; our band spans only "
          f"{160/lam:.1f}x{256/lam:.1f} hypercolumns, so the density has few Λ² of area")
    print(f"   to average over and edge effects are proportionally large.)")

    plot_cv_comparison(variants, macaque_cv,
                       savepath=outdir / "cv_vs_macaque.png")

    # ---- headline comparison table: macaque vs the smoothed condition being analysed
    from src.experiments.orientation.macaque_map import macaque_delta_curve
    from src.experiments.orientation.metrics import smoothness as _smoo
    try:
        _, mac_curves = macaque_delta_curve()
        mac_mean = np.nanmean(mac_curves, axis=0)
        mac_smoo = float(np.nanmean([_smoo(c) for c in mac_curves]))
    except Exception:                                       # noqa: BLE001
        mac_mean, mac_smoo = None, float("nan")

    # "smoothed" is a substring of "unsmoothed", so exclude the latter explicitly or this
    # silently selects the unsmoothed variant while the table header claims otherwise.
    smoothed_row = next((r for r in rows if "smoothed" in r["label"]
                         and "unsmoothed" not in r["label"]), None)
    if smoothed_row is None:
        print("  ! no smoothed variant in the comparison; summary table skipped")
    ours_curves = m["delta_curves"] if "delta_curves" in m.files else None
    our_mean = np.nanmean(ours_curves, axis=0) if ours_curves is not None else None
    our_smoo = float(np.nanmean(m["smoothness"])) if "smoothness" in m.files else float("nan")

    if smoothed_row is not None:
        fmt = lambda v, s="{:.3f}": "—" if v is None or not np.isfinite(v) else s.format(v)
        mac_col = {
            "units compared (CV)": f"{macaque_cv.size} neurons",
            "median circular variance": fmt(np.median(macaque_cv)),
            f"% orientation selective (CV < {CV_THRESH})":
                fmt(100 * np.mean(macaque_cv < CV_THRESH), "{:.1f}%"),
            "KS distance of CV distribution": "—",
            "CV similarity (1 - KS)": "—",
            "map smoothness": fmt(mac_smoo),
            "Δori, nearest distance bin":
                fmt(mac_mean[0], "{:.1f}°") if mac_mean is not None else "—",
            "Δori, plateau":
                fmt(np.nanmax(mac_mean), "{:.1f}°") if mac_mean is not None else "—",
        }
        our_col = {
            "units compared (CV)": f"{smoothed_row['n']} units",
            "median circular variance": fmt(smoothed_row["median"]),
            f"% orientation selective (CV < {CV_THRESH})":
                fmt(smoothed_row["pct_selective"], "{:.1f}%"),
            "KS distance of CV distribution": fmt(smoothed_row["ks"]),
            "CV similarity (1 - KS)": fmt(smoothed_row["similarity"]),
            "map smoothness": fmt(our_smoo),
            "Δori, nearest distance bin":
                fmt(our_mean[0], "{:.1f}°") if our_mean is not None else "—",
            "Δori, plateau":
                fmt(np.nanmax(our_mean), "{:.1f}°") if our_mean is not None else "—",
        }
        notes = [
            f"Macaque CV: Ringach et al. 2002, {macaque_cv.size} single V1 neurons. Macaque "
            f"map metrics: orientation panel of Nauhaus et al. 2012 Fig. 2, colour-decoded "
            f"(see macaque_map.py).",
            f"Both Δori curves use TDANN's binning (9 bins to 0.943·Λ), distance as a "
            f"fraction of each map's own column spacing. Smoothness = (max − first) / max.",
            f"Column spacing and pinwheel density are omitted: a {len(sel_layers)}-layer "
            f"strip is far too thin vertically to estimate a 2D periodicity or count "
            f"defects, so they exist only for the full band.",
        ]
        plot_summary_table(
            mac_col, our_col,
            label=f"Topo-Omni\n({layer_tag}, smoothed)",
            extra_notes=notes, savepath=outdir / "summary_table.png")
        print(f"> Wrote summary table: {outdir / 'summary_table.png'}")
    np.savez(
        outdir / "reference_comparison.npz",
        macaque_cv=macaque_cv,
        **{f"cv_{i}": cv for i, (_, cv) in enumerate(variants)},
        variant_labels=np.array([n for n, _ in variants]),
        ks=np.array([r["ks"] for r in rows]),
        pct_selective=np.array([r["pct_selective"] for r in rows]),
        macaque_median=mac["median"], macaque_pct_selective=mac["pct_selective"],
        ours_density=ours_density, ours_smoothness=ours_smooth,
    )
    print(f"\n> Saved comparison to: {outdir}")


if __name__ == "__main__":
    main()
