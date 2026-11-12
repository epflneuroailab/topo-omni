"""Human fMRI response profiles, one figure per localizer ROI.

The model-side counterpart is `plot.py`, which draws the same ten-condition bar profile
for a Topo-Omni region. These two are meant to sit next to each other in the figure, so
everything visual is shared: the PALETTE, the condition order and the bar/error-bar
styling are IMPORTED from `plot.py` rather than restated here. Copying the colours would
let the two halves drift the next time the palette changes -- which has already happened
once (scenes/objects/words were three near-identical golds, dE2000 = 3.0).

Input is the per-ROI condition summary written by
`fMRI/Marvi_2025/analysis/extract_condition_responses.py`: one row per (roi, condition)
with the group mean beta across subjects, its SEM, and the subject count. Error bars are
SEM across subjects, matching what the figure captions claim for the brain data.

Usage:
    python -m src.experiments.response_profiles.plot_fmri
    python -m src.experiments.response_profiles.plot_fmri --rois lh_ffa,lh_ppa
    python -m src.experiments.response_profiles.plot_fmri --networks     # language / MD
"""
import os
import glob
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Single source of truth for the colours and the condition order -- see the module
# docstring. `plot.py` calls load_dotenv() at import, which is harmless here.
from src.experiments.response_profiles.plot import (
    palette, map_conditions, conditions, conditions_labels,
)
# blend_colors averages the FB and FP palette entries for the combined "FB+FP" bar, so the
# brain panel and the model panel colour that bar identically.
from src.experiments.response_profiles.plot_cognitive import blend_colors

from dotenv import load_dotenv
load_dotenv()
REPO_DIR = os.getenv("REPO_DIR", ".")
SAVE_DIR = os.getenv("SAVE_DIR", "results")

#: localizer -> (roi_label in the CSV, display name). Mirrors `roi_list` in plot.py, so a
#: localizer's model panel and brain panel name the same region.
ROI_LIST = {
    "faces":          ("lh_ffa",   "FFA"),
    "bodies":         ("lh_eba",   "EBA"),
    "objects":        ("lh_loc",   "LOC"),
    "scenes":         ("lh_ppa",   "PPA"),
    "vwfa":           ("lh_vwfa",  "VWFA"),
    "speech":         ("speech",   "Speech (STG)"),
    "opa":            ("lh_opa",   "OPA"),
    "rsc":            ("lh_rsc",   "RSC"),
}

#: The three cognitive localizers are drawn as TWO bars, not ten, matching the model-side
#: panels in plot_cognitive.py: each one is a single contrast, and the ten-condition
#: profile is not what the figure compares. The ON/OFF spec is copied from that module's
#: `localizer_condition_map` so the two halves stay in step.
#:
#:   localizer          ON        OFF       brain region
#:   language           FB+FP     NW        language network (parcel family)
#:   multiple_demand    MATH      FB+FP     multiple-demand network (parcel family)
#:   theory_of_mind     FB        FP        rTPJ (single parcel)
#:
#: "FB+FP" is the mean of the false-belief and false-photo conditions -- in the EMFL design
#: those are the two English-story conditions, so together they are the language localizer's
#: ON set and the multiple-demand localizer's OFF set.
COGNITIVE = {
    "language":        dict(on="FB+FP", off="NW",  network="language",
                            disp="Language network"),
    "multiple_demand": dict(on="MATH",  off="FB+FP", network="md",
                            disp="Multiple Demand network"),
    "theory_of_mind":  dict(on="FB",    off="FP",  roi="lh_tpj",
                            disp="rTPJ"),
}


def load_summary(csv_path=None):
    if csv_path is None:
        hits = sorted(glob.glob(os.path.join(REPO_DIR, "data",
                                             "condition_responses_summary_*.csv")))
        if not hits:
            raise SystemExit("no data/condition_responses_summary_*.csv found; pass --csv")
        csv_path = hits[-1]
    df = pd.read_csv(csv_path)
    df["code"] = df["condition"].map(map_conditions)
    if df["code"].isna().any():
        bad = sorted(df.loc[df["code"].isna(), "condition"].unique())
        raise SystemExit(f"conditions not in map_conditions: {bad}")
    return df, csv_path


def profile_for(df, roi=None, network=None):
    """(heights, errs, n_subjects) over `conditions`, as a group mean across subjects."""
    sub = df[df.roi_label == roi] if roi else df[df.parcel_category == network]
    if sub.empty:
        raise SystemExit(f"no rows for {'roi ' + roi if roi else 'network ' + network}")
    g = sub.groupby("code")
    heights = g["mean_beta_mean"].mean().reindex(conditions)
    # Averaging parcels within a network: the parcel means are not independent draws, so
    # propagate their SEMs in quadrature rather than taking the SEM across parcels.
    n_parcels = g["mean_beta_mean"].size().reindex(conditions)
    errs = (g["mean_beta_sem"].apply(lambda s: np.sqrt((s ** 2).sum()))
            .reindex(conditions) / n_parcels)
    return heights, errs, int(sub["mean_beta_count"].max())


def blended(heights, errs, code):
    """Value and SEM for a condition code, where 'FB+FP' means the mean of the two.

    The SEMs are combined in quadrature and halved, i.e. the SEM of the MEAN of two
    conditions -- not their sum, and not the SEM of one of them.
    """
    if code == "FB+FP":
        h = (heights["FB"] + heights["FP"]) / 2.0
        e = np.sqrt(errs["FB"] ** 2 + errs["FP"] ** 2) / 2.0
        return h, e
    return heights[code], errs[code]


def draw_two_bar(heights, errs, spec, title, savepath, from_zero=False):
    """ON vs OFF, styled like the model-side cognitive panel in plot_cognitive.py.

    Square (4x4) rather than the wide ten-condition panel, x-tick labels shown, and no
    visual/non-visual divider -- that line only makes sense with all ten conditions.
    """
    codes = [spec["on"], spec["off"]]
    vals, errbars = zip(*(blended(heights, errs, c) for c in codes))
    vals, errbars = np.array(vals), np.array(errbars)

    sns.set_theme(context="paper", font_scale=2, style="whitegrid")
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor("white")

    # Baseline is taken from the bar values MINUS their error bars, not from the values
    # alone. With the values alone a condition whose SEM reaches below the smallest bar
    # has its lower whisker clipped off by the axis -- visible on rTPJ, where FP's SEM
    # extends well past the shortest bar.
    lo, hi = (vals - errbars).min(), (vals + errbars).max()
    baseline = 0.0 if from_zero else lo - 0.1 * (hi - lo)

    ax.bar(
        range(len(codes)),
        vals - baseline,
        yerr=errbars,
        color=[blend_colors(c) for c in codes],
        edgecolor="black",
        linewidth=1,
        error_kw=dict(ecolor="0.25", capsize=0, lw=2),
    )
    ax.set_xticks(range(len(codes)))
    ax.set_xticklabels(codes)
    ax.set_xlim(-0.5, len(codes) - 0.5)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y + baseline:.2f}"))
    sns.despine()
    plt.xlabel("")
    plt.ylabel("Mean Beta", fontweight="bold")
    if title:
        ax.set_title(title, fontweight="bold")
    plt.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(f"{savepath}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return dict(zip(codes, vals))


def draw(heights, errs, title, savepath, from_zero=False):
    """Bar profile styled exactly like the model-side panel in plot.py."""
    sns.set_theme(context="paper", font_scale=2, style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 4))

    # Same trick plot.py uses: shift so the smallest bar is just above the axis, then
    # relabel the ticks with the true values. Without it a profile whose conditions differ
    # by a few percent renders as ten bars of visually identical height.
    # Baseline is taken from the bar values MINUS their error bars, not from the values
    # alone. With the values alone a condition whose SEM reaches below the smallest bar
    # has its lower whisker clipped off by the axis -- visible on the cognitive panels.
    lo = (heights - errs).min()
    hi = (heights + errs).max()
    baseline = 0.0 if from_zero else lo - 0.1 * (hi - lo)

    ax.bar(
        range(len(conditions)),
        (heights - baseline).values,
        yerr=errs.values,
        color=[palette[c] for c in conditions],
        edgecolor="black",
        linewidth=1,
        error_kw=dict(ecolor="0.25", capsize=0, lw=2),
    )
    ax.set_xlim(-0.6, len(conditions) - 0.4)
    ax.set_xticks([])
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y + baseline:.2f}"))
    sns.despine()
    ax.axvline(4.5, color="lightgray", linewidth=2)   # visual | non-visual divider
    plt.xlabel("")
    plt.ylabel("Mean Beta", fontweight="bold")
    if title:
        ax.set_title(title, fontweight="bold")
    plt.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(f"{savepath}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default=None,
                   help="per-ROI condition summary. Default: the newest "
                        "data/condition_responses_summary_*.csv")
    p.add_argument("--rois", default=None,
                   help="comma-separated roi_labels. Default: the localizer ROIs in "
                        "ROI_LIST. Pass 'all' for every ROI in the CSV.")
    p.add_argument("--no_cognitive", action="store_true",
                   help="skip the three cognitive two-bar panels (language, multiple "
                        "demand, theory of mind).")
    p.add_argument("--outdir", default=None,
                   help="default: $SAVE_DIR/fmri_response_profiles")
    p.add_argument("--from_zero", action="store_true",
                   help="start bars at 0 instead of just below the smallest condition. "
                        "Off by default to match the model-side panels.")
    p.add_argument("--title", action="store_true",
                   help="draw the ROI name above each panel. OFF by default: these are "
                        "composed in drawio, which supplies its own headings, and the "
                        "model-side panels in plot.py carry no title either.")
    args = p.parse_args()

    df, csv_path = load_summary(args.csv)
    outdir = args.outdir or os.path.join(SAVE_DIR, "fmri_response_profiles")
    os.makedirs(outdir, exist_ok=True)
    print(f"> Source: {csv_path}")
    print(f"> Palette: imported from response_profiles.plot "
          f"({len(palette)} conditions)")
    print(f"> Output: {outdir}")

    if args.rois == "all":
        targets = [(r, r) for r in sorted(df.roi_label.unique())]
    elif args.rois:
        want = [r.strip() for r in args.rois.split(",") if r.strip()]
        name = {v[0]: v[1] for v in ROI_LIST.values()}
        targets = [(r, name.get(r, r)) for r in want]
    else:
        targets = [(roi, disp) for roi, disp in ROI_LIST.values()]

    for roi, disp in targets:
        h, e, n = profile_for(df, roi=roi)
        draw(h, e, f"{disp}  (n={n})" if args.title else None,
             os.path.join(outdir, f"fmri_response_profile_{roi}"),
             from_zero=args.from_zero)
        print(f"  {roi:<12} {disp:<22} n={n}  "
              f"range [{h.min():.3f}, {h.max():.3f}]  -> {roi}.png/.svg")

    if not args.no_cognitive:
        print("\n> Cognitive localizers (two bars, as in plot_cognitive.py):")
        for loc, spec in COGNITIVE.items():
            h, e, n = profile_for(df, roi=spec.get("roi"), network=spec.get("network"))
            got = draw_two_bar(
                h, e, spec, f"{spec['disp']}  (n={n})" if args.title else None,
                os.path.join(outdir, f"fmri_response_profile_cognitive_{loc}"),
                from_zero=args.from_zero)
            bars = "  ".join(f"{k}={v:+.3f}" for k, v in got.items())
            print(f"  {loc:<16} {spec['disp']:<24} n={n}  {bars}  "
                  f"-> cognitive_{loc}.png/.svg")

    print(f"\n> Condition order: {' '.join(conditions)}")
    print(f"> Labels        : {', '.join(conditions_labels)}")


if __name__ == "__main__":
    main()
