"""
Reproduce the bar plots on the right of TDANN Fig. 2E (orientation) and 2G (spatial
frequency): map smoothness for Topo-Omni and TDANN against grey "alternative model" bars,
with macaque V1 as a dashed reference line. Also runs the statistics.

Which alternative models are real here, and which are not
---------------------------------------------------------
From ``notebooks/F02_V1_Details``, the comparison models are built like this:

``TDANN`` = ``SW_0pt25``
    checkpoint ``..._isoswap_3{seed}``, swap-optimised positions. Downloadable -- included,
    with 4 of their 5 seeds (the seed-2 checkpoint is not on their OSF node).
``Task Only`` = ``SW_0``
    the same architecture trained with spatial weight 0, ``..._isoswap_3_lw0``. Downloadable
    -- included, but only the base checkpoint exists on OSF, so n = 1.
``Unoptimized``
    the same config read at ``step="random"`` on *retinotopic* (un-swapped) positions, i.e.
    an untrained network on unoptimised positions. Reproduced here directly with randomly
    initialised weights and their retinotopic position file, over 5 seeds.
``DNN-SOM`` and ``Hand-Crafted SOM``
    **not included.** These are not checkpoints; the notebook *trains* them in place
    (``FeatureRichV1SOM.build`` / ``FeaturePoorV1SOM.build``, 100k and 700k iterations).
    Reproducing them means porting their SOM implementation and training it, so any bar here
    would be a SOM of mine rather than theirs. Omitted rather than guessed.

Units caveat, inherited from TDANN
----------------------------------
Orientation preference is an angle, so its pairwise difference wraps at 90 deg. Spatial
frequency is linear and its "preference" is the *index* of the preferred SF for models, but
log2(cpd) for the colour-decoded macaque panel. TDANN mixes these units the same way. It is
only defensible because ``smoothness = (max - first) / max`` is scale-free -- the raw SF
curves are not comparable in absolute terms, and are not plotted as such.

Statistics
----------
The honest limitation is replication. TDANN and Unoptimized have several seeds, so they get a
real across-seed spread. Task Only has one checkpoint and Topo-Omni is a single training run,
so for those the only available spread is a bootstrap over the curve's own resampling, which
measures estimation noise, *not* model variability, and cannot support a between-model test.
Reported accordingly: across-seed statistics where seeds exist, and for the single-run models
a z-score against TDANN's seed distribution instead of a p-value.

Usage
-----
python -m src.experiments.orientation.smoothness_panels \
    --stats results/topo-omni/orientation_unbiased/orientation_stats.npz \
    --sheets results/topo-omni/orientation_unbiased/cortical_sheets.npy \
    --bank unbiased --layers 0-3 --fwhm 4.0
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from scipy import stats as sps
from scipy.stats import sem

from src.experiments.orientation.macaque_map import (
    tdann_bin_edges, delta_over_distance, macaque_delta_curve, macaque_sf_delta_curve,
)
from src.experiments.orientation.metrics import (
    band_grid, layer_band_mask, parse_layers, smoothness, column_spacing,
    N_VISION_LAYERS,
)
from src.experiments.orientation.tdann_model import (
    _fetch, load_tdann_resnet18, tdann_layer2_responses,
    TDANN_COLUMN_SPACING_MM, TDANN_PTP_PERCENTILE,
)
from src.experiments.orientation.run import parse_stimuli, orientation_preference

from dotenv import load_dotenv
load_dotenv()

SWAP_POSITIONS_URL = "https://osf.io/download/279sy/"
RETINOTOPIC_POSITIONS_URL = "https://osf.io/download/64664a01f9c6ce0bbbd4b948/"

# name -> {seed: checkpoint url}; None means "randomly initialised weights"
TDANN_FAMILY = {
    "TDANN": {
        0: "https://osf.io/download/exugm/",
        1: "https://osf.io/download/xstvj/",
        3: "https://osf.io/download/8twsz/",
        4: "https://osf.io/download/dtnjw/",
    },
    "Task Only": {0: "https://osf.io/download/8q32t/"},
    "Unoptimized": {s: None for s in range(5)},
}
POSITIONS_FOR = {
    "TDANN": SWAP_POSITIONS_URL,
    "Task Only": SWAP_POSITIONS_URL,
    "Unoptimized": RETINOTOPIC_POSITIONS_URL,   # un-swapped, as their "Unoptimized" uses
}
IS_ALTERNATIVE = {"TDANN": False, "Task Only": True, "Unoptimized": True}

# metric -> (label, y-axis label, circular?, normalise lines to chance?)
METRICS = {
    "orientation": ("E", "$\\Delta$ preferred orientation (/ chance)", True, True),
    "sf": ("G", "$\\Delta$ spatial frequency (/ chance)", False, True),
    # Normalised, like sf: the macaque reference is the mean |delta| of a *continuous*
    # blobness value (chance ~0.29 for its spread) while the models give the fraction of
    # pairs in *different binary* colour categories (chance ~0.49). Plotted raw, the macaque
    # line sits far below the models for purely definitional reasons.
    "color": ("I", "$\\Delta$ colour preference (/ chance)", False, True),
}

# Panel I's macaque reference is not in TDANN's public repo: LivingstoneColorTissue reads
# notebooks/assets/F02/Livingstone_1984_Figure27B.png, which 404s and is absent from their
# whole tree (unlike the Nauhaus figure behind panels E and G). Supply the figure yourself and
# it is decoded for real; the constant below is only the fallback when no file is present.
MACAQUE_COLOR_SMOOTHNESS_FIGURE_READ = 0.50
LIVINGSTONE_IMAGE = Path("data/Livingstone_1984_Figure27B.png")
# Chosen from a scan for windows lying wholly inside the blob field (no off-tissue pixels)
# with maximal power in the 0.40-0.60 mm blob band; 170 px = 3.8 mm at the scale below,
# comparable to TDANN's own ~3.4 mm crop. Smoothness varies by only 0.003 across the three
# best such windows, so this choice does not drive the result.
LIVINGSTONE_MAP_BOX = (390, 610, 560, 780)
LIVINGSTONE_SCALE_BAR_PX = 45.0      # detected: the sole dark wide-flat blob, y1057-1065 x731-776
LIVINGSTONE_SCALE_BAR_MM = 1.0


# -------------------------------------------------
# Preferences
# -------------------------------------------------

def grouped_mean(responses: np.ndarray, keys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mean response per unique key, ascending. Works for non-contiguous groupings.

    ``orientation_tuning_curve`` assumes each orientation's stimuli are contiguous, which
    holds for orientation but not for spatial frequency, since the bank is ordered
    orientation-major.
    """
    keys = np.asarray(keys)
    uniq = np.unique(keys)
    out = np.stack([responses[keys == k].mean(axis=0) for k in uniq])
    return uniq, out


def preferred_sf_index(responses: np.ndarray, sf_keys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Preferred spatial frequency as an index into the sorted SFs, TDANN's convention.

    Their ``Responses.get_preferences`` is ``groupby(metric).mean().argmax(axis=0)``, i.e.
    the index of the best-driving SF rather than its cpd value.
    """
    uniq, curve = grouped_mean(responses, sf_keys)
    return curve.argmax(axis=0).astype(float), curve


def preferred_orientation(responses: np.ndarray, ori_keys: np.ndarray,
                          baseline: str = "min") -> Tuple[np.ndarray, np.ndarray]:
    uniq, curve = grouped_mean(responses, ori_keys)
    pref, _ = orientation_preference(curve, uniq, baseline=baseline)
    return pref, curve


def preferred_color(responses: np.ndarray, colour_keys: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Binary colour preference: 0 = black/white gratings, 1 = red/cyan.

    With a 0/1 preference the absolute pairwise difference is 1 exactly when two units
    prefer different colour families, so averaging it per distance bin gives TDANN's
    "fraction preferring the other colour" directly -- no separate estimator needed.
    """
    uniq, curve = grouped_mean(responses, colour_keys)
    if len(uniq) != 2:
        raise ValueError(f"expected 2 colour families, got {uniq}")
    return curve.argmax(axis=0).astype(float), curve


# -------------------------------------------------
# Curves per model
# -------------------------------------------------

def _model_curves(responses: np.ndarray, positions: np.ndarray, meta: dict,
                  ori_keys: np.ndarray, column_spacing_val: float,
                  n_bins: int, num_samples: int, sample_size: int,
                  ptp_percentile: float = None) -> Dict[str, np.ndarray]:
    """Orientation and SF Delta-vs-distance curves for one model."""
    edges = tdann_bin_edges(column_spacing_val, n_bins=n_bins)
    out = {}

    prefs = {
        "orientation": preferred_orientation(responses, ori_keys),
        "sf": preferred_sf_index(responses, meta["sf"]),
        "color": preferred_color(responses, meta["colour"]),
    }

    for tag, (pref, curve) in prefs.items():
        circular = METRICS[tag][2]
        keep = np.isfinite(pref)
        if ptp_percentile is not None:
            ptp = curve.max(axis=0) - curve.min(axis=0)
            keep &= ptp >= np.percentile(ptp, ptp_percentile)
        common = dict(num_samples=num_samples, sample_size=sample_size, circular=circular)
        out[tag] = delta_over_distance(pref[keep], positions[keep], edges, **common)
        # Distance-shuffled control, used as the chance level when normalising lines whose
        # units differ between model and reference (spatial frequency).
        out[tag + "_shuffled"] = delta_over_distance(
            pref[keep], positions[keep], edges, shuffle=True, **common)
    return out


def tdann_family_curves(name: str, seed: int, paths: List[str], meta: dict,
                        ori_keys: np.ndarray, data_dir: Path, device: str = "cuda",
                        n_bins: int = 9, num_samples: int = 20,
                        sample_size: int = 1000) -> Dict[str, np.ndarray]:
    """Curves for one TDANN-family model/seed, running the network on the gratings."""
    from torchvision.models import resnet18

    url = TDANN_FAMILY[name][seed]
    if url is None:
        # "Unoptimized": untrained weights, matching their step="random"
        torch.manual_seed(seed)
        model = resnet18().eval().to(device)
    else:
        ckpt = _fetch(url, Path(data_dir) / f"{name.replace(' ', '_')}_seed{seed}.torch")
        model, _ = load_tdann_resnet18(ckpt, device=device)

    pos_file = _fetch(POSITIONS_FOR[name],
                      Path(data_dir) / f"positions_{'retino' if url is None else 'swap'}.npz")
    positions = np.load(pos_file, allow_pickle=True)["coordinates"].astype(float)

    resp = tdann_layer2_responses(model, paths, device=device)
    if positions.shape[0] != resp.shape[1]:
        raise RuntimeError(f"{positions.shape[0]} positions vs {resp.shape[1]} units")

    return _model_curves(resp, positions, meta, ori_keys, TDANN_COLUMN_SPACING_MM,
                         n_bins, num_samples, sample_size,
                         ptp_percentile=TDANN_PTP_PERCENTILE)


def topo_omni_curves(stats_path: Path, sheets_path: Path, bank: str, layers: str,
                     fwhm_mm: float, coords_path: str, n_bins: int = 9,
                     num_samples: int = 20, sample_size: int = 1000,
                     lam_override: float = None
                     ) -> Tuple[Dict[str, np.ndarray], float]:
    """Curves for Topo-Omni, on the smoothed sheet, restricted to ``layers``.

    ``lam_override`` pins the column spacing instead of estimating it from this model's
    own map. Distance bins are fractions of Lambda, so a per-model Lambda measures each
    model on a different ruler -- and ``column_spacing`` only returns a meaningful value
    when the map HAS a periodicity. On a structureless map it fits noise in the power
    spectrum and returns a small Lambda, which narrows the bins, samples only very close
    pairs, and inflates smoothness for exactly the models that have no structure. TDANN
    avoids this by holding every bar in its family at a fixed TDANN_COLUMN_SPACING_MM;
    this is the same fix on our side.
    """
    from src.utils.smoothing import NeuronSmoothingConv
    from src.experiments.orientation.run import GRATING_BANKS

    d = np.load(stats_path, allow_pickle=True)
    gate = d["gate"] if "gate" in d.files else d["tuned"]

    stimuli_root = Path(os.environ["STIMULI_DIR"]) / GRATING_BANKS[bank][1]
    paths, ori_deg, counts, meta = parse_stimuli(stimuli_root, color="all")
    ori_keys = np.repeat(ori_deg, counts)

    raw = np.load(sheets_path)
    coords = np.load(coords_path)
    smoothed = NeuronSmoothingConv(fwhm_mm=fwhm_mm, resolution_mm=1.0)(coords, raw)

    # Preferences on the band grid, then keep the selected units and their grid positions.
    pref_o_full, _ = preferred_orientation(smoothed, ori_keys)
    ori_band = band_grid(pref_o_full)

    # Lambda must be estimated on the FULL band, exactly as orientation_metrics does. A
    # 4-layer strip is 20 rows tall, well under one column spacing, so its power spectrum
    # cannot resolve the periodicity -- estimating from the strip returns ~75 px instead of
    # ~32 px, which widens every distance bin and silently changes smoothness.
    valid_full = (band_grid(gate.astype(float)) > 0.5) & np.isfinite(ori_band)
    if lam_override is not None:
        lam = float(lam_override)
    else:
        lam, _, _ = column_spacing(ori_band, valid_full)

    sel_layers = parse_layers(layers)
    mask2d = valid_full & layer_band_mask(sel_layers)

    yy, xx = np.nonzero(mask2d)
    positions = np.stack([xx, yy], axis=1).astype(float)
    flat = np.stack([band_grid(smoothed[i]) for i in range(smoothed.shape[0])])[:, mask2d]

    curves = _model_curves(flat, positions, meta, ori_keys, lam,
                           n_bins, num_samples, sample_size, ptp_percentile=None)
    return curves, float(lam)


# -------------------------------------------------
# Stats
# -------------------------------------------------

# A curve that is flat *at zero* means the map has no diversity in that property at all --
# every unit prefers the same thing, so no pair ever differs. TDANN's `smoothness` returns
# 1.0 for any flat curve (its `ptp == 0` branch), which would report such a degenerate map as
# perfectly smooth. Curves whose mean level is below this are reported as undefined instead.
DEGENERATE_LEVEL = 0.01


def summarise(per_seed: Dict[int, np.ndarray]) -> dict:
    """Smoothness summary for one model: across-seed spread if possible, else bootstrap."""
    levels = {s: float(np.nanmean(curves)) for s, curves in per_seed.items()}
    degenerate = {s: lv < DEGENERATE_LEVEL for s, lv in levels.items()}
    seed_vals = {s: (float("nan") if degenerate[s]
                     else float(np.nanmean([smoothness(c) for c in curves])))
                 for s, curves in per_seed.items()}
    vals = np.array([v for v in seed_vals.values() if np.isfinite(v)], dtype=float)
    n_degen = sum(degenerate.values())

    if vals.size == 0:
        return {"mean": float("nan"), "spread": 0.0, "n": 0,
                "kind": f"undefined - flat-at-zero curve in all {len(levels)} seed(s)",
                "seed_values": seed_vals, "per_curve": None, "n_degenerate": n_degen,
                "level": float(np.nanmean(list(levels.values())))}

    if len(vals) > 1:
        kind = "across seeds" + (f", {n_degen} degenerate seed(s) dropped" if n_degen else "")
        return {"mean": float(vals.mean()), "spread": float(vals.std(ddof=1)),
                "n": len(vals), "kind": kind, "seed_values": seed_vals,
                "per_curve": None, "n_degenerate": n_degen,
                "level": float(np.nanmean(list(levels.values())))}

    # single model: bootstrap the resampled curves. Estimation noise only.
    # Use the seed that SURVIVED the degeneracy filter, not simply the first one. Taking
    # `next(iter(...))` here silently bootstrapped a rejected seed whenever the filter
    # dropped some, which inflated the Unoptimized colour bar from ~0.07 to ~0.59.
    surviving = [s for s, v in seed_vals.items() if np.isfinite(v)]
    chosen = surviving[0] if surviving else next(iter(per_seed))
    curves = per_seed[chosen]
    per_curve = np.array([smoothness(c) for c in curves], dtype=float)
    rng = np.random.default_rng(0)
    boots = [np.nanmean(rng.choice(per_curve, per_curve.size, replace=True))
             for _ in range(2000)]
    kind = "bootstrap over resamples" + (f", seed {chosen} (only non-degenerate one)"
                                        if n_degen else "")
    return {"mean": float(np.nanmean(per_curve)), "spread": float(np.std(boots, ddof=1)),
            "n": 1, "kind": kind, "seed_values": seed_vals,
            "per_curve": per_curve, "n_degenerate": n_degen,
            "level": float(levels[chosen])}


def summarise_plateau_ratio(per_seed: Dict[int, np.ndarray],
                            per_seed_sh: Dict[int, np.ndarray]) -> dict:
    """Plateau of the curve divided by its own distance-shuffled level.

    ``smoothness = (max - first) / max`` is invariant to any rescaling of the curve, so
    normalising to chance cannot change it -- that invariance is exactly why curves in
    different units were comparable in the first place. What normalisation *does* reveal is
    the plateau level: whether a map's preferences ever fully decorrelate (ratio ~1) or stay
    correlated out to the largest distance measured (ratio < 1). Macaque colour sits well
    below 1 while every model reaches it, and smoothness is blind to that.
    """
    vals = {}
    for s, curves in per_seed.items():
        chance = float(np.nanmean(per_seed_sh[s]))
        if not np.isfinite(chance) or chance <= 0:
            vals[s] = float("nan")
            continue
        vals[s] = float(np.nanmax(np.nanmean(curves, axis=0)) / chance)

    finite = np.array([v for v in vals.values() if np.isfinite(v)], dtype=float)
    if finite.size > 1:
        return {"mean": float(finite.mean()), "spread": float(finite.std(ddof=1)),
                "n": finite.size, "kind": "across seeds", "seed_values": vals,
                "per_curve": None, "n_degenerate": 0, "level": float("nan")}
    if finite.size == 0:
        return {"mean": float("nan"), "spread": 0.0, "n": 0, "kind": "undefined",
                "seed_values": vals, "per_curve": None, "n_degenerate": len(vals),
                "level": float("nan")}

    seed = next(s for s, v in vals.items() if np.isfinite(v))
    chance = float(np.nanmean(per_seed_sh[seed]))
    per_curve = np.nanmax(per_seed[seed], axis=1) / chance
    rng = np.random.default_rng(0)
    boots = [np.nanmean(rng.choice(per_curve, per_curve.size, replace=True))
             for _ in range(2000)]
    return {"mean": float(finite[0]), "spread": float(np.std(boots, ddof=1)), "n": 1,
            "kind": "bootstrap over resamples", "seed_values": vals,
            "per_curve": per_curve, "n_degenerate": 0, "level": float("nan")}


def run_stats(summaries: Dict[str, dict], macaque: float, metric: str) -> List[str]:
    """Between-model statistics, restricted to what the replication actually supports."""
    lines = [f"--- {metric} ---",
             f"  macaque V1 reference: {macaque:.3f}"]
    for name, s in summaries.items():
        if not np.isfinite(s["mean"]):
            lines.append(f"  {name:<22} undefined  ({s['kind']}; mean curve level "
                         f"{s['level']:.4f} -- no diversity in this property, so "
                         f"'smoothness' is not meaningful)")
            continue
        note = f"; {s['n_degenerate']} degenerate seed(s) excluded" if s.get("n_degenerate") else ""
        lines.append(f"  {name:<22} {s['mean']:.3f} +/- {s['spread']:.3f} "
                     f"(n={s['n']}, {s['kind']}); |delta macaque| = "
                     f"{abs(s['mean'] - macaque):.3f}{note}")

    seeded = {k: v for k, v in summaries.items() if v["n"] > 1}
    if len(seeded) >= 2:
        groups = [np.array(list(v["seed_values"].values())) for v in seeded.values()]
        f, p = sps.f_oneway(*groups)
        lines.append(f"  one-way ANOVA across models with seeds "
                     f"({', '.join(seeded)}): F={f:.2f}, p={p:.2g}")
        names = list(seeded)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = np.array(list(seeded[names[i]]["seed_values"].values()))
                b = np.array(list(seeded[names[j]]["seed_values"].values()))
                t, pp = sps.ttest_ind(a, b, equal_var=False)
                u, pu = sps.mannwhitneyu(a, b, alternative="two-sided")
                lines.append(f"    {names[i]} vs {names[j]}: Welch t={t:.2f}, p={pp:.2g}"
                             f" | Mann-Whitney U={u:.0f}, p={pu:.2g}")
    else:
        lines.append("  (too few models with independent seeds for an ANOVA)")

    ref = summaries.get("TDANN")
    if ref is not None and ref["n"] > 1 and ref["spread"] > 0:
        for name, s in summaries.items():
            if s["n"] > 1 or name == "TDANN":
                continue
            z = (s["mean"] - ref["mean"]) / ref["spread"]
            lines.append(f"  {name}: z = {z:+.2f} against TDANN's seed distribution "
                         f"(n=1, so no p-value -- one training run has no model variance)")
    return lines


# -------------------------------------------------
# Plot
# -------------------------------------------------

def plot_metric_lines(per_seed: Dict[str, Dict[int, np.ndarray]],
                      shuffled: Dict[str, Dict[int, np.ndarray]],
                      x_by_model: Dict[str, np.ndarray],
                      macaque: Tuple[np.ndarray, np.ndarray],
                      ylabel: str, title: str, normalise: bool,
                      savepath=None, figsize=(5.6, 3.9)):
    """Left-hand panel of TDANN Fig. 2E/2G/2I: mean curve per model, seeds averaged.

    ``normalise`` divides each curve by its own distance-shuffled level. That is required
    for spatial frequency, where the model preference is an SF *index* but the macaque panel
    decodes to log2(cpd) -- without it the two are on incomparable axes. Orientation (degrees)
    and colour (a fraction) are already common-scale, so they are plotted raw.
    """
    # Same palette as the bars, and the whole TDANN family is dashed so the two model
    # families stay separable when their curves overlap.
    colours = {"Topo-Omni": "#5B2C9E", "Topo-Omni Task Only": "#9B7FC4",
               "Topo-Omni Step 0": "#B69ED4", "Topo-Omni Untrained": "#C3B2DC",
               "TDANN": "#C25E00", "Task Only": "#E3A669", "Unoptimized": "#EBC9A4"}
    TDANN_FAMILY_NAMES = {"TDANN", "Task Only", "Unoptimized"}
    fig, ax = plt.subplots(figsize=figsize)

    for name, seeds in per_seed.items():
        curves = np.concatenate([seeds[s] for s in sorted(seeds)], axis=0)
        mean = np.nanmean(curves, axis=0)
        if normalise:
            sh = np.concatenate([shuffled[name][s] for s in sorted(shuffled[name])], axis=0)
            mean = mean / np.nanmean(sh)
        err = sem(curves, axis=0, nan_policy="omit")
        if normalise:
            err = err / np.nanmean(sh)
        x = 100.0 * x_by_model[name]
        ls = "--" if name in TDANN_FAMILY_NAMES else "-"
        ax.plot(x, mean, lw=1.9, ls=ls, color=colours.get(name, "0.6"), label=name)
        ax.fill_between(x, mean - err, mean + err, color=colours.get(name, "0.6"),
                        alpha=0.20, linewidth=0)

    if macaque is not None:
        # A list of (label, x, curves, shuffled) when a metric has several reference sources.
        refs = macaque if isinstance(macaque, list) else [("Macaque V1", *macaque)]
        ref_styles = [("#1B7F3B", "-"), ("#6FBF8B", ":")]
        for k, (rlabel, mx, mc, mc_sh) in enumerate(refs):
            mm = np.nanmean(mc, axis=0)
            if normalise:
                # Divide by the reference's OWN distance-shuffled level, the same treatment
                # the model curves get. Using its asymptote instead would be wrong wherever
                # the map is still correlated at the largest distance measured.
                mm = mm / np.nanmean(mc_sh)
            col, ls = ref_styles[k % len(ref_styles)]
            ax.plot(100.0 * mx, mm, lw=2.0, ls=ls, color=col, label=rlabel)

    ax.set_xlabel("pairwise distance (% column spacing)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=7.5)
    sns.despine(ax=ax)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig, ax


# Two model families, each carrying the SAME three training conditions, so the effect of
# the spatial loss reads off within a family rather than across families. Shade runs dark
# (spatial + task) to light (untrained).
#
# Both "untrained" bars are randomly initialised weights: TDANN's is a random resnet18
# (see tdann_family_curves), Topo-Omni's is the architecture built via `_from_config` with
# no pretrained checkpoint (load_topo_omni(random_init=True)), adaptors at identity. The
# one residual asymmetry is positional: TDANN's Unoptimized also reverts to retinotopic
# (un-swapped) positions, whereas Topo-Omni's sheet coordinates are fixed by coords.npy
# for every condition, so only the weights differ.
BAR_GROUPS = (
    ("Topo-Omni", ("#5B2C9E", "#9B7FC4", "#B69ED4", "#D9CEEA"),
     (("Topo-Omni", "Spatial\n+ Task"),
      ("Topo-Omni Task Only", "Task\nOnly"),
      # Step 0 sits between Task Only and Untrained on purpose: it keeps the PRETRAINED
      # backbone and only the cortical adaptors are at init, so it isolates what the
      # spatial loss added on top of representations the backbone already had. Untrained
      # discards the backbone too and is the absolute floor.
      ("Topo-Omni Step 0", "Pretrained"),
      ("Topo-Omni Untrained", "Untrained"))),
    ("TDANN", ("#C25E00", "#E3A669", "#F5E0C9"),
     (("TDANN", "Spatial\n+ Task"),
      ("Task Only", "Task\nOnly"),
      ("Unoptimized", "Untrained"))),
)

TOPO_UNTRAINED_NOTE = ("Both 'Untrained' bars are randomly initialised weights. TDANN's also "
                       "reverts to retinotopic positions;\n  Topo-Omni's sheet coordinates are "
                       "fixed across all conditions, so only the weights differ.")


def plot_smoothness_bars(summaries: Dict[str, dict], macaque, title: str,
                         savepath=None, approx: bool = False, figsize=(6.6, 4.1)):
    """TDANN Fig. 2E/2G-style bars, grouped into the Topo-Omni and TDANN families."""
    fig, ax = plt.subplots(figsize=figsize)
    xt, xl, spans, placed = [], [], [], set()
    x, gap = 0.0, 1.0

    def draw(xpos, key, colour):
        s = summaries.get(key)
        if s is None or not np.isfinite(s["mean"]):
            ax.text(xpos, 0.015, "n/a", ha="center", fontsize=7.5, color="0.45")
            if s is not None:
                placed.add(key)
            return
        ax.bar(xpos, s["mean"], yerr=s["spread"], capsize=3, width=0.82,
               color=colour, edgecolor="0.25", linewidth=0.7,
               error_kw={"lw": 1.0, "ecolor": "0.3"})
        if s["n"] > 1:
            pts = [v for v in s["seed_values"].values() if np.isfinite(v)]
            ax.scatter([xpos] * len(pts), pts, s=9, color="0.2", zorder=3, linewidths=0)
        placed.add(key)

    for gname, palette, members in BAR_GROUPS:
        start = x
        for (key, blabel), colour in zip(members, palette):
            draw(x, key, colour)
            xt.append(x); xl.append(blabel); x += 1.0
        spans.append((gname, start, x - 1.0))
        x += gap

    # Anything the groups did not cover is still drawn, rather than silently dropped.
    leftover = [k for k in summaries if k not in placed]
    if leftover:
        print(f"  ! bars not in BAR_GROUPS, appended as 'other': {', '.join(leftover)}")
        start = x
        for key in leftover:
            draw(x, key, "0.78")
            xt.append(x); xl.append(key.replace(" ", "\n")); x += 1.0
        spans.append(("other", start, x - 1.0))

    refs = macaque if isinstance(macaque, (list, tuple)) else [("V1", macaque)]
    ref_styles = [("#1B7F3B", "--"), ("#6FBF8B", ":")]
    for k, (rlabel, rval) in enumerate(refs):
        if not np.isfinite(rval):
            continue
        col, ls = ref_styles[k % len(ref_styles)]
        ax.axhline(rval, ls=ls, lw=1.5, color=col)
        ax.text(x - 0.55, rval, " " + rlabel + ("*" if approx else ""),
                color=col, va="center", fontsize=8, weight="bold")

    ax.set_xticks(xt)
    ax.set_xticklabels(xl, fontsize=7.8)
    ax.set_xlim(-0.7, x - 0.3)
    ax.set_ylabel("Smoothness")
    ax.set_title(title, fontsize=10)
    tops = [s["mean"] + s["spread"] for s in summaries.values() if np.isfinite(s["mean"])]
    tops += [v for _, v in refs if np.isfinite(v)]
    top = max(1.02, (max(tops) if tops else 1.0) * 1.15)
    ax.set_ylim(0, top)
    sns.despine(ax=ax)

    # Family labels on a bracket beneath the condition labels.
    for gname, lo, hi in spans:
        ax.annotate("", xy=(lo - 0.4, -0.155), xytext=(hi + 0.4, -0.155),
                    xycoords=("data", "axes fraction"),
                    textcoords=("data", "axes fraction"),
                    arrowprops=dict(arrowstyle="-", lw=1.1, color="0.3"))
        ax.text(0.5 * (lo + hi), -0.20, gname, ha="center", va="top", fontsize=9.5,
                weight="bold", transform=ax.get_xaxis_transform())

    notes = []
    if any(k.startswith("Topo-Omni Untrained") for k in placed):
        notes.append(TOPO_UNTRAINED_NOTE)
    if approx:
        notes.append("* macaque colour reference read off the published panel: the "
                     "Livingstone 1984\n  figure it decodes is not in TDANN's repo, so it "
                     "cannot be reproduced here.")
    for j, note in enumerate(notes):
        ax.text(0.0, -0.30 - j * 0.13, note, transform=ax.transAxes,
                fontsize=6.6, color="0.35", va="top")

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig, ax


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", type=str,
                    default="results/topo-omni/orientation_unbiased/orientation_stats.npz")
    ap.add_argument("--sheets", type=str,
                    default="results/topo-omni/orientation_unbiased/cortical_sheets.npy")
    ap.add_argument("--coords", type=str,
                    default="src/neighborhoods/model=qwen2-5omni-3b-unified-frontend-20-v4_radius=20_"
                            "neighborhoods=20_coords=42/coords.npy")
    # The other two Topo-Omni training conditions, each an orientation-eval output dir.
    # Omit either one and its bar renders as "n/a" rather than failing.
    ap.add_argument("--stats-task-only", type=str, default=None,
                    help="orientation_stats.npz for the TASK-LOSS-ONLY Topo-Omni run")
    ap.add_argument("--sheets-task-only", type=str, default=None)
    ap.add_argument("--stats-step0", type=str, default=None,
                    help="orientation_stats.npz for the STEP-0 control: pretrained "
                         "backbone, cortical adaptors at init, zero optimisation steps. "
                         "The tightest control for 'what did the spatial loss add'.")
    ap.add_argument("--sheets-step0", type=str, default=None)
    ap.add_argument("--stats-untrained", type=str, default=None,
                    help="orientation_stats.npz for the UNTRAINED Topo-Omni run "
                         "(pretrained backbone, cortical adaptors at init)")
    ap.add_argument("--sheets-untrained", type=str, default=None)
    ap.add_argument("--lambda-px", type=float, default=None,
                    help="force the column spacing (px) instead of estimating it from the "
                         "reference model. Needed to compare ACROSS runs: two checkpoints "
                         "estimate different Lambda (33.2 vs 25.8), and the distance bins "
                         "are fractions of Lambda, so their smoothness values sit on "
                         "different rulers until this is pinned to a common value.")
    ap.add_argument("--per-model-lambda", action="store_true",
                    help="estimate the column spacing separately for each Topo-Omni "
                         "condition instead of pinning them all to the reference model's. "
                         "Off by default: a per-model Lambda measures each bar on a "
                         "different ruler (see topo_omni_curves).")
    ap.add_argument("--bank", type=str, default="unbiased")
    ap.add_argument("--layers", type=str, default="0-3")
    ap.add_argument("--fwhm", type=float, default=4.0)
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--data-dir", type=str, default="data/tdann")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--tdann-macaque-color", type=str,
                    default="data/TDANN_Macaque_V1_Colors.png",
                    help="TDANN's rendered macaque chromatic-preference panel (Fig. 2H), "
                         "plotted as a second colour reference where present.")
    ap.add_argument("--livingstone", type=str, default=str(LIVINGSTONE_IMAGE),
                    help="Livingstone & Hubel 1984 Fig. 27B image, for the panel-I macaque "
                         "reference. Not redistributed by TDANN, so it must be supplied; "
                         "without it the reference falls back to a figure-read constant.")
    ap.add_argument("--num-samples", type=int, default=20)
    args = ap.parse_args()

    outdir = Path(args.outdir) if args.outdir else Path(args.stats).parent / "panels"
    outdir.mkdir(parents=True, exist_ok=True)

    from src.experiments.orientation.run import GRATING_BANKS
    stimuli_root = Path(os.environ["STIMULI_DIR"]) / GRATING_BANKS[args.bank][1]
    paths, ori_deg, counts, meta = parse_stimuli(stimuli_root, color="all")
    ori_keys = np.repeat(ori_deg, counts)
    print(f"> bank '{args.bank}': {len(paths)} gratings")

    tags = list(METRICS)
    per_model: Dict[str, Dict[str, Dict[int, np.ndarray]]] = {
        k: {} for k in tags + [f"{k}_shuffled" for k in tags]}
    x_by_model: Dict[str, Dict[str, np.ndarray]] = {k: {} for k in tags}

    # --- Topo-Omni
    curves, lam = topo_omni_curves(Path(args.stats), Path(args.sheets), args.bank,
                                   args.layers, args.fwhm, args.coords,
                                   num_samples=args.num_samples,
                                   lam_override=args.lambda_px)
    for tag in tags:
        per_model[tag]["Topo-Omni"] = {0: curves[tag]}
        per_model[f"{tag}_shuffled"]["Topo-Omni"] = {0: curves[f"{tag}_shuffled"]}
        edges = tdann_bin_edges(lam)
        x_by_model[tag]["Topo-Omni"] = (0.5 * (edges[:-1] + edges[1:])) / lam
    print(f"> Topo-Omni (layers {args.layers}, fwhm={args.fwhm}): Lambda={lam:.1f} px"
          + (" [FORCED via --lambda-px]" if args.lambda_px else "")
          + ("" if args.per_model_lambda else "  [pinned for every Topo-Omni condition]"))

    # --- the other two Topo-Omni conditions, computed identically so the three bars in the
    # Topo-Omni group differ only by what the model was trained with.
    for label, st, sh in (("Topo-Omni Task Only", args.stats_task_only, args.sheets_task_only),
                          ("Topo-Omni Step 0", args.stats_step0, args.sheets_step0),
                          ("Topo-Omni Untrained", args.stats_untrained, args.sheets_untrained)):
        if not st or not sh:
            print(f"  ! {label}: no --stats/--sheets given, bar will show n/a")
            continue
        if not (Path(st).exists() and Path(sh).exists()):
            print(f"  ! {label}: {st} or {sh} missing, bar will show n/a")
            continue
        c, lam_v = topo_omni_curves(Path(st), Path(sh), args.bank, args.layers,
                                    args.fwhm, args.coords, num_samples=args.num_samples,
                                    lam_override=None if args.per_model_lambda else lam)
        for tag in tags:
            per_model[tag][label] = {0: c[tag]}
            per_model[f"{tag}_shuffled"][label] = {0: c[f"{tag}_shuffled"]}
            e = tdann_bin_edges(lam_v)
            x_by_model[tag][label] = (0.5 * (e[:-1] + e[1:])) / lam_v
        print(f"> {label} (layers {args.layers}, fwhm={args.fwhm}): Lambda={lam_v:.1f} px, "
              + ", ".join(f"{t} {smoothness(np.nanmean(c[t], axis=0)):.3f}" for t in tags))

    # --- TDANN family
    for name, seeds in TDANN_FAMILY.items():
        for seed in sorted(seeds):
            try:
                c = tdann_family_curves(name, seed, paths, meta, ori_keys,
                                        Path(args.data_dir), device=args.device,
                                        num_samples=args.num_samples)
            except Exception as err:                          # noqa: BLE001
                print(f"  ! {name} seed {seed} skipped ({type(err).__name__}: {err})")
                continue
            for tag in tags:
                per_model[tag].setdefault(name, {})[seed] = c[tag]
                per_model[f"{tag}_shuffled"].setdefault(name, {})[seed] = c[f"{tag}_shuffled"]
                edges = tdann_bin_edges(TDANN_COLUMN_SPACING_MM)
                x_by_model[tag][name] = ((0.5 * (edges[:-1] + edges[1:]))
                                         / TDANN_COLUMN_SPACING_MM)
            print(f"  {name} seed {seed}: " + ", ".join(
                f"{tag} {np.nanmean([smoothness(x) for x in c[tag]]):.3f}" for tag in tags))

    # --- macaque references
    mx_o, mac_ori = macaque_delta_curve()
    _, mac_ori_sh = macaque_delta_curve(shuffle=True)
    mx_s, mac_sf = macaque_sf_delta_curve()
    _, mac_sf_sh = macaque_sf_delta_curve(shuffle=True)
    macaque = {
        "orientation": float(np.nanmean([smoothness(c) for c in mac_ori])),
        "sf": float(np.nanmean([smoothness(c) for c in mac_sf])),
    }
    macaque_curves = {"orientation": (mx_o, mac_ori, mac_ori_sh),
                      "sf": (mx_s, mac_sf, mac_sf_sh)}

    colour_is_figure_read = True
    extra_color_refs: List[Tuple[str, float, float]] = []
    if Path(args.livingstone).exists():
        from src.experiments.orientation.macaque_map import livingstone_color_curve
        lv = dict(map_box=LIVINGSTONE_MAP_BOX,
                  scale_bar_px=LIVINGSTONE_SCALE_BAR_PX,
                  scale_bar_len_mm=LIVINGSTONE_SCALE_BAR_MM)
        mx_c, mac_col = livingstone_color_curve(Path(args.livingstone), **lv)
        _, mac_col_sh = livingstone_color_curve(Path(args.livingstone), shuffle=True, **lv)
        macaque["color"] = float(np.nanmean([smoothness(c) for c in mac_col]))
        macaque_curves["color"] = [("Macaque V1 (raw fig, 3.8 mm)",
                                   mx_c, mac_col, mac_col_sh)]

        # TDANN's own rendering of the same Livingstone data, as a second reference. Shown
        # rather than substituted: it agrees on smoothness but its 0.84 mm field of view
        # cannot support the plateau (see tdann_macaque_color_curve).
        if Path(args.tdann_macaque_color).exists():
            from src.experiments.orientation.macaque_map import tdann_macaque_color_curve
            mx_t, mac_t = tdann_macaque_color_curve(Path(args.tdann_macaque_color))
            _, mac_t_sh = tdann_macaque_color_curve(Path(args.tdann_macaque_color),
                                                    shuffle=True)
            # TDANN's rendering of the same Livingstone data is the SINGLE colour
            # reference: it REPLACES the raw-figure (3.8 mm) decode rather than being shown
            # alongside it, so neither the line nor the bar panel carries two V1 lines.
            raw_fig_smoothness = macaque["color"]
            macaque_curves["color"] = [
                ("Macaque V1 (TDANN panel, 0.84 mm)", mx_t, mac_t, mac_t_sh)]
            macaque["color"] = float(np.nanmean([smoothness(c) for c in mac_t]))
            print(f"  colour reference = TDANN's rendered panel: smoothness "
                  f"{macaque['color']:.3f} (the raw-figure 3.8 mm decode, "
                  f"{raw_fig_smoothness:.3f}, is computed but not plotted). Note its 0.84 mm "
                  f"field of view cannot support the plateau statistic.")
        print(f"  colour reference: plateau {np.nanmean(mac_col, axis=0)[-3:].mean():.3f} "
              f"vs its own chance level {np.nanmean(mac_col_sh):.3f} -- a continuous "
              f"blobness value, not a binary category, hence the normalised axis")
        colour_is_figure_read = False
    else:
        macaque["color"] = MACAQUE_COLOR_SMOOTHNESS_FIGURE_READ
        macaque_curves["color"] = None
        print(f"  ! {args.livingstone} absent; colour reference falls back to the "
              f"figure-read {macaque['color']:.2f}")

    print(f"\n> macaque smoothness: orientation {macaque['orientation']:.3f}, "
          f"sf {macaque['sf']:.3f}, colour {macaque['color']:.3f}"
          + (" (figure-read)" if colour_is_figure_read else
             " (decoded from the supplied Livingstone 1984 Fig. 27B)"))

    # --- summarise, plot, test
    all_lines = []
    titles = {"orientation": "Orientation", "sf": "Spatial frequency", "color": "Colour"}
    for tag, (panel, ylabel, _circ, normalise) in METRICS.items():
        summaries = {n: summarise(v) for n, v in per_model[tag].items()}
        lines = run_stats(summaries, macaque[tag], tag)
        all_lines += lines + [""]
        print("\n" + "\n".join(lines))

        sm_refs = [("V1", macaque[tag])] + [(n, v) for n, v, _ in
                                            (extra_color_refs if tag == "color" else [])]
        plot_smoothness_bars(summaries, sm_refs,
                             f"{titles[tag]} map smoothness",
                             approx=(tag == "color" and colour_is_figure_read),
                             savepath=outdir / f"panel{panel}_smoothness_{tag}.png")

        # Chance-normalised plateau. Reported as its own panel because it is a different
        # statistic, not a rescaling of the bars above (smoothness cannot change under
        # rescaling -- see summarise_plateau_ratio).
        ratios = {n: summarise_plateau_ratio(per_model[tag][n],
                                             per_model[f"{tag}_shuffled"][n])
                  for n in per_model[tag]}
        mac_ratio = float("nan")
        if macaque_curves[tag] is not None:
            first = macaque_curves[tag]
            _, mc, mc_sh = first[0][1:] if isinstance(first, list) else first
            mac_ratio = float(np.nanmax(np.nanmean(mc, axis=0)) / np.nanmean(mc_sh))
        rlines = [f"  plateau / chance  (1.0 = fully decorrelated by the largest distance)",
                  f"    macaque V1     {mac_ratio:.3f}"]
        for n, s in ratios.items():
            rlines.append(f"    {n:<22} {s['mean']:.3f} +/- {s['spread']:.3f} (n={s['n']})")
        all_lines += rlines + [""]
        print("\n".join(rlines))
        pl_refs = [("V1", mac_ratio)] + [(n, r) for n, _, r in
                                         (extra_color_refs if tag == "color" else [])]
        plot_smoothness_bars(ratios, pl_refs,
                             f"{titles[tag]}: plateau / chance",
                             approx=False,
                             savepath=outdir / f"panel{panel}_plateau_{tag}.png")
        plot_metric_lines(per_model[tag], per_model[tag + "_shuffled"], x_by_model[tag],
                          macaque_curves[tag], ylabel,
                          f"{titles[tag]}: $\\Delta$ preference vs distance", normalise,
                          savepath=outdir / f"panel{panel}_lines_{tag}.png")
        np.savez(outdir / f"smoothness_{tag}.npz",
                 macaque=macaque[tag],
                 **{f"{n}_mean": s["mean"] for n, s in summaries.items()},
                 **{f"{n}_spread": s["spread"] for n, s in summaries.items()},
                 **{f"{n}_n": s["n"] for n, s in summaries.items()})

    (outdir / "smoothness_stats.txt").write_text("\n".join(all_lines))
    print(f"\n> Saved bars, arrays and stats to: {outdir}")


if __name__ == "__main__":
    main()
