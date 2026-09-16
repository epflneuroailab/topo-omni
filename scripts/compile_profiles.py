"""Brain vs Topo-Omni vs non-topo response-profile grid, one row per localizer.

Model profiles are CROSS-VALIDATED and EQUAL-WEIGHTED: within each split, average over
stimuli per unit; then average the two splits, each counting once. Averaging the raw
concatenated rows instead would weight the splits by their clip counts (6 vs 4 here).

rho is Spearman between the model's 10-condition profile and the ROI's, with a permutation
p from shuffling the condition labels of the model profile (n=10000).
"""
import os, argparse, pickle
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from scipy import stats

from src.experiments.response_profiles.plot import map_conditions, palette, conditions_labels

ap = argparse.ArgumentParser()
ap.add_argument("--topo-dir", default="results/topo-omni-tiled32-4608-snap438/response_profiles")
ap.add_argument("--nontopo-dir",
                default="../topo-omni-bkhmsi/results/qwen2_5_3b_task_7/response_profiles")
ap.add_argument("--brain-csv", default="data/condition_responses_summary_20260310_203831.csv")
ap.add_argument("--topo-fwhm", default="4.0")
ap.add_argument("--nontopo-fwhm", default="4.0")
ap.add_argument("--top-k", type=int, default=1)
ap.add_argument("--n-perm", type=int, default=10000)
ap.add_argument("--corr", default="pearson", choices=["pearson", "spearman"])
ap.add_argument("--eps", type=float, default=0.1,
                help="baseline is min - eps*(max-min); bars are drawn from there so the "
                     "axis floor sits just below the smallest bar. Tick labels still show "
                     "the true values, and the correlation uses unshifted data.")
ap.add_argument("--out", default="results/topo-omni-tiled32-4608-snap438/response_profiles/profile_grid.png")
a = ap.parse_args()

ORDER = list(map_conditions.values())
ROWS = [("faces", "lh_ffa", "FFA"), ("scenes", "lh_ppa", "PPA"), ("objects", "lh_loc", "LOC"),
        ("vwfa", "lh_vwfa", "VWFA"), ("bodies", "lh_eba", "EBA"), ("speech", "speech", "STG")]
ROWLAB = {"faces": "Faces", "scenes": "Scenes", "objects": "Objects",
          "vwfa": "VWFA", "bodies": "Bodies", "speech": "Speech"}

def model_profile(dirpath, loc, fwhm):
    """Equal-weight CV profile + SE of that equal-weight mean, per condition."""
    per_split_unit, per_split_stim = [], []
    for s in ("even", "odd"):
        f = f"{dirpath}/{loc}_response_profiles_top{a.top_k}_{s}_fwhm_mm={fwhm}_anat=True.pkl"
        if not os.path.exists(f):
            return None, None, f
        x = pd.DataFrame(pickle.load(open(f, "rb")))
        x["condition"] = x["condition"].map(map_conditions)
        per_split_unit.append(x.groupby(["unit_idx", "condition"])["unit_response"].mean())
        st = x.groupby(["condition", "stimuli_idx"])["unit_response"].mean().groupby("condition")
        per_split_stim.append((st.mean(), st.std(ddof=1) / np.sqrt(st.count())))
    cv = (per_split_unit[0] + per_split_unit[1]) / 2.0          # each split counts once
    prof = cv.groupby("condition").mean().reindex(ORDER)
    se = np.sqrt((per_split_stim[0][1].reindex(ORDER) ** 2 +
                  per_split_stim[1][1].reindex(ORDER) ** 2) / 4.0)
    return prof.values, se.values, None

def brain_profile(csv, roi):
    d = pd.read_csv(csv)
    d = d[d.roi_label == roi].copy()
    d["condition"] = d["condition"].map(map_conditions)
    d = d.set_index("condition").reindex(ORDER)
    return d["mean_beta_mean"].values, d["mean_beta_sem"].values

def corr_perm(model, human, n_perm):
    """Correlation between the two 10-condition profiles, with a label-permutation p.

    Both Pearson and Spearman are invariant to the per-panel baseline shift used for
    display, so these are computed on the raw profiles either way.
    """
    m, h = np.asarray(model, float), np.asarray(human, float)
    ok = np.isfinite(m) & np.isfinite(h)
    m, h = m[ok], h[ok]
    _f = (lambda x, y: stats.pearsonr(x, y)[0]) if a.corr == "pearson" \
        else (lambda x, y: stats.spearmanr(x, y)[0])
    r = _f(m, h)
    g = np.random.default_rng(0)
    null = np.array([_f(g.permutation(m), h) for _ in range(n_perm)])
    p = (np.sum(np.abs(null) >= abs(r)) + 1) / (n_perm + 1)
    return r, p

def stars(p):
    if p < 0.001: return r"$p<0.001^{***}$"
    if p < 0.01:  return r"$p<0.01^{**}$"
    if p < 0.05:  return r"$p<0.05^{*}$"
    return rf"$p={p:.2f}^{{n.s.}}$"

fig, axes = plt.subplots(len(ROWS), 3, figsize=(11.0, 2.05 * len(ROWS)))
COLS = [("Brain Data", "#6E8B3D"), ("Topo-Omni", "#4169C8"), ("Qwen2.5-3B (SFT; Non-Topo)", "#4169C8")]

for r, (loc, roi, roiname) in enumerate(ROWS):
    hp, hse = brain_profile(a.brain_csv, roi)
    tp, tse, terr = model_profile(a.topo_dir, loc, a.topo_fwhm)
    npf, nse, nerr = model_profile(a.nontopo_dir, loc, a.nontopo_fwhm)
    for c, (prof, se, title) in enumerate(
            [(hp, hse, roiname), (tp, tse, None), (npf, nse, None)]):
        ax = axes[r, c]
        for sp in ax.spines.values():
            sp.set_edgecolor(COLS[c][1]); sp.set_linewidth(1.4)
        if prof is None:
            ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes,
                    fontsize=8, color="0.5")
            ax.set_xticks([]); ax.set_yticks([]); continue
        # Draw from a baseline just under the smallest bar, so differences between
        # conditions are visible even when the profile sits far from zero. The shift is
        # undone in the tick labels, so the axis still reads true values.
        _lo, _hi = np.nanmin(prof), np.nanmax(prof)
        _rng = _hi - _lo
        base = _lo - (a.eps * _rng if _rng > 0 else max(abs(_lo) * a.eps, 1e-6))
        ax.bar(range(len(ORDER)), prof - base, yerr=se, color=[palette[k] for k in ORDER],
               edgecolor="black", linewidth=0.5, error_kw=dict(ecolor="0.35", lw=1, capsize=0))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v + base:.2f}"))
        if _lo < 0 < _hi:
            ax.axhline(-base, color="0.6", lw=0.6)      # true zero, in shifted coords
        ax.set_xticks([])
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.tick_params(labelsize=7)
        if c == 0:
            ax.set_title(roiname, fontsize=10)
            ax.set_ylabel("Beta", fontsize=8, fontweight="bold")
        else:
            rho, p = corr_perm(prof, hp, a.n_perm)
            ax.set_title(rf"$\rho={rho:.2f}$, {stars(p)}", fontsize=10)
            ax.set_ylabel("Mean Activation", fontsize=7, fontweight="bold")
            print(f"{loc:8s} {'topo' if c == 1 else 'nontopo':8s} rho={rho:+.3f} p={p:.4f}")
    axes[r, 0].text(-0.42, 0.5, ROWLAB[loc], transform=axes[r, 0].transAxes, rotation=90,
                    ha="center", va="center", fontsize=11, fontweight="bold")

for c, (name, _) in enumerate(COLS):
    axes[0, c].annotate(name, xy=(0.5, 1.38), xycoords="axes fraction",
                        ha="center", fontsize=11, fontweight="bold")

handles = [Rectangle((0, 0), 1, 1, facecolor=palette[k], edgecolor="black", linewidth=0.5)
           for k in ORDER]
fig.legend(handles, conditions_labels, loc="lower center", ncol=10, frameon=True,
           fontsize=7.5, bbox_to_anchor=(0.5, -0.012))
fig.tight_layout(rect=[0.02, 0.035, 1, 0.96])
os.makedirs(os.path.dirname(a.out), exist_ok=True)
fig.savefig(a.out, dpi=300, bbox_inches="tight")
print("saved", a.out)
