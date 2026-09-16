import os
import numpy as np
import pandas as pd
import pickle as pkl

import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from matplotlib.ticker import FuncFormatter

from dotenv import load_dotenv
load_dotenv()

CKPT_DIR = os.getenv("CKPT_DIR")
SAVE_DIR = os.getenv("SAVE_DIR")
STIMULI_DIR = os.getenv("STIMULI_DIR")

def read_pickle(filepath):  
    with open(filepath, "rb") as f:
        data = pkl.load(f)
    return data

map_conditions = {
    "false_belief": "FB",
    "false_photo": "FP",
    "nonwords": "NW",
    "math": "MATH",
}

# -----------------------------
# Example dummy data
# -----------------------------
conditions_left = ["Fa", "B", "S", "O", "W"]
conditions_right = ["FB", "FP", "NW", "QLT", "MATH"]
conditions = conditions_left + conditions_right

conditions_labels = [
    "Faces",
    "Bodies",
    "Scenes",
    "Objects",
    "Words",
    "False Belief",
    "False Photo",
    "Nonwords",
    "Quilted Speech",
    "Math"
]

# -----------------------------
# Color palettes
# -----------------------------
palette_left = sns.color_palette(
    # The 5 VISUAL conditions. Identical to the vision family in
    # selectivity/plot.py, so faces/bodies/scenes/objects/words carry the same colour
    # whether they appear as a localizer on the map or as a condition here.
    # Scenes/Objects/Words were #F1C232/#EFC94C/#D4AC0D -- three golds, with
    # Scenes vs Objects at dE2000 = 3.0 (indistinguishable).
    ["#E64B35", "#F39B2F", "#F1C232", "#A0522D", "#9C6B00"]
)

palette_right = sns.color_palette(
    # The 5 NON-VISUAL conditions, re-spaced across teal -> blue -> violet. FB/FP were
    # dE2000 = 6.4 apart and FP/NW 8.7 -- the same collision the map had.
    #
    # NOT a 1:1 copy of the map's cool colours, deliberately. There the slots hold
    # unrelated categories from different bands (slot 5 = theory_of_mind, slot 6 =
    # vocals), so they were split into a purple text family and a teal audio family.
    # Here slots 5 and 6 are False Belief and False Photo -- two ToM conditions that
    # blend_colors() AVERAGES into the single "FB+FP" bar. Purple + teal averages to
    # #6d7fb5, a desaturated mud. Keeping FB/FP as two distinguishable teals blends to
    # #40a08c instead, and keeps the whole right block cool so the visual /
    # non-visual divider still reads.
    ["#66C2A5", "#1B7F72", "#4A9BD1", "#2A4B8D", "#7B4FA8"]
)

palette = dict(zip(conditions, palette_left + palette_right))


def blend_colors(conditions):
    if conditions == "FB+FP":
        conditions = ["FB", "FP"]
    else:
        conditions = [conditions]

    colors = [palette[c] for c in conditions if c in palette]
    return tuple(np.mean(colors, axis=0)) if colors else (0.5, 0.5, 0.5)

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate response profiles for different conditions.")
    parser.add_argument("--localizer", type=str, default="faces", help="Localizer condition to evaluate (e.g., faces, bodies, scenes)")
    parser.add_argument("--top_k_pct", type=int, default=10, help="Top k percent of units to keep")
    parser.add_argument("--model_name", type=str, default="topo-omni",
                        help="run title under $SAVE_DIR holding response_profiles/ (was hardcoded)")
    parser.add_argument("--significant_q", type=float, default=None,
                        help="read the profiles produced with this significance gate "
                             "(names the _sig= suffix on the pickles and the figure).")

    args = parser.parse_args()
    localizer = args.localizer

    print(f"> Plotting response profiles for {localizer} localizer, top_k_pct={args.top_k_pct}...")

    top_k = args.top_k_pct  # top k percent of units to keep

    model_name = args.model_name
    save_dir = f"{SAVE_DIR}/{model_name}/response_profiles"

    sig_tag = f"_sig={args.significant_q}" if args.significant_q else ""

    results_path = f"{save_dir}/{localizer}_response_profiles_top{top_k}_even{sig_tag}.pkl"
    results_even = read_pickle(results_path)
    df_even = pd.DataFrame(results_even)
    df_even["odd_or_even"] = "even"

    results_path = f"{save_dir}/{localizer}_response_profiles_top{top_k}_odd{sig_tag}.pkl"
    results_odd = read_pickle(results_path)
    df_odd = pd.DataFrame(results_odd)
    df_odd["odd_or_even"] = "odd"

    df = pd.concat([df_even, df_odd], ignore_index=True)

    localizer_condition_map = {
        "language_text": {"ON": "FB+FP", "OFF": "NW"},
        "theory_of_mind_text": {"ON": "FB", "OFF": "FP"},
        "multiple_demand_text": {"ON": "MATH", "OFF": "FB+FP"},
    }

    df["condition"] = df["condition"].map(localizer_condition_map[localizer])
    target_cond = localizer_condition_map[localizer]["ON"]
    
    # Statistical tests
    # ---------------------------------------------------------------
    condition_order = localizer_condition_map[localizer].values()

    # locate the per-unit / per-cluster id column (anything that isn't value/label/split)
    meta_cols = {"condition", "unit_response", "odd_or_even"}
    unit_candidates = [c for c in df.columns if c not in meta_cols]
    print("df columns:", list(df.columns), "| unit id candidate(s):", unit_candidates)

    # Keep only units measured in BOTH halves. unit ids are sheet positions and the two
    # halves localize different masks, so their unit sets overlap without being equal.
    # A unit present in one half only is not cross-validated at all: averaging it in mixes
    # CV'd with single-split measurements, and when the two masks differ in size it leaves
    # the paired test comparing rows that have no counterpart. Restricting here keeps the
    # bar heights and the statistic describing the same population.
    if unit_candidates:
        _uc = unit_candidates[0]
        _n_all = df[_uc].nunique()
        _both = df.groupby(_uc)["odd_or_even"].nunique() == 2
        _keep = _both[_both].index
        df = df[df[_uc].isin(_keep)].copy()
        print(f"[cv] units in both halves: {len(_keep)} / {_n_all} "
              f"({100 * len(_keep) / max(_n_all, 1):.1f}%)")

    # ---- (1) selectivity: target vs. non-target categories ----
    if unit_candidates:
        unit_col = unit_candidates[0]
        # Cross-validated profile per unit: average WITHIN each split first, then across
        # the two splits, so each direction counts once.
        #
        # Grouping the concatenated frame in one step (the previous behaviour) averages
        # raw ROWS, and the splits do not contribute equally many: the text sets are
        # partitioned by run-number parity, which gives 6 even vs 4 odd stimuli per
        # condition (12 vs 8 for the language ON set), so the even-localized direction
        # silently carried 60% of the weight. Same rule as response_profiles/plot.py.
        _per_split = df.groupby([unit_col, "condition", "odd_or_even"],
                                as_index=False)["unit_response"].mean()
        cv = _per_split.groupby([unit_col, "condition"], as_index=False)["unit_response"].mean()
        wide = cv.pivot(index=unit_col, columns="condition", values="unit_response")
        target = wide[target_cond].values
        others = wide.drop(columns=[target_cond]).mean(axis=1).values   # each unit's mean non-target

        # paired across units (units contribute both target and non-target) -- primary
        t_pair, p_pair = stats.ttest_rel(target, others)
        # Welch on the two pools (matches your cluster-scoring convention) -- alternative
        t_welch, p_welch = stats.ttest_ind(target, others, equal_var=False)
        dprime = (target.mean() - others.mean()) / np.sqrt((target.var(ddof=1) + others.var(ddof=1)) / 2)

        print(f"[selectivity] paired t={t_pair:.2f}, p={p_pair:.2e} | "
            f"Welch t={t_welch:.2f}, p={p_welch:.2e} | d'={dprime:.2f} (n={len(target)} units)")
        model_profile = wide.mean(axis=0).reindex(condition_order).values
    else:
        print("No unit id column found; using pooled Welch t-test.")
        tv = df.loc[df.condition == target_cond, "unit_response"].values
        ov = df.loc[df.condition != target_cond, "unit_response"].values
        t_welch, p_welch = stats.ttest_ind(tv, ov, equal_var=False)
        dprime = (tv.mean() - ov.mean()) / np.sqrt((tv.var(ddof=1) + ov.var(ddof=1)) / 2)
        print(f"[selectivity] Welch t={t_welch:.2f}, p={p_welch:.2e} | d'={dprime:.2f}")
        model_profile = df.groupby("condition")["unit_response"].mean().reindex(condition_order).values

    
    condition_values = list(localizer_condition_map[localizer].values())

    
    # df["unit_response"] = df["unit_response"].abs()

    # make the most negative value at zero to better visualize the differences between conditions
    # baseline = df.groupby("condition").mean("unit_response")["unit_response"].min()
    
    # Equal-weight bar heights, matching the per-unit CV rule above. Feeding seaborn the
    # per-stimulus rows re-introduces the split imbalance, since it would average 6 even
    # points against 4 odd ones (12 vs 8 for the language ON set). So: collapse to a mean
    # per (condition, stimulus) within each split, take each split mean, then average the
    # two splits equally. The error bar is the SE of that equal-weight mean,
    # sqrt( (se_even^2 + se_odd^2) / 4 ), propagating both splits stimulus-level noise.
    # Same construction as response_profiles/plot.py.
    _stim = (df.groupby(["condition", "stimuli_idx", "odd_or_even"], as_index=False)
               ["unit_response"].mean())
    _g = _stim.groupby(["condition", "odd_or_even"])["unit_response"]
    _split_mean = _g.mean().unstack("odd_or_even")
    _split_se = (_g.std(ddof=1) / _g.count().pow(0.5)).unstack("odd_or_even")

    heights = _split_mean.mean(axis=1).reindex(condition_values)
    errs = (_split_se.pow(2).sum(axis=1) / 4.0).pow(0.5).reindex(condition_values)

    baseline = model_profile.min() - 0.1 * (model_profile.max() - model_profile.min())  # slightly above the minimum for better visualization

    sns.set_theme(context="paper", font_scale=2, style="whitegrid")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor("white")

    ax.bar(
        range(len(condition_values)),
        (heights - baseline).values,
        yerr=errs.values,
        color=[blend_colors(c) for c in condition_values],
        edgecolor="black",
        linewidth=1,
        error_kw=dict(ecolor="0.25", capsize=0, lw=2),
    )
    ax.set_xticks(range(len(condition_values)))
    ax.set_xticklabels(condition_values)

    # make yticks less frequent
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))

    sns.despine()
    if len(condition_values) > 5:
        ax.axvline(4.5, color="lightgray", linewidth=2)
    ax.set_xlim(-0.5, len(condition_values) - 0.5)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y + baseline:.2f}"))

    # plt.ylim(0.03)
    # plt.title(f"Response Profiles for {localizer.capitalize()} Localizer")
    plt.xlabel("")
    plt.ylabel("Mean Activation", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{localizer}_response_profiles_top{top_k}{sig_tag}.png", dpi=300)
