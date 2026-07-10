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
    ["#E64B35", "#F39B2F", "#F1C232", "#EFC94C", "#D4AC0D"]
)

palette_right = sns.color_palette(
    ["#5AB4AC", "#4C9F9B", "#3F8F9C", "#3B5B92", "#2F3E75"]
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

    args = parser.parse_args()
    localizer = args.localizer

    print(f"> Plotting response profiles for {localizer} localizer, top_k_pct={args.top_k_pct}...")

    top_k = args.top_k_pct  # top k percent of units to keep

    model_name = "topo-omni"
    save_dir = f"{SAVE_DIR}/{model_name}/response_profiles"

    results_path = f"{save_dir}/{localizer}_response_profiles_top{top_k}_even.pkl"
    results_even = read_pickle(results_path)
    df_even = pd.DataFrame(results_even)
    df_even["odd_or_even"] = "even"

    results_path = f"{save_dir}/{localizer}_response_profiles_top{top_k}_odd.pkl"
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

    # ---- (1) selectivity: target vs. non-target categories ----
    if unit_candidates:
        unit_col = unit_candidates[0]
        # cross-validated profile per unit (average over even/odd splits)
        cv = df.groupby([unit_col, "condition"], as_index=False)["unit_response"].mean()
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
    
    mean_df =  df.groupby(["condition", "stimuli_idx", "odd_or_even"]).mean("unit_response").reset_index()

    baseline = model_profile.min() - 0.1 * (model_profile.max() - model_profile.min())  # slightly above the minimum for better visualization

    mean_df["unit_response"] = mean_df["unit_response"] - baseline
    
    sns.set_theme(context="paper", font_scale=2, style="whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

    
    # fig_single, ax = plt.subplots(figsize=(4, 4))
    # fig_single.patch.set_facecolor('white')
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor('white')

    # light gray background
    # ax.set_facecolor("#ECECEC")

    sns.barplot(
        hue="condition", 
        y="unit_response", 
        data=mean_df, 
        order=condition_values,
        palette=[blend_colors(c) for c in condition_values],
        errorbar="se",
        edgecolor="black",
        linewidth=1,
        ax=ax,
        legend=False
    )

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
    plt.savefig(f"{save_dir}/{localizer}_response_profiles_top{top_k}.png", dpi=300)