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

REPO_DIR = os.getenv("REPO_DIR")
CKPT_DIR = os.getenv("CKPT_DIR")
SAVE_DIR = os.getenv("SAVE_DIR")
STIMULI_DIR = os.getenv("STIMULI_DIR")

def read_pickle(filepath):  
    with open(filepath, "rb") as f:
        data = pkl.load(f)
    return data

map_conditions = {
    "faces": "Fa",
    "bodies": "B",
    "scenes": "S",
    "objects": "O",
    "words_scr_objects": "W",
    "false_belief": "FB",
    "false_photo": "FP",
    "nonwords": "NW",
    "quilted_speech": "QLT",
    "math": "MATH",
}

localizer_to_target = {
    "vwfa": "W",
    "speech": "QLT",
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

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate response profiles for different conditions.")
    parser.add_argument("--model_name", type=str, default="topo-omni", help="Model name to evaluate")
    parser.add_argument("--localizer", type=str, default="faces", help="Localizer condition to evaluate (e.g., faces, bodies, scenes)")
    parser.add_argument("--top_k_pct", type=int, default=1, help="Top k percent of units to keep")
    parser.add_argument("--fwhm_mm", type=float, default=4.0, help="FWHM in mm for smoothing the cortical sheet")
    parser.add_argument("--anatomical_constraint", type=str, default="true", choices=["true", "false"], help="Whether to apply anatomical constraints for selectivity mask")

    args = parser.parse_args()
    localizer = args.localizer

    top_k = args.top_k_pct  # top k percent of units to keep
    anatomical_constraint = args.anatomical_constraint == "true"
    fwhm_mm = args.fwhm_mm

    model_name = args.model_name
    save_dir = f"{SAVE_DIR}/{model_name}/response_profiles"

    results_path = f"{save_dir}/{localizer}_response_profiles_top{top_k}_even_fwhm_mm={fwhm_mm}_anat={anatomical_constraint}.pkl"

    if not os.path.exists(results_path):
        print(f">> Falling back to old path without fwhm_mm and anatomical_constraint: {results_path}")
        results_path = f"{save_dir}/{localizer}_response_profiles_top{top_k}_even.pkl"

    results_even = read_pickle(results_path)
    df_even = pd.DataFrame(results_even)
    df_even["odd_or_even"] = "even"

    results_path = f"{save_dir}/{localizer}_response_profiles_top{top_k}_odd_fwhm_mm={fwhm_mm}_anat={anatomical_constraint}.pkl"
    if not os.path.exists(results_path):
        print(f">> Falling back to old path without fwhm_mm and anatomical_constraint: {results_path}")
        results_path = f"{save_dir}/{localizer}_response_profiles_top{top_k}_odd.pkl"

    results_odd = read_pickle(results_path)
    df_odd = pd.DataFrame(results_odd)
    df_odd["odd_or_even"] = "odd"

    df = pd.concat([df_even, df_odd], ignore_index=True)

    df["condition"] = df["condition"].map(map_conditions)

    # ---------------------------------------------------------------
    # Statistical tests
    # ---------------------------------------------------------------
    target_cond = localizer_to_target.get(localizer, map_conditions.get(localizer, None))
    condition_order = list(map_conditions.values())

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

    # ---- (2) Model <-> human FFA response-profile correspondence ----
    def profile_correspondence(model_profile, human_profile, method="spearman",
                            n_perm=10000, seed=0):
        a = np.asarray(model_profile, float)
        b = np.asarray(human_profile, float)
        rng = np.random.default_rng(seed)
        corr = (lambda x, y: stats.spearmanr(x, y)[0]) if method == "spearman" \
            else (lambda x, y: stats.pearsonr(x, y)[0])
        r_obs = corr(a, b)
        null = np.array([corr(rng.permutation(a), b) for _ in range(n_perm)])   # shuffle category labels
        p = (np.sum(np.abs(null) >= np.abs(r_obs)) + 1) / (n_perm + 1)
        n = len(a)
        boot = np.array([corr(a[i := rng.integers(0, n, n)], b[i]) for _ in range(n_perm)])
        ci = np.nanpercentile(boot, [2.5, 97.5])
        return r_obs, p, ci

    # ---- Human FFA profile (from the fMRI ROI CSV) -----------------------------
    CSV_PATH  = os.path.join(REPO_DIR, "data", "condition_responses_summary_20260310_203831.csv")  # path to the human ROI-response csv
    
    roi_list = {
        "faces": ('lh_ffa', 'FFA'), 
        "bodies": ('lh_eba', 'EBA'), 
        "objects": ('lh_loc', 'LOC'),
        "scenes": ('lh_ppa', 'PPA'), 
        "vwfa": ('lh_vwfa', 'VWFA'), 
        "speech": ('speech', 'Speech'),
        "theory_of_mind": ('lh_tpj', 'rTPJ'),
        "opa": ('lh_opa', 'OPA'), 
        "rsc": ('lh_rsc', 'RSC'),
    }
    
    ROIS = [roi_list[localizer][0]]

    roi_data = pd.read_csv(CSV_PATH)
    roi_data = roi_data[roi_data["roi_label"].isin(ROIS)].copy()
    roi_data["condition"] = roi_data["condition"].map(map_conditions)      # full name -> "Fa","B",...

    # per-subject roi_data profile (averaged over hemispheres if >1), ordered like the model
    # roi_data_subj = (roi_data.groupby(["condition"])["mean_beta_mean"].mean()
    #             .reindex(columns=condition_order))

    human_profile = roi_data.groupby(["condition"])["mean_beta_mean"].mean().reindex(condition_order)  # group-mean profile

    # headline correspondence: group-mean model vs group-mean FFA
    r, p, ci = profile_correspondence(human_profile, model_profile, method="spearman")
    print(f"[model-ROI] group Spearman rho={r:.3f}, p={p:.4f}, 95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")

    r, p, ci = profile_correspondence(human_profile, model_profile, method="pearson")
    print(f"[model-ROI] group Pearson ={r:.3f}, p={p:.4f}, 95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")
    
    # make the most negative value at zero to better visualize the differences between conditions
    mean_df =  df.groupby(["condition", "stimuli_idx", "odd_or_even"]).mean("unit_response").reset_index()

    # baseline = mean_df["unit_response"].min()
    baseline = model_profile.min() - 0.1 * (model_profile.max() - model_profile.min())  # slightly above the minimum for better visualization

    mean_df["unit_response"] = mean_df["unit_response"] - baseline
    
    sns.set_theme(context="paper", font_scale=2, style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 4))

    sns.barplot(
        x="condition", 
        y="unit_response", 
        data=mean_df, 
        order=list(map_conditions.values()),
        palette=palette,
        errorbar="se",
        edgecolor="black",
        linewidth=1,
        ax=ax
    )

    # remove xticks
    ax.set_xticks([])

    # make yticks less frequent
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))

    sns.despine()
    ax.axvline(4.5, color="lightgray", linewidth=2)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y + baseline:.2f}"))

    plt.xlabel("")
    plt.ylabel("Mean Activation", fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{localizer}_response_profiles_top{top_k}.png", dpi=300)