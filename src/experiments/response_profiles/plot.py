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

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate response profiles for different conditions.")
    parser.add_argument("--model_name", type=str, default="topo-omni", help="Model name to evaluate")
    parser.add_argument("--localizer", type=str, default="faces", help="Localizer condition to evaluate (e.g., faces, bodies, scenes)")
    parser.add_argument("--top_k_pct", type=int, default=1, help="Top k percent of units to keep")
    parser.add_argument("--fwhm_mm", type=float, default=4.0, help="FWHM in mm for smoothing the cortical sheet")
    parser.add_argument("--anatomical_constraint", type=str, default="true", choices=["true", "false"], help="Whether to apply anatomical constraints for selectivity mask")
    parser.add_argument("--significant_q", type=float, default=None,
                        help="read the profiles produced with this significance gate "
                             "(names the _sig= suffix on the pickles and the figure).")

    args = parser.parse_args()
    localizer = args.localizer

    top_k = args.top_k_pct  # top k percent of units to keep
    anatomical_constraint = args.anatomical_constraint == "true"
    fwhm_mm = args.fwhm_mm

    model_name = args.model_name
    save_dir = f"{SAVE_DIR}/{model_name}/response_profiles"

    sig_tag = f"_sig={args.significant_q}" if args.significant_q else ""

    results_path = f"{save_dir}/{localizer}_response_profiles_top{top_k}_even_fwhm_mm={fwhm_mm}_anat={anatomical_constraint}{sig_tag}.pkl"

    # The legacy fallback drops fwhm_mm AND anat AND the significance gate. For a gated run
    # that would silently load the UNGATED profiles under the gated figure name, so the
    # fallback is only allowed when no gate was requested.
    if not os.path.exists(results_path) and not sig_tag:
        print(f">> Falling back to old path without fwhm_mm and anatomical_constraint: {results_path}")
        results_path = f"{save_dir}/{localizer}_response_profiles_top{top_k}_even.pkl"

    results_even = read_pickle(results_path)
    df_even = pd.DataFrame(results_even)
    df_even["odd_or_even"] = "even"

    results_path = f"{save_dir}/{localizer}_response_profiles_top{top_k}_odd_fwhm_mm={fwhm_mm}_anat={anatomical_constraint}{sig_tag}.pkl"
    if not os.path.exists(results_path) and not sig_tag:
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
        # raw ROWS, and the splits do not contribute equally many: the clips are
        # partitioned by run-number parity, which here gives 6 odd vs 4 even, so the
        # even-localized direction silently carried 60%% of the weight. On the faces
        # localizer that moved quilted_speech from +0.188 to +0.129 and false_photo from
        # +0.085 to +0.123 -- enough to change how the non-visual block reads.
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
    # Bar heights must use the SAME equal-weight rule as the per-unit CV profile above.
    # Feeding seaborn the per-stimulus rows would re-introduce the 6:4 imbalance, since it
    # would average 6 even-split points against 4 odd-split ones. So: collapse to a mean per
    # (condition, stimulus) within each split, take each split's mean, then average the two
    # splits equally. The error bar is the SE of that equal-weight mean,
    # sqrt( (se_even^2 + se_odd^2) / 4 ), which propagates both splits' stimulus-level noise.
    _stim = (df.groupby(["condition", "stimuli_idx", "odd_or_even"], as_index=False)
               ["unit_response"].mean())
    _g = _stim.groupby(["condition", "odd_or_even"])["unit_response"]
    _split_mean = _g.mean().unstack("odd_or_even")
    _split_se = (_g.std(ddof=1) / _g.count().pow(0.5)).unstack("odd_or_even")

    condition_order = list(map_conditions.values())
    heights = _split_mean.mean(axis=1).reindex(condition_order)          # equal weight
    errs = (_split_se.pow(2).sum(axis=1) / 4.0).pow(0.5).reindex(condition_order)

    baseline = model_profile.min() - 0.1 * (model_profile.max() - model_profile.min())

    sns.set_theme(context="paper", font_scale=2, style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.bar(
        range(len(condition_order)),
        (heights - baseline).values,
        yerr=errs.values,
        color=[palette[c] for c in condition_order],
        edgecolor="black",
        linewidth=1,
        error_kw=dict(ecolor="0.25", capsize=0, lw=2),
    )
    ax.set_xlim(-0.6, len(condition_order) - 0.4)

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
    # fwhm_mm and anatomical_constraint are INPUTS but were absent from the output name,
    # so plotting the same localizer at two FWHM values silently overwrote the first figure.
    # They name the input pickles; they belong in the output too.
    _out = (f"{save_dir}/{localizer}_response_profiles_top{top_k}"
            f"_fwhm_mm={fwhm_mm}_anat={anatomical_constraint}{sig_tag}.png")
    plt.savefig(_out, dpi=300)
    print(f"> saved {_out}")