#!/usr/bin/env python3
"""Cross-validated response extraction + 2-bar profile (Fig. 3b, step cv_05).

For each subject, measure the mean beta inside the *other* fold's fROI mask, so the
mask and the responses never share data:
  * fold-B vocal/non-vocal betas averaged within fold-A's mask;
  * fold-A vocal/non-vocal betas averaged within fold-B's mask;
  * per-condition response = mean of the two cross-validated estimates.

Then draw the group 2-bar plot (vocal vs non-vocal, mean ± SEM), non-vocal shifted to
the zero baseline (same convention as Marvi Fig. 4). Produces the paper's Fig. 3b
2-bar profile and the ``cv_responses.csv`` golden fixture.

Lineage (docs/DESIGN.md §2.4):  cv_04 group fROI -> **cv_05 responses + plot**.
  input : <results-root>/04_cross_validation/group/half-{A,B}_fROI_mask.nii.gz
          + .../per_subject/sub*/half-{A,B}_{vocal,nonvocal}_beta.nii.gz  (218)
  output: <results-root>/04_cross_validation/cv_responses.csv + cv_bar_plot.{svg,png}

PORT NOTES vs dev-repo `src/cv_05_extract_responses_and_plot.py` (@ f842b1a):
  * Faithful port; parameterized by `--results-root` (was hard-coded `results/...`).
  * `extract_responses()` is side-effect-free so the golden master can assert on the
    returned frame without touching disk. The plotting/CSV writing move to `main()`.

DETERMINISM (docs/DESIGN.md §2.2/§6): extraction is **pure nibabel + numpy** (mean-in-mask) —
NO nilearn — so it is bitwise version-robust. It reproduces the published
``cv_responses.csv`` to ~1 ULP (max|Δ| ≈ 8e-17), and its golden master is asserted at
`atol=1e-12`. (Contrast with cv_04's `SecondLevelModel`, which is version-pinned.)
"""
from __future__ import annotations

import argparse
from pathlib import Path

# --- Layout constants (match the precomputed CV cut) ---
N_SUBJECTS = 218
SUBJECT_TEMPLATE = "sub{:03d}_Ed"
CV_SUBDIR = "04_cross_validation"
PER_SUBJECT_SUBDIR = "per_subject"
GROUP_SUBDIR = "group"


def subject_ids(n_subjects=N_SUBJECTS):
    return [SUBJECT_TEMPLATE.format(i) for i in range(1, n_subjects + 1)]


def extract_mean_in_mask(beta_path: Path, mask_bool) -> float:
    """Mean beta within the boolean mask (NaN-aware), matching the dev extractor."""
    import nibabel as nib  # lazy
    import numpy as np

    data = nib.load(str(beta_path)).get_fdata()
    return float(np.nanmean(data[mask_bool]))


def extract_responses(results_root: Path, n_subjects=N_SUBJECTS):
    """Cross-validated per-subject responses. Returns a DataFrame.

    Columns: ``subject``, ``vocal_beta``, ``nonvocal_beta``. A subject is included iff
    all four half-split beta maps are present (mirrors the dev loader).
    """
    import nibabel as nib  # lazy
    import pandas as pd

    cv_dir = Path(results_root) / CV_SUBDIR
    group_dir = cv_dir / GROUP_SUBDIR
    per_subject_dir = cv_dir / PER_SUBJECT_SUBDIR

    mask_A = nib.load(str(group_dir / "half-A_fROI_mask.nii.gz")).get_fdata().astype(bool)
    mask_B = nib.load(str(group_dir / "half-B_fROI_mask.nii.gz")).get_fdata().astype(bool)
    if mask_A.sum() == 0 or mask_B.sum() == 0:
        raise RuntimeError("One or both fROI masks are empty — re-check cv_04.")

    rows = []
    for sid in subject_ids(n_subjects):
        subj = per_subject_dir / sid
        needed = [subj / f"half-{f}_{m}_beta.nii.gz"
                  for f in ("A", "B") for m in ("vocal", "nonvocal")]
        if not all(p.exists() for p in needed):
            continue  # subject not yet processed

        # Fold-B responses measured within mask A; fold-A responses within mask B.
        vocal_B = extract_mean_in_mask(subj / "half-B_vocal_beta.nii.gz", mask_A)
        nonvocal_B = extract_mean_in_mask(subj / "half-B_nonvocal_beta.nii.gz", mask_A)
        vocal_A = extract_mean_in_mask(subj / "half-A_vocal_beta.nii.gz", mask_B)
        nonvocal_A = extract_mean_in_mask(subj / "half-A_nonvocal_beta.nii.gz", mask_B)

        rows.append({
            "subject": sid,
            "vocal_beta": (vocal_A + vocal_B) / 2.0,
            "nonvocal_beta": (nonvocal_A + nonvocal_B) / 2.0,
        })

    return pd.DataFrame(rows)


def plot_responses(df, out_dir: Path):
    """2-bar plot: vocal vs non-vocal, group mean ± SEM (non-vocal at the zero baseline).

    Not golden-mastered (a visual artefact); the numeric net is ``cv_responses.csv``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats

    # Shift non-vocal to zero (baseline convention matching Marvi Fig. 4).
    baseline = df["nonvocal_beta"].mean()
    df = df.copy()
    df["vocal_beta"] -= baseline
    df["nonvocal_beta"] -= baseline

    long_df = df[["vocal_beta", "nonvocal_beta"]].rename(
        columns={"vocal_beta": "Vocals", "nonvocal_beta": "Non-Vocals"}
    ).melt(var_name="condition", value_name="beta")

    # One-sided paired t-test: vocal > non-vocal (printed, not plotted).
    t_val, p_val = stats.ttest_rel(df["vocal_beta"], df["nonvocal_beta"], alternative="greater")
    print(f"One-sided paired t-test (vocal > non-vocal): t({len(df) - 1}) = {t_val:.3f}, p = {p_val:.4f}")

    sns.set_theme(context="paper", font_scale=2, style="whitegrid")
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    sns.barplot(x="condition", y="beta", data=long_df, order=["Vocals", "Non-Vocals"],
                palette="Blues_r", errorbar="se", ax=ax)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylim(-0.025, 0.14)
    ax.set_yticks([0.0, 0.1])
    ax.set_ylabel("Beta", fontweight="bold")
    ax.set_xlabel("")
    sns.despine()
    plt.tight_layout()

    for fmt in ("svg", "png"):
        out = out_dir / f"cv_bar_plot.{fmt}"
        fig.savefig(str(out), dpi=300 if fmt == "png" else None, bbox_inches="tight")
        print(f"Saved: {out.name}")
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", type=Path, required=True,
                   help="Root holding 04_cross_validation/ (group masks + per_subject betas; output too).")
    p.add_argument("--n-subjects", type=int, default=N_SUBJECTS)
    p.add_argument("--no-figure", action="store_true", help="Write cv_responses.csv only; skip the plot.")
    p.add_argument("--plots-dir", type=Path, default=None,
                   help="Directory for cv_bar_plot.{svg,png} (default: <results-root>/04_cross_validation, "
                        "alongside cv_responses.csv). The CSV always stays under --results-root.")
    args = p.parse_args(argv)

    from scipy import stats  # lazy (for the group-mean report)

    df = extract_responses(args.results_root, args.n_subjects)

    out_dir = args.results_root / CV_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(out_dir / "cv_responses.csv"), index=False)
    print(f"Extracted responses for {len(df)} / {args.n_subjects} subjects -> cv_responses.csv")
    print(f"  Vocal    : {df['vocal_beta'].mean():.4f} ± {stats.sem(df['vocal_beta']):.4f}")
    print(f"  Non-vocal: {df['nonvocal_beta'].mean():.4f} ± {stats.sem(df['nonvocal_beta']):.4f}")

    if not args.no_figure:
        plots_dir = args.plots_dir or out_dir
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_responses(df, plots_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
