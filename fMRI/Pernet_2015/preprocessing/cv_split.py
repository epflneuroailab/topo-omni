"""Pernet CV Stage-0: block-level fold split (seed=42) + per-subject half GLMs.

Faithful port of the dev pipeline (20241003_pernet_2015 @ f842b1a):
  - cv_01_define_fold_split.py  -> define_fold_split()  (random 10+10 block split, seed=42)
  - cv_02_split_glm_single_subject.py -> run_split_glm() (per subject: preproc once, two half-GLMs)

Produces the precomputed CV cut consumed by analysis/cv_04:
  <results-root>/04_cross_validation/fold_split.json
  <results-root>/04_cross_validation/per_subject/<sub>/half-{A,B}_{contrast,vocal_beta,nonvocal_beta}.nii.gz

Path-agnostic: the raw BIDS root and the output root are parameters — there is no
baked-in dataset path (the dev scripts hard-coded /work/.../datasets/pernet_2015 and
sourced setup_fsl.sh; here FSL must already be on PATH, see preprocessing/README.md).

Numerics (RNG sequence, GLM design, contrasts) are verbatim from the dev scripts so
the published fold_split.json and per_subject/ maps reproduce.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib

from .timing import load_stimulus_order

# ─────────────────────────────────────────────────────────────────────────────
BLOCK_DURATION = 8.0   # seconds (dev cv_01/cv_02)
SEED = 42              # dev cv_01 block half-split seed
CV_SUBDIR = "04_cross_validation"
PER_SUBJECT_SUBDIR = "per_subject"
# ─────────────────────────────────────────────────────────────────────────────


# ── cv_01: fold split ────────────────────────────────────────────────────────
def define_fold_split(tva_loc_path: str, results_root: str) -> tuple[dict, Path]:
    """Random 10+10 block half-split (seed=42), exported as fold_split.json.

    Verbatim RNG sequence from dev cv_01_define_fold_split.py so the published
    fold_split.json reproduces: default_rng(42) -> permutation(20) [vocal] ->
    permutation(20) [nonvocal].
    """
    order = load_stimulus_order(str(tva_loc_path))   # list of 60 values

    # Position → onset time mapping
    vocal_positions    = [(i, v) for i, v in enumerate(order) if 1  <= v <= 20]
    nonvocal_positions = [(i, v) for i, v in enumerate(order) if 21 <= v <= 40]

    assert len(vocal_positions)    == 20, f"Expected 20 vocal blocks, got {len(vocal_positions)}"
    assert len(nonvocal_positions) == 20, f"Expected 20 non-vocal blocks, got {len(nonvocal_positions)}"

    rng = np.random.default_rng(SEED)
    v_perm  = rng.permutation(20)
    nv_perm = rng.permutation(20)

    def make_fold(perm_indices, positions):
        return sorted(
            [{"block_number": positions[i][1],
              "sequence_position": positions[i][0],
              "onset_sec": positions[i][0] * BLOCK_DURATION}
             for i in perm_indices],
            key=lambda x: x["onset_sec"]
        )

    vocal_A    = make_fold(v_perm[:10],  vocal_positions)
    vocal_B    = make_fold(v_perm[10:],  vocal_positions)
    nonvocal_A = make_fold(nv_perm[:10], nonvocal_positions)
    nonvocal_B = make_fold(nv_perm[10:], nonvocal_positions)

    split = {
        "seed": SEED,
        "block_duration_sec": BLOCK_DURATION,
        "description": (
            "Block-level half-split for cross-validated TVA fROI analysis. "
            "fold_A defines the fROI; responses are measured from fold_B, then vice versa."
        ),
        "note_for_model": (
            "block_number 1–20 = vocal blocks, 21–40 = non-vocal blocks. "
            "Use block_number to match blocks to stimuli in the Pernet stimulus set."
        ),
        "fold_A": {
            "vocal":    vocal_A,
            "nonvocal": nonvocal_A,
        },
        "fold_B": {
            "vocal":    vocal_B,
            "nonvocal": nonvocal_B,
        },
    }

    out_dir = Path(results_root) / CV_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fold_split.json"
    with open(out_path, "w") as f:
        json.dump(split, f, indent=2)

    return split, out_path


def load_fold_split(results_root: str) -> dict:
    fold_split = Path(results_root) / CV_SUBDIR / "fold_split.json"
    if not fold_split.exists():
        raise FileNotFoundError(
            f"Fold split not found: {fold_split}\nRun define_fold_split() (cv_01) first."
        )
    with open(fold_split) as f:
        return json.load(f)


# ── cv_02: per-subject split GLMs ────────────────────────────────────────────
def make_events_df(fold_data: dict) -> pd.DataFrame:
    """Build a nilearn-compatible events DataFrame from one fold's block list."""
    rows = []
    for block in fold_data["vocal"]:
        rows.append({"onset": block["onset_sec"], "duration": 8.0, "trial_type": "vocal"})
    for block in fold_data["nonvocal"]:
        rows.append({"onset": block["onset_sec"], "duration": 8.0, "trial_type": "non_vocal"})
    df = pd.DataFrame(rows).sort_values("onset").reset_index(drop=True)
    return df


def get_condition_beta(model, design_matrix: pd.DataFrame,
                       trial_type: str) -> nib.Nifti1Image:
    """
    Extract the beta map for a single condition (effect size relative to
    implicit baseline) using a unit contrast vector.
    """
    contrast_vec = np.zeros(len(design_matrix.columns))
    col_names = list(design_matrix.columns)
    if trial_type not in col_names:
        raise ValueError(f"'{trial_type}' not in design matrix columns: {col_names}")
    contrast_vec[col_names.index(trial_type)] = 1.0
    return model.compute_contrast(contrast_vec, output_type="effect_size")


def run_split_glm(subject_id: str, fold_split: dict,
                  raw_root: str, results_root: str) -> bool:
    """Per-subject half-split GLMs (preprocess once, fit fold-A and fold-B).

    Verbatim port of dev cv_02.run_split_glm — same preprocessing, design matrix,
    FirstLevelModel settings, and vocal>non_vocal / per-condition-beta contrasts.
    Paths are parameters: subjects dir = <raw_root>/subs, outputs to
    <results_root>/04_cross_validation/per_subject/<subject_id>/.
    """
    from nilearn.glm.first_level import FirstLevelModel

    from .data_loader import Pernet2015DataLoader
    from .volumetric_glm import VolumetricGLMAnalyzer
    from .motion_correction import extract_motion_from_dataset

    data_dir = Path(raw_root) / "subs"
    out_dir = Path(results_root) / CV_SUBDIR / PER_SUBJECT_SUBDIR / subject_id

    print(f"\n{'='*70}")
    print(f"CV SPLIT GLM: {subject_id}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already completed
    expected = [out_dir / f"half-{f}_{m}.nii.gz"
                for f in ("A", "B")
                for m in ("contrast", "vocal_beta", "nonvocal_beta")]
    if all(p.exists() for p in expected):
        print(f"  Outputs already exist, skipping {subject_id}")
        return True

    # ── 1. Load data ──────────────────────────────────────────────────────────
    loader = Pernet2015DataLoader(base_path=str(raw_root))
    func_img = loader.load_functional_data(subject_id)
    anat_img = loader.load_anatomical_data(subject_id)
    if func_img is None:
        raise RuntimeError(f"No functional data found for {subject_id}")

    # ── 2. Motion regressors from SPM .mat file ───────────────────────────────
    print("Extracting motion regressors from SPM .mat file …")
    try:
        motion_regressors, _, motion_stats = extract_motion_from_dataset(
            subject_id, str(data_dir)
        )
        print(f"  {motion_regressors.shape[1]} motion regressors, "
              f"{motion_stats['n_total_outliers']} outlier regressors")
    except Exception as e:
        print(f"  WARNING: motion extraction failed ({e}) — proceeding without motion regressors")
        motion_regressors = None

    # ── 3. Preprocess BOLD ONCE (expensive — ~30–60 min) ─────────────────────
    print("\nPreprocessing BOLD (slice-timing → motion correction → MNI → smoothing) …")
    analyzer = VolumetricGLMAnalyzer()
    preprocessed_img = analyzer.preprocess_functional_data(func_img, anat_img)
    n_scans = preprocessed_img.shape[-1]
    print(f"  Preprocessed BOLD shape: {preprocessed_img.shape}")

    # ── 4. Fit GLMs for each fold ─────────────────────────────────────────────
    for fold_name in ("A", "B"):
        print(f"\n--- Fold {fold_name} GLM ---")

        events_df = make_events_df(fold_split[f"fold_{fold_name}"])
        print(f"  Events: {len(events_df)} blocks "
              f"({(events_df.trial_type=='vocal').sum()} vocal, "
              f"{(events_df.trial_type=='non_vocal').sum()} non-vocal)")

        design_matrix = analyzer.create_design_matrix(
            events_df, n_scans,
            motion_regressors=motion_regressors
        )
        print(f"  Design matrix: {design_matrix.shape[1]} columns")

        model = FirstLevelModel(
            t_r=analyzer.tr,
            hrf_model=analyzer.hrf_model,
            drift_model=None,
            high_pass=None,
            noise_model="ar1",
            standardize=False,
            signal_scaling=0,
            smoothing_fwhm=None,
        )
        model.fit(preprocessed_img, design_matrices=design_matrix)
        print(f"  GLM fitted")

        # Vocal > non-vocal contrast (effect size map → used for group fROI definition)
        col_names = list(design_matrix.columns)
        cvec = np.zeros(len(col_names))
        cvec[col_names.index("vocal")]     =  1.0
        cvec[col_names.index("non_vocal")] = -1.0
        contrast_img = model.compute_contrast(cvec, output_type="effect_size")

        # Per-condition beta maps (used for response extraction)
        vocal_beta    = get_condition_beta(model, design_matrix, "vocal")
        nonvocal_beta = get_condition_beta(model, design_matrix, "non_vocal")

        # Save
        nib.save(contrast_img,    out_dir / f"half-{fold_name}_contrast.nii.gz")
        nib.save(vocal_beta,      out_dir / f"half-{fold_name}_vocal_beta.nii.gz")
        nib.save(nonvocal_beta,   out_dir / f"half-{fold_name}_nonvocal_beta.nii.gz")
        print(f"  Saved half-{fold_name} outputs to {out_dir}")

    # Done
    print(f"\nDone: {subject_id}  ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    return True
