#!/usr/bin/env python3
"""Concatenated first-level GLM in native T1w space (Branch B, step 08).

Pool all 5 effloc runs into ONE design matrix and fit a single nilearn `FirstLevelModel`
per subject × modality in native **T1w** volume space — the FS-FAST-style route the
Marvi et al. (2025) Figs 2 & 3 text describes ("concatenates all functional runs, fits
regression coefficients ... to the voxel-wise time course"). Saves beta / tmap / pval /
signed_log_p per contrast.

  5× T1w BOLD runs  --time-concat + one FirstLevelModel-->  beta/tmap/pval/signed_log_p (T1w)

Lineage (README §9):  **08 concatenated_glm** -> 09 project_to_native_surface -> 10 render.
Branch B ships the CONCATENATED GLM (08), NOT the per-run 07 route (PLAN §7 / index §8).
  input : <derivatives-root>/<subj>/func/<subj>_task-effloc_run-*_space-T1w_desc-preproc_bold.nii.gz
          + <subj>/func/<subj>_task-effloc_run-*_desc-confounds_timeseries.tsv (6 motion)
          + events from <raw-root>/<subj>/func/*_task-effloc{Visual,Auditory}Conditions_run-*_events.tsv
  output: <output-dir>/<subj>/<modality>/<subj>_<modality>_<contrast>_concat_space-T1w_<map>.nii.gz

PORT NOTES vs dev-repo `src/08_batch_concatenated_glm_T1w_space.py` (@ ef1da34):
  * Faithful port of the concat + GLM + signed-log-p math (byte-for-byte). Only emfl import
    is `emfl.io.events.get_effloc_events` (same as dev). Contrast KEYS match
    `config.{VISUAL,AUDITORY}_CONTRASTS`; the GLM formulas (condition-regressor algebra) are
    kept inline, identical to dev — they mirror Branch A's engine `_volumetric_contrast_formulas`.
  * **Explicit `--raw-root` for events** (dev hard-derived the raw tree by
    `str(derivatives).replace('derivatives','orig_data')`). When `--raw-root` is omitted the dev
    string-swap still applies (backward compatible) — same deviation Branch A's `first_level_glm`
    carries.
  * `run_concatenated_glm()` stays side-effect-free (returns the maps dict, as dev did); the save
    moves entirely to `main()` via `save_contrast_maps`.
  * Parameterized `--derivatives-root` / `--output-dir` (were hard-coded dev paths).

DETERMINISM (docs/DESIGN.md §6): HEAVY nilearn `FirstLevelModel` fit → **NOT golden-mastered** (same
regime as Branch A 06/splits). Deliverable = faithful port + parameterized paths + a spatial-r
provenance spot-check vs the published `concatenated_glm/` cut. Fit is memory-heavy (5 runs
pooled) → run on SLURM bigmem, not the login node (README §1).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib

# Make `Marvi_2025/` importable so `emfl` + `config` resolve when run as a script.
_DATASET_DIR = Path(__file__).resolve().parent.parent
if str(_DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(_DATASET_DIR))

from emfl.io.events import get_effloc_events  # noqa: E402
from emfl.config import (  # noqa: E402
    ALL_SUBJECTS,
    VISUAL_CONTRASTS,
    AUDITORY_CONTRASTS,
)

DEFAULT_RUNS = ("001", "002", "003", "004", "005")

# 6 motion confounds (dev 08 `load_confounds`), matching the original FS-FAST analysis.
MOTION_CONFOUNDS = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]


def _events_subject_dir(derivatives_root: Path, raw_root, subject: str) -> Path:
    """Raw-BIDS subject dir for events; explicit --raw-root or dev derivatives->orig_data swap."""
    if raw_root:
        return Path(raw_root) / subject
    return Path(str(derivatives_root).replace("derivatives", "orig_data")) / subject


def load_confounds(derivatives_root: Path, subject: str, task: str, run: str) -> pd.DataFrame:
    """6 motion params (trans/rot x/y/z), NaN->0. Faithful to dev 08."""
    confounds_path = (
        derivatives_root
        / subject
        / "func"
        / f"{subject}_task-{task}_run-{run}_desc-confounds_timeseries.tsv"
    )
    if not confounds_path.exists():
        raise FileNotFoundError(f"Confounds file not found: {confounds_path}")
    confounds_df = pd.read_csv(confounds_path, sep="\t")
    available = [c for c in MOTION_CONFOUNDS if c in confounds_df.columns]
    return confounds_df[available].fillna(0)


def concatenate_runs(
    derivatives_root: Path, subject: str, runs, space: str = "T1w", raw_root=None
) -> tuple:
    """Time-concatenate BOLD + events (onset-shifted) + confounds across runs. Faithful to dev 08."""
    events_subject_dir = _events_subject_dir(derivatives_root, raw_root, subject)

    bold_imgs = []
    events_list = []
    confounds_list = []
    cumulative_time = 0.0

    for run in runs:
        bold_path = (
            derivatives_root
            / subject
            / "func"
            / f"{subject}_task-effloc_run-{run}_space-{space}_desc-preproc_bold.nii.gz"
        )
        if not bold_path.exists():
            raise FileNotFoundError(f"BOLD file not found: {bold_path}")
        img = nib.load(str(bold_path))
        bold_imgs.append(img)

        tr = float(img.header.get_zooms()[3])
        n_volumes = img.shape[3]
        run_duration = n_volumes * tr

        events_visual = get_effloc_events(events_subject_dir, run, modality="visual")
        events_auditory = get_effloc_events(events_subject_dir, run, modality="auditory")
        events_combined = pd.concat([events_visual, events_auditory], ignore_index=True)
        events_combined["onset"] = events_combined["onset"] + cumulative_time
        events_list.append(events_combined)

        confounds_list.append(load_confounds(derivatives_root, subject, "effloc", run))
        cumulative_time += run_duration

    bold_data_list = [img.get_fdata() for img in bold_imgs]
    bold_concat_data = np.concatenate(bold_data_list, axis=3)
    bold_concat = nib.Nifti1Image(
        bold_concat_data, bold_imgs[0].affine, bold_imgs[0].header
    )

    events_concat = (
        pd.concat(events_list, ignore_index=True)
        .sort_values("onset")
        .reset_index(drop=True)
    )
    confounds_concat = pd.concat(confounds_list, ignore_index=True).reset_index(drop=True)
    return bold_concat, events_concat, confounds_concat


def compute_signed_log_p(t_map, p_map):
    """signed log-p = -log10(max(p,1e-300)) * sign(t), with inf->±300, nan->0. Faithful to dev 08."""
    t_data = t_map.get_fdata()
    p_data = p_map.get_fdata()
    p_data_thresh = np.maximum(p_data, 1e-300)
    signed_log_p_data = -np.log10(p_data_thresh) * np.sign(t_data)
    signed_log_p_data = np.nan_to_num(
        signed_log_p_data, nan=0.0, posinf=300.0, neginf=-300.0
    )
    return nib.Nifti1Image(signed_log_p_data, t_map.affine, t_map.header)


def concat_contrast_formulas(modality: str) -> dict:
    """{contrast_key: GLM formula} for a modality (inline, byte-faithful to dev 08)."""
    if modality == "visual":
        return {
            "faces_vs_objects": "faces - objects",
            "scenes_vs_objects": "scenes - objects",
            "bodies_vs_objects": "bodies - objects",
            "words_vs_objects": "words_scr_objects - objects",
            "objects_vs_words": "objects - words_scr_objects",
        }
    return {
        "false_belief_vs_false_photo": "false_belief - false_photo",
        "english_vs_nonwords": "0.5*false_belief + 0.5*false_photo - nonwords",
        "nonwords_vs_quilted": "nonwords - quilted_speech",
        "math_vs_theory_of_mind": "math - 0.5*false_belief - 0.5*false_photo",
    }


def run_concatenated_glm(
    derivatives_root: Path,
    subject: str,
    runs,
    modality: str,
    space: str = "T1w",
    smoothing_fwhm: float = 3.0,
    contrasts_filter=None,
    raw_root=None,
) -> dict:
    """Fit ONE GLM over the concatenated runs; return {contrast: {beta,tmap,pval,signed_log_p}}.

    Side-effect-free (no disk writes). Heavy nilearn fit.
    """
    from nilearn.glm.first_level import FirstLevelModel

    print(f"\n{'='*70}\nConcatenated GLM: {subject} | {modality.upper()} | space={space}")
    bold_concat, events_concat, confounds_concat = concatenate_runs(
        derivatives_root, subject, runs, space, raw_root=raw_root
    )
    tr = float(bold_concat.header.get_zooms()[3])
    print(f"  BOLD {bold_concat.shape}  TR={tr:.3f}s  events={len(events_concat)}  "
          f"confounds={confounds_concat.shape}")

    glm = FirstLevelModel(
        t_r=tr,
        noise_model="ar1",
        standardize=False,
        hrf_model="spm",
        drift_model="polynomial",
        drift_order=1,
        high_pass=0.01,
        smoothing_fwhm=smoothing_fwhm,
        minimize_memory=False,
    )
    glm.fit(bold_concat, events=events_concat, confounds=confounds_concat)

    contrasts = concat_contrast_formulas(modality)
    if contrasts_filter:
        contrasts = {k: v for k, v in contrasts.items() if k in contrasts_filter}

    contrast_maps = {}
    for contrast_name, formula in contrasts.items():
        beta_map = glm.compute_contrast(formula, output_type="effect_size")
        t_map = glm.compute_contrast(formula, output_type="stat")
        p_map = glm.compute_contrast(formula, output_type="p_value")
        signed_log_p_map = compute_signed_log_p(t_map, p_map)
        contrast_maps[contrast_name] = {
            "beta": beta_map,
            "tmap": t_map,
            "pval": p_map,
            "signed_log_p": signed_log_p_map,
        }
        slp = signed_log_p_map.get_fdata()
        print(f"  {contrast_name}: sig p<0.001 = {int(np.sum(slp > 3))}+/"
              f"{int(np.sum(slp < -3))}-")
    return contrast_maps


def contrast_map_path(output_dir: Path, subject: str, modality: str, contrast: str, map_type: str) -> Path:
    """Single source of truth for the on-disk concat-GLM map layout (dev 08)."""
    return (
        output_dir
        / subject
        / modality
        / f"{subject}_{modality}_{contrast}_concat_space-T1w_{map_type}.nii.gz"
    )


def save_contrast_maps(output_dir: Path, subject: str, modality: str, contrast_maps: dict):
    """Write each contrast map to `contrast_map_path(...)`."""
    for contrast_name, maps in contrast_maps.items():
        for map_type, img in maps.items():
            out_path = contrast_map_path(output_dir, subject, modality, contrast_name, map_type)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(img, str(out_path))
            print(f"  saved {out_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Concatenated first-level GLM in T1w space (Branch B, 08)."
    )
    parser.add_argument("--subjects", nargs="+", default=list(ALL_SUBJECTS))
    parser.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS))
    parser.add_argument("--derivatives-root", type=str, required=True)
    parser.add_argument(
        "--raw-root",
        type=str,
        default=None,
        help="Raw-BIDS root for event TSVs (default: derived from derivatives-root).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output dir (default: <derivatives-root>/concatenated_glm).",
    )
    parser.add_argument("--space", type=str, default="T1w")
    parser.add_argument("--smoothing", type=float, default=3.0)
    parser.add_argument(
        "--modality", type=str, default=None, choices=["visual", "auditory"]
    )
    parser.add_argument("--contrasts", nargs="+", default=None)
    parser.add_argument("--test", action="store_true", help="First subject only.")
    args = parser.parse_args()

    derivatives_root = Path(args.derivatives_root)
    output_dir = (
        Path(args.output_dir) if args.output_dir else derivatives_root / "concatenated_glm"
    )
    subjects = args.subjects[:1] if args.test else args.subjects
    modalities = [args.modality] if args.modality else ["visual", "auditory"]

    print("=" * 70)
    print("BRANCH B / 08: concatenated GLM (T1w native)")
    print(f"  subjects={len(subjects)}  runs={len(args.runs)}  modalities={modalities}")
    print(f"  raw (events)={args.raw_root or '(derived from derivatives)'}")
    print(f"  output_dir={output_dir}")
    print("=" * 70)

    failed = []
    for subject in subjects:
        for modality in modalities:
            try:
                maps = run_concatenated_glm(
                    derivatives_root,
                    subject,
                    args.runs,
                    modality,
                    space=args.space,
                    smoothing_fwhm=args.smoothing,
                    contrasts_filter=args.contrasts,
                    raw_root=args.raw_root,
                )
                save_contrast_maps(output_dir, subject, modality, maps)
                print(f"✓ {subject} {modality} complete")
            except Exception as e:  # noqa: BLE001 — dev batch semantics: log + continue
                print(f"✗ {subject} {modality} FAILED: {e}")
                failed.append(f"{subject}_{modality}")

    print("=" * 70)
    print(f"CONCATENATED GLM COMPLETE — {len(failed)} failed")
    if failed:
        for job in failed:
            print(f"  - {job}")
        sys.exit(1)


if __name__ == "__main__":
    main()
