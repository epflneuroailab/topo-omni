#!/usr/bin/env python3
"""Group-level fROI definition for the cross-validated TVA profile (Fig. 3b, step cv_04).

For each block-level half-split fold (A and B):
  1. Stack the per-subject half-split vocal>non-vocal effect-size maps.
  2. Fit a second-level one-sample t-test (identical engine to 01_group_analysis.py).
  3. Threshold at Pernet's GRF-FWE value t > 4.79 -> a binary fROI mask.

The fold-A mask localizes voxels; responses are then measured from fold-B data (and
vice versa) in cv_05 — the mask and the responses never share data, so the profile is
cross-validated (no double-dipping).

Lineage (docs/DESIGN.md §2.4):  cv_02/cv_03 per-subject split GLMs (the precomputed CV cut)
  -> **cv_04 group fROI** -> cv_05 response extraction + 2-bar plot.
  input : <results-root>/04_cross_validation/per_subject/sub*/half-{A,B}_contrast.nii.gz  (218)
  output: <results-root>/04_cross_validation/group/half-{A,B}_{t_map,fROI_mask}.nii.gz
          + half-{A,B}_summary.json

PORT NOTES vs dev-repo `src/cv_04_group_froi_analysis.py` (@ f842b1a):
  * Faithful port; parameterized by `--results-root` (was hard-coded `results/...`).
  * The threshold is a strict `t > 4.79` (as in the dev code), matching Pernet's GRF-FWE
    value t(1,217), p<0.05, n=218. No behavioural change to the numerics.
  * `compute()` is side-effect-free (reads inputs only) so the golden master can assert
    on the returned masks/t-maps without touching disk.

⚠ nilearn-version pinning (docs/DESIGN.md §2.2/§4): same **dataset-specific GLM engine** as 01
(`SecondLevelModel`), pinned to Pernet's env (nilearn 0.10.4). Its numerics are NOT
assumed version-robust — the fROI voxel counts must be calibrated under the pinned env,
not under Marvi/Jung's 0.12.1 (Pernet_2015/tests/test_cv_group_froi_golden.py).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# --- Fixed analysis parameters (match the published run; do not drift) ---
N_SUBJECTS = 218
SUBJECT_TEMPLATE = "sub{:03d}_Ed"            # sub001_Ed .. sub218_Ed
CV_SUBDIR = "04_cross_validation"
PER_SUBJECT_SUBDIR = "per_subject"
GROUP_SUBDIR = "group"
FOLDS = ("A", "B")
FWE_THRESHOLD = 4.79                          # Pernet 2015 GRF-FWE, t(1,217), p<0.05, n=218


def subject_ids(n_subjects=N_SUBJECTS):
    return [SUBJECT_TEMPLATE.format(i) for i in range(1, n_subjects + 1)]


def load_fold_contrasts(per_subject_dir: Path, fold: str, n_subjects=N_SUBJECTS):
    """Load ``half-{fold}_contrast.nii.gz`` for every subject that has it.

    Returns ``(images, loaded_ids)`` in subject order; a subject is included iff its
    fold contrast file is present (mirrors the dev loader).
    """
    import nibabel as nib  # lazy (nilearn-env only)

    images, loaded = [], []
    for sid in subject_ids(n_subjects):
        f = per_subject_dir / sid / f"half-{fold}_contrast.nii.gz"
        if f.exists():
            images.append(nib.load(str(f)))
            loaded.append(sid)
    if not images:
        raise FileNotFoundError(f"No half-{fold}_contrast.nii.gz found under {per_subject_dir}")
    return images, loaded


def compute(results_root: Path, n_subjects=N_SUBJECTS):
    """Define the fROI masks for both folds. Returns ``{fold: {...}}``.

    Each fold entry holds the group ``t_map`` (Nifti1Image), the binary ``mask``
    (Nifti1Image), ``n_subjects`` and ``n_froi_voxels``. Side-effect-free.
    """
    import nibabel as nib
    from nilearn.glm.second_level import SecondLevelModel

    per_subject_dir = Path(results_root) / CV_SUBDIR / PER_SUBJECT_SUBDIR

    results = {}
    for fold in FOLDS:
        images, loaded = load_fold_contrasts(per_subject_dir, fold, n_subjects)

        # One-sample t-test: intercept-only design (identical to 01_group_analysis.py).
        design = pd.DataFrame({"intercept": np.ones(len(images))})
        model = SecondLevelModel(smoothing_fwhm=None)
        model.fit(second_level_input=images, design_matrix=design)
        t_map = model.compute_contrast("intercept", output_type="stat")

        # Pernet GRF-FWE: hard threshold at t > 4.79 (positive one-tailed).
        mask_data = (t_map.get_fdata() > FWE_THRESHOLD).astype(np.uint8)
        mask_img = nib.Nifti1Image(mask_data, t_map.affine, t_map.header)

        results[fold] = {
            "t_map": t_map,
            "mask": mask_img,
            "n_subjects": len(loaded),
            "subject_ids": loaded,
            "n_froi_voxels": int(mask_data.sum()),
        }
    return results


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", type=Path, required=True,
                   help="Root holding 04_cross_validation/per_subject/ (input) and .../group/ (output).")
    p.add_argument("--n-subjects", type=int, default=N_SUBJECTS)
    args = p.parse_args(argv)

    import nibabel as nib  # lazy

    results = compute(args.results_root, args.n_subjects)

    out_dir = args.results_root / CV_SUBDIR / GROUP_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for fold in FOLDS:
        r = results[fold]
        nib.save(r["t_map"], str(out_dir / f"half-{fold}_t_map.nii.gz"))
        nib.save(r["mask"], str(out_dir / f"half-{fold}_fROI_mask.nii.gz"))
        summary = {
            "fold": fold,
            "n_subjects": r["n_subjects"],
            "fwe_method": "GRF-FWE p<0.05, Pernet 2015",
            "fwe_threshold_t": float(FWE_THRESHOLD),
            "n_froi_voxels": r["n_froi_voxels"],
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        (out_dir / f"half-{fold}_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"fold {fold}: n={r['n_subjects']}, FWE t>{FWE_THRESHOLD}, "
              f"fROI={r['n_froi_voxels']:,} voxels")
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
