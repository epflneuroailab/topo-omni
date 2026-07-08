#!/usr/bin/env python3
"""Volumetric group analysis for the Pernet voice localizer (Fig. 3b lineage step 01).

Second-level (random-effects) one-sample t-test over the per-subject vocal>non-vocal
contrast estimates, following Pernet et al. (2015): "the contrast images from each
participant were entered in a one-sample t-test." Produces the group t/z/p maps, the
GRF-FWE-thresholded map, and the cluster table.

Lineage (docs/DESIGN.md §2.4):  00 volumetric GLM  ->  **01 group analysis**  ->  02 surface.
  input : <results-root>/00_volumetric_GLM/sub*/sub*_contrast_estimates.nii.gz  (218, MNI 2mm)
  output: <results-root>/01_group_analysis/{t_map,z_map,p_map,thresholded_t_map}.nii.gz
          + cluster_table.csv + analysis_summary.json

PORT NOTES vs dev-repo `src/01_group_analysis.py` + `src/processing/group_analysis.py`
(@ f842b1a):
  * Faithful port of the `VolumetricGroupAnalysis` path (the one 01 actually runs); the
    unused surface-based `GroupAnalyzer` class and all progress-bar/print scaffolding
    are dropped.
  * Parameterized by `--results-root` (was hard-coded `results/...`). No behavioural
    change to the numerics.
  * FWE threshold is Pernet's fixed GRF value **t >= 4.79** (t(1,217), p<0.05 FWE,
    n=218) — applied as a hard threshold, exactly as the dev code did (`apply_fwe_correction`
    ignores its alpha and uses `pernet_threshold`).

⚠ nilearn-version pinning (docs/DESIGN.md §2.2/§4): this is a **dataset-specific GLM engine**
and stays local; it is pinned to Pernet's env (nilearn 0.10.4, see
environment/analysis_env_pernet.yml). The `SecondLevelModel` numerics are NOT assumed
version-robust — the golden master must be calibrated under that pinned env, not under
Marvi/Jung's 0.12.1 (Pernet_2015/tests/test_group_analysis_golden.py).

STATUS: ported (Stage 1). Reproduction not yet re-run under the pinned 0.10.4 env in
this workspace (218-subject second-level fit; memory-heavy) — golden tolerance is a
calibration TODO, see the test.
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
CONTRAST_FILE_SUFFIX = "_contrast_estimates.nii.gz"
FWE_THRESHOLD = 4.79                          # Pernet 2015 GRF-FWE, t(1,217), p<0.05, n=218
FWE_ALPHA = 0.05
CLUSTER_THRESHOLD = 10                        # min cluster size (voxels) for reporting
PEAK_MIN_DISTANCE = 8.0                       # mm between reported peaks


def subject_ids(n_subjects=N_SUBJECTS):
    return [SUBJECT_TEMPLATE.format(i) for i in range(1, n_subjects + 1)]


def load_contrast_images(glm_dir: Path, n_subjects=N_SUBJECTS):
    """Load per-subject contrast-estimate images that exist under ``glm_dir``.

    Returns ``(images, loaded_ids)`` in subject order. Mirrors the dev loader: a
    subject is included iff its contrast file is present.
    """
    import nibabel as nib  # lazy (nilearn-env only)

    images, loaded = [], []
    for sid in subject_ids(n_subjects):
        f = glm_dir / sid / f"{sid}{CONTRAST_FILE_SUFFIX}"
        if f.exists():
            images.append(nib.load(str(f)))
            loaded.append(sid)
    if not images:
        raise FileNotFoundError(f"No contrast estimates found under {glm_dir}")
    return images, loaded


def compute(results_root: Path, n_subjects=N_SUBJECTS):
    """Run the second-level GLM and return maps + cluster table + summary.

    Kept side-effect-free (reads inputs only) so the golden-master test can call it and
    assert on the returned maps without touching disk.
    """
    from nilearn.glm.second_level import SecondLevelModel
    from nilearn.image import math_img
    from nilearn.reporting import get_clusters_table

    results_root = Path(results_root)
    glm_dir = results_root / "00_volumetric_GLM"
    images, loaded = load_contrast_images(glm_dir, n_subjects)

    # One-sample t-test: intercept-only design (random effects over contrast images).
    design = pd.DataFrame({"intercept": np.ones(len(images))})
    model = SecondLevelModel(smoothing_fwhm=None, mask_img=None,
                             minimize_memory=True, n_jobs=1)
    model.fit(second_level_input=images, design_matrix=design)

    t_map = model.compute_contrast("intercept", output_type="stat")
    z_map = model.compute_contrast("intercept", output_type="z_score")
    p_map = model.compute_contrast("intercept", output_type="p_value")

    # Pernet GRF-FWE: hard threshold at t >= 4.79 (positive one-tailed).
    thresholded = math_img(f"img * (img >= {FWE_THRESHOLD})", img=t_map)

    clusters = get_clusters_table(
        thresholded, stat_threshold=FWE_THRESHOLD,
        cluster_threshold=CLUSTER_THRESHOLD, min_distance=PEAK_MIN_DISTANCE,
    )
    if len(clusters) > 0:
        clusters = clusters.copy()
        clusters["threshold"] = FWE_THRESHOLD
        clusters["correction"] = "FWE"
        # get_clusters_table puts ints on main-peak rows but "" on sub-peak rows;
        # coerce so the mixed column sorts (sub-peaks -> NaN, sorted last). Matches
        # the dev repo's numeric-coercion fallback (group_analysis.py:867).
        clusters["Cluster Size (mm3)"] = pd.to_numeric(
            clusters["Cluster Size (mm3)"], errors="coerce"
        )
        clusters = clusters.sort_values("Cluster Size (mm3)", ascending=False)

    summary = {
        "n_subjects": len(loaded),
        "subject_ids": loaded,
        "fwe_threshold": float(FWE_THRESHOLD),
        "n_clusters": int(len(clusters)),
        "pernet_threshold": FWE_THRESHOLD,
        "fwe_alpha": FWE_ALPHA,
        "cluster_threshold": CLUSTER_THRESHOLD,
    }
    maps = {"t_map": t_map, "z_map": z_map, "p_map": p_map,
            "thresholded_t_map": thresholded}
    return maps, clusters, summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", type=Path, required=True,
                   help="Root holding 00_volumetric_GLM/ (input) and 01_group_analysis/ (output).")
    p.add_argument("--n-subjects", type=int, default=N_SUBJECTS)
    args = p.parse_args(argv)

    import nibabel as nib  # lazy

    maps, clusters, summary = compute(args.results_root, args.n_subjects)

    out_dir = args.results_root / "01_group_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, img in maps.items():
        nib.save(img, str(out_dir / f"{name}.nii.gz"))
    clusters.to_csv(out_dir / "cluster_table.csv", index=False)
    summary_out = {"analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **summary}
    (out_dir / "analysis_summary.json").write_text(json.dumps(summary_out, indent=2, default=str))

    print(f"Group analysis: {summary['n_subjects']} subjects, "
          f"FWE t>={summary['fwe_threshold']}, {summary['n_clusters']} clusters -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
