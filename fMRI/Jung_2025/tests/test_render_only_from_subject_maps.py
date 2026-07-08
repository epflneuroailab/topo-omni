"""Characterization — render-only reproduction from the shipped subject-level t-maps.

This pins the **precomputed-cut / Tier-1 reproduction path** (docs/DESIGN.md §5.1-C / §9 step 2):
`glm_engine.run_full_analysis(reuse_subject_maps=True)` loads the shipped per-subject
`subject_level/cluster-XX/sub-*/sub-*_hemi-{L,R}_tstat.func.gii` maps and runs ONLY the
deterministic group ttest+FDR — with NO access to the ~1.5 TB fsaverage6 BOLD. This is what
`make_figures.py --input-source precomputed` does, and what the OSF cut must support.

Distinct from `test_group_maps_golden.py`, which recomputes the subject maps from BOLD in
float64 and reproduces the published group maps BITWISE (ATOL=1e-9). Here the inputs are the
shipped subject t-maps, stored as **float32 GIFTI** — so re-deriving the group stats from them
carries a float32 storage floor: the group maps match the published `group_level/` only to
~1e-5 on t (spatial-r still ≈ 1). That is scientifically identical (the thresholded figure is
unchanged) but NOT bitwise, and the tolerance below is calibrated to that floor, then frozen.

LIGHT: loads 78 small per-subject t-map GIFTIs per cluster (~31 MB/cluster) — runs on the
login node in seconds. Gated only on the on-disk subject-level cut being present.
"""
import importlib.util
import os
from pathlib import Path

import pytest

_DATASET = Path(__file__).resolve().parent.parent
_ANALYSIS = _DATASET / "analysis"
_DATA = _DATASET / "data" / "cluster_assignments"
_SUBDIR = "cluster_contrasts_new54clusters"
_CSV_NAME = "cluster_assignments_new54clusters.csv"

PAPER_CLUSTER_IDS = [5, 32, 49, 6, 30, 31]

_CANDIDATE_DERIV = [
    os.environ.get("JUNG_DERIVATIVES_ROOT"),
    "/work/upschrimpf1/mehrer/datasets/fMRI_movie_watching/spacetop/ds005256/derivatives",
]

# --- FROZEN render-only tolerance (calibrate → freeze, docs/DESIGN.md §6). ---
# Measured 2026-07-07 by carving the 6 published clusters' subject_level/ into a clean cut and
# reproducing the group maps via reuse mode (make_figures --input-source precomputed): worst
# max|Δ| vs published group_level = 3.34e-6 on tstat (float32 storage floor of the shipped
# subject t-maps); worst Pearson r = 0.999999999999 (1 − 6.3e-13) across all 36 maps
# (6 clusters × {tstat,pval,mean} × {L,R}). Frozen with headroom.
ATOL_RENDER = 1e-5        # measured worst = 3.34e-6 (float32 subject-map storage)
MIN_PEARSON_R = 1 - 1e-9  # measured worst = 1 − 6.3e-13


def _resolve(cands, marker):
    for c in cands:
        if c and (Path(c) / marker).exists():
            return Path(c)
    return None


def _load_engine():
    import sys
    sys.path.insert(0, str(_ANALYSIS))
    spec = importlib.util.spec_from_file_location("jung_glm_engine_ro", _ANALYSIS / "glm_engine.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("cid", PAPER_CLUSTER_IDS)
def test_render_only_reproduces_group_maps(tmp_path, cid):
    import json
    import numpy as np
    import nibabel as nib

    deriv = _resolve(_CANDIDATE_DERIV, _SUBDIR)
    if deriv is None:
        pytest.skip("derivatives cut not found (set JUNG_DERIVATIVES_ROOT)")
    cut = deriv / _SUBDIR
    if not (cut / "subject_level" / f"cluster-{cid:02d}").is_dir():
        pytest.skip(f"shipped subject_level/cluster-{cid:02d} not present")

    engine = _load_engine()
    published = cut / "group_level"

    def gii(p):
        return np.asarray(nib.load(str(p)).darrays[0].data, dtype=np.float64)

    out = tmp_path / f"new54_c{cid}"
    ok = engine.run_full_analysis(
        target_id=cid, output_dir=out,
        bold_dir=None, confounds_dir=None, bids_dir=None,   # not read in reuse mode
        cluster_file=str(_DATA / _CSV_NAME),
        reuse_subject_maps=True,
        subject_maps_root=str(cut),                          # the read-only shipped cut
    )
    assert ok

    summary = json.loads((out / "group_level" / "summary.json").read_text())
    assert summary["n_subjects"] == 78 and summary["df"] == 77  # the published n=78/df=77

    prefix = f"group_cluster-{cid:02d}_space-fsaverage6"
    for maptype in ("tstat", "pval", "mean"):
        for hemi in ("L", "R"):
            name = f"{prefix}_hemi-{hemi}_{maptype}.func.gii"
            got = gii(out / "group_level" / name)
            ref = gii(published / name)
            assert np.corrcoef(got, ref)[0, 1] >= MIN_PEARSON_R, f"{name}: spatial-r too low"
            assert np.allclose(got, ref, atol=ATOL_RENDER, rtol=0), f"{name}: exceeds render tol"
