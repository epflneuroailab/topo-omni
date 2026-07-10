"""Heavy golden master — group cluster-contrast maps (docs/DESIGN.md §6 Tier 1, new54 family).

Re-runs the full cluster GLM (glm_engine.run_full_analysis) on the on-disk fsaverage6
derivatives and asserts the group `{tstat,pval,mean}` maps reproduce the published
`group_level/` GIFTIs, plus the n=78 / df=77 invariant (docs/DESIGN.md §7). Covers all six
published clusters: 5/32/49 (Fig. 6 / D4) + 6/30/31 (Fig. D5) — README §4.

The engine is pure numpy/scipy/nibabel (no nilearn) so it is expected to reproduce the
published maps to near machine precision. The tight tolerance below is **calibrated then
frozen** (docs/DESIGN.md §6): run tests/calibrate_heavy_golden.py under the pinned env (the
neuromod / numpy-2.2.5 stack that produced the maps), read the measured worst max|Δ| and
worst Pearson r, and bake them in here — replacing the @skip.

HEAVY: loads 78×13-run fsaverage6 BOLD (~5 GB/subject); run on SLURM bigmem, not inline
(README §3). Gated behind JUNG_RUN_HEAVY=1 so the default fast suite never triggers it.

STATUS: calibrated on SLURM 2026-07-07 and frozen below (ATOL=1e-9, MIN_PEARSON_R=1−1e-9).
All six published clusters reproduce the on-disk group maps BITWISE (worst max|Δ| = 0.0).
Active — gated only by JUNG_RUN_HEAVY=1 (heavy, run on SLURM bigmem).
"""
import importlib.util
import os
from pathlib import Path

import pytest

_DATASET = Path(__file__).resolve().parent.parent
_ANALYSIS = _DATASET / "analysis"
_DATA = _DATASET / "data" / "cluster_assignments"

_CANDIDATE_DERIV = [
    os.environ.get("JUNG_DERIVATIVES_ROOT"),
    "/work/upschrimpf1/mehrer/datasets/fMRI_movie_watching/spacetop/ds005256/derivatives",
]
_CANDIDATE_BIDS = [
    os.environ.get("JUNG_RAW_ROOT"),
    "/work/upschrimpf1/mehrer/datasets/fMRI_movie_watching/spacetop/ds005256",
]

# The six published clusters (all new54): 5/32/49 → Fig. 6/D4, 6/30/31 → Fig. D5.
PAPER_CLUSTER_IDS = [5, 32, 49, 6, 30, 31]
_CSV_NAME = "cluster_assignments_new54clusters.csv"
_SUBDIR = "cluster_contrasts_new54clusters"

# --- FROZEN tolerances (calibrate → freeze, docs/DESIGN.md §6). ---
# Calibrated on SLURM bigmem 2026-07-07 under the pinned neuromod env (python 3.10.19 /
# numpy 2.2.5 / scipy 1.15.3 / nibabel 5.3.2 / nilearn 0.12.1) via calibrate_heavy_golden.py,
# arrays 65434649_3/4/5 (c6/30/31) + 65435894_0/1/2 (c5/32/49). All six published clusters
# reproduce the on-disk group tstat/pval/mean GIFTIs BITWISE: worst max|Δ| = 0.0 across all
# 36 maps (6 clusters × {tstat,pval,mean} × {L,R}); worst Pearson r = 0.9999999999999998
# (= 1 − 2e-16, float64 rounding). This is expected: per-subject t-maps are bitwise
# (job 65434411) and the group step is deterministic ttest_1samp + FDR. Frozen with
# generous headroom over the measured 0.0 (mirrors Pernet's engine-golden policy).
ATOL = 1e-9              # measured worst = 0.0; catches any real drift, ~∞ headroom
MIN_PEARSON_R = 1 - 1e-9 # measured worst = 1 − 2e-16

_RUN_HEAVY = os.environ.get("JUNG_RUN_HEAVY") == "1"

pytestmark = pytest.mark.skipif(
    not _RUN_HEAVY, reason="heavy group GLM (set JUNG_RUN_HEAVY=1; run on SLURM bigmem)"
)


def _resolve(cands, marker):
    for c in cands:
        if c and (Path(c) / marker).exists():
            return Path(c)
    return None


def _load_engine():
    import sys
    sys.path.insert(0, str(_ANALYSIS))
    spec = importlib.util.spec_from_file_location("jung_glm_engine", _ANALYSIS / "glm_engine.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("cid", PAPER_CLUSTER_IDS)
def test_group_maps_reproduce_published(tmp_path, cid):
    import numpy as np
    import nibabel as nib

    deriv = _resolve(_CANDIDATE_DERIV, "sub-0001")
    bids = _resolve(_CANDIDATE_BIDS, "sub-0001")
    if deriv is None or bids is None:
        pytest.skip("derivatives / raw BIDS not found (set JUNG_DERIVATIVES_ROOT / JUNG_RAW_ROOT)")

    engine = _load_engine()
    published = deriv / _SUBDIR / "group_level"

    def gii(p):
        return np.asarray(nib.load(str(p)).darrays[0].data, dtype=np.float64)

    out = tmp_path / f"new54_c{cid}"
    engine.run_full_analysis(
        target_id=cid, output_dir=out,
        bold_dir=str(deriv), confounds_dir=str(deriv), bids_dir=str(bids),
        cluster_file=str(_DATA / _CSV_NAME),
    )
    import json
    summary = json.loads((out / "group_level" / "summary.json").read_text())
    assert summary["n_subjects"] == 78 and summary["df"] == 77  # the published analysis

    prefix = f"group_cluster-{cid:02d}_space-fsaverage6"
    for maptype in ("tstat", "pval", "mean"):
        for hemi in ("L", "R"):
            name = f"{prefix}_hemi-{hemi}_{maptype}.func.gii"
            got = gii(out / "group_level" / name)
            ref = gii(published / name)
            assert np.corrcoef(got, ref)[0, 1] >= MIN_PEARSON_R
            assert np.allclose(got, ref, atol=ATOL, rtol=0)
