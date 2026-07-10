"""Golden master for the CV group fROI definition — step cv_04 (docs/DESIGN.md §6 Tier 1).

Re-runs the two half-split second-level GLMs and compares the group t-maps + fROI masks
against the published ``results/04_cross_validation/group/half-{A,B}_*.nii.gz``.

Gated exactly like step 01's golden master, and for the same two reasons:

  1. **Heavy**: fitting two 218-subject `SecondLevelModel`s loads the full stacks into
     memory. Opt-in via ``PERNET_RUN_HEAVY=1`` so a normal `pytest` run never triggers it.

  2. **Version-pinned tolerance CALIBRATED (2026-07-06).** `SecondLevelModel` is a
     dataset-specific GLM engine pinned to Pernet's nilearn **0.10.4** (docs/DESIGN.md §2.2);
     the published masks (A=2185, B=2944 voxels @ t>4.79) were produced under 0.10.4.
     Measured on a bigmem node (SLURM job 65405112) under BOTH pinned envs: both folds
     reproduce the published counts EXACTLY (Δ0) with Dice=1.0, bit-identical across
     0.10.4 and 0.12.1. `test_froi_mask_counts` is now enabled.

DATA-GATED: needs <root>/04_cross_validation/per_subject/sub*/half-{A,B}_contrast.nii.gz
and the golden <root>/04_cross_validation/group/half-{A,B}_{t_map,fROI_mask}.nii.gz.
Point it with PERNET_RESULTS_ROOT.
"""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

_DATASET = Path(__file__).resolve().parent.parent
_CANDIDATE_ROOTS = [
    os.environ.get("PERNET_RESULTS_ROOT"),
    "/work/upschrimpf1/mehrer/code/20241003_pernet_2015/results",
]

_RUN_HEAVY = os.environ.get("PERNET_RUN_HEAVY") == "1"

# Published fROI voxel counts (t > 4.79), from group/half-{A,B}_summary.json.
_PUBLISHED_VOXELS = {"A": 2185, "B": 2944}


def _has_nilearn():
    try:
        import nilearn  # noqa: F401
        return True
    except ImportError:
        return False


def _resolve_root():
    for root in _CANDIDATE_ROOTS:
        if not root:
            continue
        root = Path(root)
        cv = root / "04_cross_validation"
        if (cv / "per_subject" / "sub001_Ed" / "half-A_contrast.nii.gz").exists() \
                and (cv / "group" / "half-A_t_map.nii.gz").exists():
            return root
    return None


def _load_driver():
    path = _DATASET / "analysis" / "cv_04_group_froi_analysis.py"
    spec = importlib.util.spec_from_file_location("pernet_cv_04", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pytestmark = [
    pytest.mark.skipif(not _has_nilearn(), reason="nilearn not installed"),
    pytest.mark.skipif(not _RUN_HEAVY, reason="heavy 2x 218-subject fit; set PERNET_RUN_HEAVY=1"),
]


@pytest.fixture(scope="module")
def result_and_gold():
    root = _resolve_root()
    if root is None:
        pytest.skip("CV per-subject contrasts + golden group maps not found (set PERNET_RESULTS_ROOT)")
    import nibabel as nib

    driver = _load_driver()
    results = driver.compute(root)
    group = root / "04_cross_validation" / "group"
    gold = {fold: {
        "t_map": nib.load(str(group / f"half-{fold}_t_map.nii.gz")).get_fdata(),
        "mask": nib.load(str(group / f"half-{fold}_fROI_mask.nii.gz")).get_fdata(),
    } for fold in ("A", "B")}
    return results, gold


def test_both_folds_full_cohort(result_and_gold):
    results, _ = result_and_gold
    assert set(results) == {"A", "B"}
    for fold in ("A", "B"):
        assert results[fold]["n_subjects"] == 218
        assert results[fold]["n_froi_voxels"] > 0   # masks are non-empty


@pytest.mark.parametrize("fold", ["A", "B"])
def test_tmap_shape_and_correlation(result_and_gold, fold):
    results, gold = result_and_gold
    a = results[fold]["t_map"].get_fdata()
    b = gold[fold]["t_map"]
    assert a.shape == b.shape
    m = np.isfinite(a) & np.isfinite(b)
    # Version-stable invariant: the half-split group t-maps match in structure.
    assert np.corrcoef(a[m], b[m])[0, 1] > 0.9999


# Calibrated 2026-07-06 on bigmem (SLURM job 65405112): both folds reproduce the
# published fROI counts EXACTLY (A=2185, B=2944, Δ0) with Dice=1.0, bit-identical
# across nilearn 0.10.4 and 0.12.1. See docs/DESIGN.md "DONE — heavy golden-master calibration".
@pytest.mark.parametrize("fold", ["A", "B"])
def test_froi_mask_counts(result_and_gold, fold):
    results, gold = result_and_gold
    assert results[fold]["n_froi_voxels"] == _PUBLISHED_VOXELS[fold]
    got = results[fold]["mask"].get_fdata().astype(bool)
    ref = gold[fold]["mask"].astype(bool)
    dice = 2 * (got & ref).sum() / (got.sum() + ref.sum())
    assert dice > 0.99
