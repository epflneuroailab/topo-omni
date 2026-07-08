"""Golden master for the Pernet group analysis — step 01 (docs/DESIGN.md §6 Tier 1).

Re-runs the second-level GLM over the 218 per-subject contrast maps and compares the
group t/z/p maps against the published ``results/01_group_analysis/*.nii.gz``.

TWO reasons this is gated harder than the other golden masters:

  1. **Heavy**: fitting a 218-subject `SecondLevelModel` loads the full 4-D stack into
     memory. It is opt-in via ``PERNET_RUN_HEAVY=1`` so a normal `pytest` run never
     triggers it (it can OOM small machines).

  2. **Version-pinned tolerance CALIBRATED (2026-07-06).** `SecondLevelModel` numerics
     are a dataset-specific GLM engine pinned to Pernet's nilearn **0.10.4** (docs/DESIGN.md §2.2).
     The published maps were produced under 0.10.4; the per-voxel tolerance was measured
     on a bigmem node (SLURM job 65405112) under BOTH pinned envs and found bit-identical:
     worst max|Δ| = 1.11e-15 (machine epsilon) across 0.10.4 and 0.12.1. `ATOL` is now
     frozen at 1e-13 and `test_tzp_within_tolerance` is enabled.

DATA-GATED: needs <root>/00_volumetric_GLM/sub*/*_contrast_estimates.nii.gz and the
golden <root>/01_group_analysis/*.nii.gz. Point it with PERNET_RESULTS_ROOT.
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
        if (root / "00_volumetric_GLM" / "sub001_Ed" / "sub001_Ed_contrast_estimates.nii.gz").exists() \
                and (root / "01_group_analysis" / "t_map.nii.gz").exists():
            return root
    return None


def _load_driver():
    path = _DATASET / "analysis" / "01_group_analysis.py"
    spec = importlib.util.spec_from_file_location("pernet_01_group_analysis", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pytestmark = [
    pytest.mark.skipif(not _has_nilearn(), reason="nilearn not installed"),
    pytest.mark.skipif(not _RUN_HEAVY, reason="heavy 218-subject fit; set PERNET_RUN_HEAVY=1"),
]


@pytest.fixture(scope="module")
def result_and_gold():
    root = _resolve_root()
    if root is None:
        pytest.skip("00 contrast maps + golden 01 maps not found (set PERNET_RESULTS_ROOT)")
    import nibabel as nib

    driver = _load_driver()
    maps, clusters, summary = driver.compute(root)
    gold = {name: nib.load(str(root / "01_group_analysis" / f"{name}.nii.gz")).get_fdata()
            for name in ("t_map", "z_map", "p_map")}
    got = {name: maps[name].get_fdata() for name in ("t_map", "z_map", "p_map")}
    return got, gold, clusters, summary


@pytest.mark.parametrize("name", ["t_map", "z_map", "p_map"])
def test_map_shape_and_correlation(result_and_gold, name):
    got, gold, _, _ = result_and_gold
    a, b = got[name], gold[name]
    assert a.shape == b.shape
    m = np.isfinite(a) & np.isfinite(b)
    # Version-stable invariant: the group maps are essentially identical in structure.
    assert np.corrcoef(a[m], b[m])[0, 1] > 0.9999


def test_cluster_count_and_peak(result_and_gold):
    _, gold, clusters, summary = result_and_gold
    assert summary["n_clusters"] == 4                      # published: 4 FWE clusters
    assert clusters["Peak Stat"].max() == pytest.approx(15.76, abs=0.1)


def test_tzp_within_tolerance(result_and_gold):
    got, gold, _, _ = result_and_gold
    # Calibrated 2026-07-06 on bigmem (SLURM job 65405112), both pinned envs:
    # worst per-voxel max|Δ| = 1.11e-15 (machine epsilon) and BIT-IDENTICAL across
    # nilearn 0.10.4 (py3.8/numpy1.22.4) and 0.12.1 (py3.9/numpy1.26.4). Frozen at
    # 1e-13 (~90x headroom). See docs/DESIGN.md "DONE — heavy golden-master calibration".
    ATOL = 1e-13
    for name in ("t_map", "z_map", "p_map"):
        a, b = got[name], gold[name]
        m = np.isfinite(a) & np.isfinite(b)
        assert np.allclose(a[m], b[m], atol=ATOL)
