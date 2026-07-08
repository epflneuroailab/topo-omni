"""Golden master for Fig. 3b surface projection — step 02 (docs/DESIGN.md §6 Tier 1).

Projects the group maps with `core.surface` and asserts the surface arrays reproduce
the published ``surface_data_fsaverage6.npz`` **exactly**. Calibration (measure->freeze,
docs/DESIGN.md §6): under nilearn 0.12.1 reproducing a 0.10.4-produced reference, every array
matched to max|Δ| = 0.0 with identical NaN patterns — the `vol_to_surf` projection is
bitwise version-robust, so this is asserted at exact equality, not a tolerance.

DATA-GATED: needs <root>/01_group_analysis/{t_map,z_map,thresholded_t_map}.nii.gz plus
the golden <root>/02_surface_projection/surface_data_fsaverage6.npz, and nilearn +
fsaverage6. Skips cleanly when unavailable (always-on net: core/tests/test_surface.py).
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

# npz keys the projection produces, per hemisphere.
_KEYS = ["t_map", "z_map", "thresholded"]


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
        tmap = root / "01_group_analysis" / "t_map.nii.gz"
        npz = root / "02_surface_projection" / "surface_data_fsaverage6.npz"
        if tmap.exists() and npz.exists():
            return root
    return None


def _load_driver():
    path = _DATASET / "analysis" / "02_surface_projection.py"
    spec = importlib.util.spec_from_file_location("pernet_02_surface_projection", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pytestmark = pytest.mark.skipif(not _has_nilearn(), reason="nilearn not installed")


@pytest.fixture(scope="module")
def result_and_gold():
    root = _resolve_root()
    if root is None:
        pytest.skip("01 maps + golden npz not found (set PERNET_RESULTS_ROOT)")
    driver = _load_driver()
    result = driver.compute(root)
    gold = np.load(str(root / "02_surface_projection" / "surface_data_fsaverage6.npz"))
    return result, gold


@pytest.mark.parametrize("key", _KEYS)
@pytest.mark.parametrize("hemi", ["lh", "rh"])
def test_surface_array_reproduced_exactly(result_and_gold, key, hemi):
    result, gold = result_and_gold
    got = result[key][hemi]
    ref = gold[f"{key}_{hemi}"]
    assert got.shape == ref.shape == (40962,)
    # Exact reproduction, NaN-aware (calibrated max|Δ| = 0.0).
    assert np.array_equal(np.isnan(got), np.isnan(ref))
    assert np.array_equal(got[~np.isnan(got)], ref[~np.isnan(ref)])
