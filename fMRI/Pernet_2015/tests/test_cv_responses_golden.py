"""Golden master for the cross-validated fROI profile — step cv_05 (docs/DESIGN.md §6 Tier 1).

Re-extracts the per-subject cross-validated responses (mean beta in the opposite fold's
fROI mask) and asserts they reproduce the published ``cv_responses.csv`` **exactly**.

Extraction is pure nibabel + numpy (mean-in-mask) — NO nilearn — so it is bitwise
version-robust. Calibration (measure->freeze, docs/DESIGN.md §6): reproduces the published
values to max|Δ| ≈ 8e-17 (~1 ULP, from `nanmean` summation order); frozen at
`atol=1e-12`. This decouples cleanly from cv_04's version-pinned GLM: the test feeds the
*published* fROI masks, so it never re-runs the second-level fit.

DATA-GATED (nibabel + pandas + the published CV cut): needs
<root>/04_cross_validation/group/half-{A,B}_fROI_mask.nii.gz, the per-subject
half-split betas, and the golden <root>/04_cross_validation/cv_responses.csv. Skips
cleanly when unavailable. Point it with PERNET_RESULTS_ROOT.
"""
import importlib.util
import os
from pathlib import Path

import pytest

_DATASET = Path(__file__).resolve().parent.parent
_CANDIDATE_ROOTS = [
    os.environ.get("PERNET_RESULTS_ROOT"),
    "/work/upschrimpf1/mehrer/code/20241003_pernet_2015/results",
]

# Frozen tolerance (calibrated max|Δ| ≈ 8e-17; see module docstring).
ATOL = 1e-12


def _has_deps():
    try:
        import nibabel  # noqa: F401
        import pandas  # noqa: F401
        return True
    except ImportError:
        return False


def _resolve_root():
    for root in _CANDIDATE_ROOTS:
        if not root:
            continue
        root = Path(root)
        cv = root / "04_cross_validation"
        if (cv / "group" / "half-A_fROI_mask.nii.gz").exists() \
                and (cv / "cv_responses.csv").exists() \
                and (cv / "per_subject" / "sub001_Ed" / "half-A_vocal_beta.nii.gz").exists():
            return root
    return None


def _load_driver():
    path = _DATASET / "analysis" / "cv_05_extract_responses_and_plot.py"
    spec = importlib.util.spec_from_file_location("pernet_cv_05", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pytestmark = pytest.mark.skipif(not _has_deps(), reason="nibabel + pandas not installed")


@pytest.fixture(scope="module")
def result_and_gold():
    root = _resolve_root()
    if root is None:
        pytest.skip("published CV cut (masks + betas + cv_responses.csv) not found "
                    "(set PERNET_RESULTS_ROOT)")
    import pandas as pd

    driver = _load_driver()
    got = driver.extract_responses(root).set_index("subject").sort_index()
    gold = pd.read_csv(str(root / "04_cross_validation" / "cv_responses.csv")) \
        .set_index("subject").sort_index()
    return got, gold


def test_same_subjects(result_and_gold):
    got, gold = result_and_gold
    assert list(got.index) == list(gold.index)
    assert len(got) == len(gold) == 218


@pytest.mark.parametrize("column", ["vocal_beta", "nonvocal_beta"])
def test_responses_reproduced_exactly(result_and_gold, column):
    import numpy as np

    got, gold = result_and_gold
    a = got[column].to_numpy()
    b = gold.loc[got.index, column].to_numpy()
    # Version-robust extraction: exact to a floating-point ULP (frozen atol=1e-12).
    assert np.allclose(a, b, atol=ATOL, rtol=0)
