"""Golden master for the Fig-A2 data — Branch A step `extract_condition_responses` (PLAN §6 Tier 1).

Re-runs the cross-validated per-condition beta extraction (even-mask -> odd-run betas,
odd-mask -> even-run betas, averaged per condition, then (even+odd)/2) from the *published*
fROI masks + split effect maps, and asserts it reproduces the dev-computed reference pinned
in `tests/fixtures/condition_responses_golden.csv` — a 10-ROI x 10-condition slice of
sub-kaneff01 from the canonical dev run
`outputs/condition_responses_details_20260310_203831.csv`.

The slice spans every category/hemisphere pattern: julian/ffa, language/ag, md/insula,
tom/tpj (L+R); speech (bilateral); vwfa (lh-only). Compared columns: even_beta, odd_beta,
mean_beta (+ the modality label).

Every beta is a pure nibabel-load + numpy mean-in-mask of a published effect map on the
fROI's own grid — no resampling, no nilearn — so the table reproduces to float-summation
precision. FLOAT_ATOL is therefore calibrated tight and frozen (the Pernet cv_05 regime): a
numeric drift trips a soft failure rather than silently changing the published Fig-A2 values.

By feeding the published masks + effect maps, the test never re-defines the fROIs or re-runs
the GLM — it isolates the extraction boundary (the same decoupling define_frois's and
cross_validation's goldens use). Each fixture ROI is recomputed on its own (per-ROI, via the
driver's `froi_mask_paths` + the vendored extractor) so the run stays targeted, not a full
50-ROI subject sweep.

DATA-GATED (nibabel + numpy + pandas + the published derivatives cut): needs
<deriv>/<subj>/{frois,first_level_glm}/. Skips cleanly when unavailable. Point the
derivatives root with MARVI_DERIVATIVES_ROOT.
"""
import importlib.util
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import pytest

_DATASET = Path(__file__).resolve().parent.parent
if str(_DATASET) not in sys.path:
    sys.path.insert(0, str(_DATASET))

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "condition_responses_golden.csv"

_CANDIDATE_DERIV_ROOTS = [
    os.environ.get("MARVI_DERIVATIVES_ROOT"),
    "/work/upschrimpf1/mehrer/datasets/Marvi_2025_efficient_fMRI_localizer/derivatives",
]

# Subject used for the golden master (has the full published fROI + GLM cut).
SUBJECT = "sub-kaneff01"

# Frozen tolerance: mean-in-mask of published effect maps reproduces to summation precision.
# Kept as a floor so a numpy/nibabel drift surfaces as a soft numeric failure.
FLOAT_ATOL = 1e-9

_BETA_COLS = ["even_beta", "odd_beta", "mean_beta"]


def _has_deps():
    try:
        import nibabel  # noqa: F401
        import numpy  # noqa: F401
        import pandas  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _has_deps(), reason="nibabel + numpy + pandas not installed")


def _load_driver():
    path = _DATASET / "analysis" / "extract_condition_responses.py"
    spec = importlib.util.spec_from_file_location("marvi_extract_condition_responses", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _resolve_deriv_root():
    """First candidate root with the published fROI masks + split effect maps."""
    for root in _CANDIDATE_DERIV_ROOTS:
        if not root:
            continue
        root = Path(root)
        subj = root / SUBJECT
        if (subj / "frois").exists() and (subj / "first_level_glm").exists():
            return root
    return None


def _hemi_arg(hemisphere):
    """Fixture hemisphere cell -> the froi_mask_paths arg ('lh'/'rh'/None)."""
    if isinstance(hemisphere, float) and math.isnan(hemisphere):
        return None
    return None if hemisphere in ("bilateral", "", "None", None) else str(hemisphere)


def _recompute_roi(driver, extractor, root, cat, parcel, hemi_arg, space):
    """Cross-validated per-condition betas for ONE ROI (mirrors the driver's inner loop).

    Returns ``{condition: {"even_beta", "odd_beta", "mean_beta", "modality"}}`` or None if
    the split masks are missing on disk.
    """
    even_mask, odd_mask = driver.froi_mask_paths(root, SUBJECT, cat, parcel, hemi_arg, space)
    if not even_mask.exists() or not odd_mask.exists():
        return None
    even_df = extractor.extract_condition_responses(roi_mask_path=even_mask, run_split="odd")
    odd_df = extractor.extract_condition_responses(roi_mask_path=odd_mask, run_split="even")
    out = {}
    for condition in even_df["condition"].unique():
        even_beta = even_df[even_df["condition"] == condition]["beta"].mean()
        odd_beta = odd_df[odd_df["condition"] == condition]["beta"].mean()
        out[condition] = {
            "even_beta": float(even_beta),
            "odd_beta": float(odd_beta),
            "mean_beta": float((even_beta + odd_beta) / 2.0),
            "modality": driver._modality_of(condition),
        }
    return out


def _fixture_by_roi():
    """Group the pinned fixture rows by roi_label -> (cat, parcel, hemi, {condition: row})."""
    if not _FIXTURE.exists():
        return {}
    import pandas as pd
    df = pd.read_csv(_FIXTURE)
    grouped = {}
    for roi_label, sub in df.groupby("roi_label"):
        first = sub.iloc[0]
        grouped[roi_label] = {
            "parcel_category": first["parcel_category"],
            "parcel_name": first["parcel_name"],
            "hemisphere": first["hemisphere"],
            "rows": {r["condition"]: r for _, r in sub.iterrows()},
        }
    return grouped


_FIXTURE_BY_ROI = _fixture_by_roi()


@pytest.fixture(scope="module")
def context():
    if not _FIXTURE_BY_ROI:
        pytest.skip(f"golden fixture missing: {_FIXTURE}")
    driver = _load_driver()
    root = _resolve_deriv_root()
    if root is None:
        pytest.skip("published derivatives cut (frois/ + first_level_glm/) not found "
                    "(set MARVI_DERIVATIVES_ROOT)")
    from emfl.roi import ROIResponseExtractor
    extractor = ROIResponseExtractor(
        derivatives_dir=root, subject_id=SUBJECT, space=driver.DEFAULT_SPACE)
    return driver, root, extractor


@pytest.mark.parametrize(
    "roi_label", sorted(_FIXTURE_BY_ROI.keys()) or ["no_fixture"])
def test_condition_responses_reproduce_reference(context, roi_label):
    driver, root, extractor = context
    spec = _FIXTURE_BY_ROI[roi_label]
    hemi_arg = _hemi_arg(spec["hemisphere"])

    got = _recompute_roi(
        driver, extractor, root, spec["parcel_category"], spec["parcel_name"],
        hemi_arg, driver.DEFAULT_SPACE)
    assert got is not None, f"{roi_label}: published split fROI masks not found on disk"

    ref_rows = spec["rows"]
    # Same set of conditions (all 10 localizer conditions).
    assert set(got.keys()) == set(ref_rows.keys()), (
        f"{roi_label}: conditions differ — got {sorted(got)} vs "
        f"reference {sorted(ref_rows)}")

    for condition, ref in ref_rows.items():
        rec = got[condition]
        assert rec["modality"] == ref["modality"], (
            f"{roi_label}/{condition}: modality {rec['modality']} != {ref['modality']}")
        for col in _BETA_COLS:
            assert abs(rec[col] - float(ref[col])) <= FLOAT_ATOL, (
                f"{roi_label}/{condition}: {col} {rec[col]} != reference {ref[col]} "
                f"(|diff| {abs(rec[col] - float(ref[col])):.3e} > {FLOAT_ATOL:.1e})")
