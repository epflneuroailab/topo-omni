"""Golden master for fROI cross-validation — Branch A step `cross_validation` (PLAN §6 Tier 1).

Re-runs the split-half reliability analysis (odd-mask -> even-run responses, even-mask ->
odd-run responses, Dice, voxel-wise spatial-pattern correlation, mean preferred-contrast
response) from the *published* fROI masks + split z-maps, and asserts it reproduces the
dev-computed reference:

  * scalar metrics (Dice, spatial correlation + p, mean responses, voxel/response counts)
    are pinned in `tests/fixtures/cross_validation_golden.csv` — a slice of the canonical
    dev run `outputs/cross_validation_details_20251215_135301.csv` for sub-kaneff01;
  * the per-ROI response extractions are compared column-for-column against the *published*
    on-disk `derivatives/<subj>/roi_cross_validation/<subj>_<roi>_{even_from_odd,
    odd_from_even}.csv`.

Dice is pure numpy over the two published masks; the mean responses are mean-in-mask of the
published z-maps (pure numpy). The one version-sensitive step is nilearn
`resample_to_img(parcel, ..., 'nearest')` inside the spatial-pattern correlation. Under the
pinned env (nilearn 0.12.1) every metric reproduces to float tolerance, so FLOAT_ATOL is
calibrated tight and frozen; a nilearn drift surfaces as a soft numeric failure rather than
silently changing the published reliability numbers.

By feeding the published masks + z-maps, the test never re-defines the fROIs or re-runs the
GLM — it isolates the cross-validation boundary (the same decoupling define_frois's golden
uses).

DATA-GATED (nibabel + nilearn + numpy + pandas + scipy + the published derivatives cut):
needs <deriv>/<subj>/{frois,first_level_glm,roi_cross_validation}/ and the anatomical
parcels (emfl.config.get_parcels_dir). Skips cleanly when unavailable. Point the derivatives
root with MARVI_DERIVATIVES_ROOT.
"""
import importlib.util
import math
import os
import sys
from pathlib import Path

import pytest

_DATASET = Path(__file__).resolve().parent.parent
if str(_DATASET) not in sys.path:
    sys.path.insert(0, str(_DATASET))

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "cross_validation_golden.csv"

_CANDIDATE_DERIV_ROOTS = [
    os.environ.get("MARVI_DERIVATIVES_ROOT"),
    "/work/upschrimpf1/mehrer/datasets/Marvi_2025_efficient_fMRI_localizer/derivatives",
]

# Subject used for the golden master (has the full published fROI + CV cut).
SUBJECT = "sub-kaneff01"

# Frozen tolerance: matched-env recompute reproduces the dev reference to float precision.
# Kept as a floor so a nilearn/scipy drift surfaces as a soft numeric failure.
FLOAT_ATOL = 1e-6

# Scalar columns of the details CSV that the driver reproduces (excludes save-time paths).
_SCALAR_COLS = [
    "dice_coefficient", "even_mean_response", "odd_mean_response", "mean_difference",
    "spatial_correlation", "spatial_pval",
]
_COUNT_COLS = [
    "froi_odd_voxels", "responses_even_from_odd", "froi_even_voxels",
    "responses_odd_from_even",
]


def _has_deps():
    try:
        import nibabel  # noqa: F401
        import nilearn  # noqa: F401
        import numpy  # noqa: F401
        import pandas  # noqa: F401
        import scipy  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _has_deps(), reason="nibabel + nilearn + numpy + pandas + scipy not installed")


def _load_driver():
    path = _DATASET / "analysis" / "cross_validation.py"
    spec = importlib.util.spec_from_file_location("marvi_cross_validation", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _resolve_deriv_root():
    """First candidate root with the published fROI masks, GLM z-maps and CV responses."""
    for root in _CANDIDATE_DERIV_ROOTS:
        if not root:
            continue
        root = Path(root)
        subj = root / SUBJECT
        if ((subj / "frois").exists() and (subj / "first_level_glm").exists()
                and (subj / "roi_cross_validation").exists()):
            return root
    return None


def _hemi_of(row):
    """Fixture hemisphere cell -> 'lh'/'rh'/None (NaN for the midline/bilateral masks)."""
    h = row["hemisphere"]
    if isinstance(h, float) and math.isnan(h):
        return None
    return None if h in (None, "", "None") else str(h)


def _close(got, ref, atol=FLOAT_ATOL):
    """Float compare that treats NaN==NaN as equal (dev leaves NaN where a metric is absent)."""
    g_nan = got is None or (isinstance(got, float) and math.isnan(got))
    r_nan = ref is None or (isinstance(ref, float) and math.isnan(ref))
    if g_nan or r_nan:
        return g_nan and r_nan
    return abs(float(got) - float(ref)) <= atol


def _fixture_rows():
    if not _FIXTURE.exists():
        return []
    import pandas as pd
    df = pd.read_csv(_FIXTURE)
    return list(df.to_dict("records"))


_FIXTURE_ROWS = _fixture_rows()


@pytest.fixture(scope="module")
def context():
    if not _FIXTURE_ROWS:
        pytest.skip(f"golden fixture missing: {_FIXTURE}")
    driver = _load_driver()
    root = _resolve_deriv_root()
    if root is None:
        pytest.skip("published derivatives cut (frois/ + first_level_glm/ + "
                    "roi_cross_validation/) not found (set MARVI_DERIVATIVES_ROOT)")
    from emfl.config import get_parcels_dir
    from emfl.roi import CrossValidationAnalyzer

    analyzer = CrossValidationAnalyzer(
        parcels_dir=get_parcels_dir(),
        derivatives_dir=root,
        subject_id=SUBJECT,
        space=driver.DEFAULT_SPACE,
        percentile=driver.DEFAULT_FROI_PERCENTILE,
    )
    return driver, root, analyzer


def _assert_responses_match_disk(driver, root, roi_label, got_df, kind):
    """Compare a recomputed response frame to the published on-disk CSV, row-for-row."""
    import numpy as np
    import pandas as pd

    even_path, odd_path = driver.response_csv_paths(root, SUBJECT, roi_label)
    disk_path = even_path if kind == "even" else odd_path
    if not disk_path.exists():
        pytest.skip(f"published response CSV missing: {disk_path.name}")
    gold = pd.read_csv(disk_path)

    assert len(got_df) == len(gold), (
        f"{roi_label} {kind}: {len(got_df)} responses != published {len(gold)}")

    # `run` is a BIDS zero-padded label ("002") in the recomputed frame; writing it to CSV
    # and reading it back coerces it to int (2) — a round-trip dtype artifact, not a data
    # difference. Normalize both sides to int before keying so the comparison is faithful.
    key = ["run", "modality", "contrast"]
    got_df = got_df.copy(); gold = gold.copy()
    got_df["run"] = got_df["run"].astype(int)
    gold["run"] = gold["run"].astype(int)
    g = got_df.set_index(key).sort_index()
    d = gold.set_index(key).sort_index()
    assert list(g.index) == list(d.index), f"{roi_label} {kind}: response keys differ"

    # numeric response within tolerance, n_voxels exact
    np.testing.assert_allclose(
        g["response"].to_numpy(float), d["response"].to_numpy(float),
        atol=FLOAT_ATOL, err_msg=f"{roi_label} {kind}: response values drift")
    assert list(g["n_voxels"].astype(int)) == list(d["n_voxels"].astype(int)), (
        f"{roi_label} {kind}: n_voxels differ")


@pytest.mark.parametrize(
    "ref", _FIXTURE_ROWS,
    ids=[r["roi_label"] for r in _FIXTURE_ROWS] or ["no_fixture"])
def test_cross_validation_reproduces_reference(context, ref):
    driver, root, analyzer = context

    cat = ref["parcel_category"]
    parcel = ref["parcel_name"]
    hemi = _hemi_of(ref)

    record, resp_even, resp_odd = driver.cross_validate_roi(analyzer, cat, parcel, hemi)

    # Identity / bookkeeping.
    assert record["roi_label"] == ref["roi_label"]

    # Integer counts must match exactly.
    for col in _COUNT_COLS:
        assert int(record[col]) == int(ref[col]), (
            f"{ref['roi_label']}: {col} {record[col]} != reference {ref[col]}")

    # Float reliability metrics within frozen tolerance (NaN==NaN allowed).
    for col in _SCALAR_COLS:
        assert _close(record.get(col), ref[col]), (
            f"{ref['roi_label']}: {col} {record.get(col)} != reference {ref[col]}")

    # Per-ROI response extractions reproduce the published on-disk CSVs.
    _assert_responses_match_disk(driver, root, ref["roi_label"], resp_even, "even")
    _assert_responses_match_disk(driver, root, ref["roi_label"], resp_odd, "odd")
