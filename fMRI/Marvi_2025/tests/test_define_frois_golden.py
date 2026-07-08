"""Golden master for fROI definition — Branch A step `define_frois` (docs/DESIGN.md §6 Tier 1).

Re-defines functional ROIs (top-10% z within each anatomical parcel) from the *published*
split z-maps and asserts each reproduces the published on-disk fROI mask
(`derivatives/<subj>/frois/<cat>_<parcel>/..._split-{even,odd}_froi.nii.gz`).

The selection is pure numpy (`np.percentile` + boolean mask over the split-averaged
z-map); the one version-sensitive step is nilearn `resample_to_img(parcel, ..., 'nearest')`.
Under the pinned env (nilearn 0.12.1, matching the dev pipeline) the resample is identical,
so the masks reproduce **bitwise** — we assert Dice == 1.0 AND exact voxel-count equality.
The Dice tolerance is calibrated-then-frozen (docs/DESIGN.md §6): should nilearn drift, the mask
overlap degrades gracefully and DICE_MIN documents the accepted floor.

By feeding the published z-maps, the test never re-runs the first-level GLM — it isolates
the fROI-definition boundary (the same decoupling Pernet's cv_05 golden uses).

DATA-GATED (nibabel + nilearn + numpy + the published derivatives cut): needs
<deriv>/<subj>/first_level_glm/effloc_*_split-*/run-*/..._zmap.nii.gz, the anatomical
parcels (emfl.config.get_parcels_dir → vendored data/PARCELS), and the published
<deriv>/<subj>/frois/ masks. Skips cleanly when unavailable. Point the derivatives root
with MARVI_DERIVATIVES_ROOT.
"""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

_DATASET = Path(__file__).resolve().parent.parent
if str(_DATASET) not in sys.path:
    sys.path.insert(0, str(_DATASET))

_CANDIDATE_DERIV_ROOTS = [
    os.environ.get("MARVI_DERIVATIVES_ROOT"),
    "/work/upschrimpf1/mehrer/datasets/Marvi_2025_efficient_fMRI_localizer/derivatives",
]

# Subject used for the golden master (has the full published GLM + fROI cut).
SUBJECT = "sub-kaneff01"

# Representative parcels — one per (category, hemisphere-pattern) so every code path is
# exercised: bilateral L+R, midline single mask (None), and left-only. The fROI algorithm
# is parcel-agnostic, so this sample fully characterizes it without a multi-minute sweep.
#   (category, parcel, hemisphere)
SAMPLE_SPECS = [
    ("julian", "ffa", "lh"),     # bilateral (ventral visual)
    ("julian", "ffa", "rh"),
    ("language", "ag", "lh"),    # bilateral (language)
    ("language", "ag", "rh"),
    ("md", "insula", "lh"),      # bilateral (multiple-demand)
    ("md", "insula", "rh"),
    ("tom", "tpj", "lh"),        # bilateral (theory-of-mind)
    ("tom", "tpj", "rh"),
    ("tom", "pc", None),         # midline / single bilateral mask
    ("speech", "speech", None),  # single bilateral mask
    ("vwfa", "vwfa", "lh"),      # left-only
]
SPLITS = ("even", "odd")

# Frozen tolerance: matched-env resample reproduces the published masks bitwise
# (calibrated Dice == 1.0). Kept as a floor so a nilearn drift surfaces as a soft failure.
DICE_MIN = 1.0


def _has_deps():
    try:
        import nibabel  # noqa: F401
        import nilearn  # noqa: F401
        import numpy  # noqa: F401
        return True
    except ImportError:
        return False


def _load_driver():
    path = _DATASET / "analysis" / "define_frois.py"
    spec = importlib.util.spec_from_file_location("marvi_define_frois", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _resolve_deriv_root(driver):
    """First candidate root that has both a published z-map and a published fROI mask."""
    for root in _CANDIDATE_DERIV_ROOTS:
        if not root:
            continue
        root = Path(root)
        published = driver.froi_output_path(root, SUBJECT, "julian", "ffa", "lh", "even")
        glm_dir = root / SUBJECT / "first_level_glm"
        if published.exists() and glm_dir.exists():
            return root
    return None


pytestmark = pytest.mark.skipif(not _has_deps(), reason="nibabel + nilearn + numpy not installed")


def _dice(a_bool, b_bool):
    import numpy as np

    a, b = a_bool.astype(bool), b_bool.astype(bool)
    denom = int(a.sum()) + int(b.sum())
    if denom == 0:
        return 1.0  # both empty → identical
    return 2.0 * int(np.logical_and(a, b).sum()) / denom


@pytest.fixture(scope="module")
def context():
    driver = _load_driver()
    root = _resolve_deriv_root(driver)
    if root is None:
        pytest.skip("published derivatives cut (z-maps + frois/) not found "
                    "(set MARVI_DERIVATIVES_ROOT)")
    from emfl.config import get_parcels_dir
    from emfl.roi import fROIDefiner

    parcels_dir = get_parcels_dir()
    definer = fROIDefiner(
        parcels_dir=parcels_dir,
        derivatives_dir=root,
        subject_id=SUBJECT,
        space=driver.DEFAULT_SPACE,
    )
    return driver, root, definer


@pytest.mark.parametrize("cat,parcel,hemi", SAMPLE_SPECS,
                         ids=[f"{c}_{p}_{h or 'bilat'}" for c, p, h in SAMPLE_SPECS])
@pytest.mark.parametrize("split", SPLITS)
def test_froi_reproduces_published_mask(context, cat, parcel, hemi, split):
    import nibabel as nib

    driver, root, definer = context

    published_path = driver.froi_output_path(root, SUBJECT, cat, parcel, hemi, split)
    if not published_path.exists():
        pytest.skip(f"published mask missing: {published_path.name}")

    froi_img, meta = definer.define_froi(
        parcel_category=cat, parcel_name=parcel, hemisphere=hemi,
        percentile=driver.DEFAULT_FROI_PERCENTILE, run_split=split,
    )
    got = froi_img.get_fdata() > 0
    gold = nib.load(str(published_path)).get_fdata() > 0

    # Exact voxel-count parity (numpy selection is deterministic under matched nilearn).
    assert int(got.sum()) == int(gold.sum()), (
        f"voxel count {int(got.sum())} != published {int(gold.sum())} "
        f"for {cat}/{parcel}/{hemi}/{split}")
    # And identical placement.
    dice = _dice(got, gold)
    assert dice >= DICE_MIN, f"Dice {dice:.6f} < {DICE_MIN} for {cat}/{parcel}/{hemi}/{split}"
