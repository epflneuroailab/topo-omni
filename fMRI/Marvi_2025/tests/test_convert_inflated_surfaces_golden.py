"""Golden master for Branch B step 11 — FreeSurfer inflated -> GIFTI (PLAN §6 Tier 1).

Re-runs the `nibabel.freesurfer.read_geometry` -> GIFTI conversion on the shipped
FreeSurfer recon-all output (`sourcedata/freesurfer/<subj>/surf/{lh,rh}.inflated`) and
asserts it reproduces the published inflated display meshes
(`<subj>/anat/<subj>_hemi-{L,R}_inflated.surf.gii`) **bitwise** — both the POINTSET
(vertex coords) and TRIANGLE (faces) arrays.

Why bitwise: the conversion is a pure nibabel geometry read + array cast (no FreeSurfer
CLI, no resampling). Calibration (2026-07-07, base python: nibabel 5.3.2 / numpy 1.26.4)
measured **coord max|Δ| = 0.0 and face max|Δ| = 0** across sub-kaneff01 & sub-kaneff06,
both hemispheres — confirming the published anat GIFTIs were produced by exactly this
converter (not by a different fMRIPrep/mris_convert path). Tolerance frozen at exact.

DATA-GATED (nibabel + numpy + the published cut): needs
<deriv>/sourcedata/freesurfer/<subj>/surf/{lh,rh}.inflated and
<deriv>/<subj>/anat/<subj>_hemi-*_inflated.surf.gii. Skips cleanly when unavailable.
Point the derivatives root with MARVI_DERIVATIVES_ROOT.
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

# Two subjects (differing vertex counts) x both hemispheres.
SUBJECTS = ("sub-kaneff01", "sub-kaneff06")
HEMIS = (("lh", "L"), ("rh", "R"))

# Frozen: the conversion reproduces the published GIFTI exactly (calibrated max|Δ| = 0).
ATOL = 0.0


def _has_deps():
    try:
        import nibabel  # noqa: F401
        import numpy  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _has_deps(), reason="nibabel + numpy not installed")


def _load_driver():
    path = _DATASET / "analysis" / "convert_inflated_surfaces.py"
    spec = importlib.util.spec_from_file_location("marvi_convert_inflated_surfaces", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _resolve_deriv_root():
    for root in _CANDIDATE_DERIV_ROOTS:
        if not root:
            continue
        root = Path(root)
        if (root / "sourcedata" / "freesurfer" / SUBJECTS[0]).exists() and (
            root / SUBJECTS[0] / "anat"
        ).exists():
            return root
    return None


def _pointset_and_triangle(gii):
    """Return (coords, faces) darrays from a GIFTI, matched by intent."""
    import numpy as np

    coords = faces = None
    for d in gii.darrays:
        intent = d.intent if isinstance(d.intent, str) else str(d.intent)
        if "POINTSET" in intent:
            coords = np.asarray(d.data)
        elif "TRIANGLE" in intent:
            faces = np.asarray(d.data)
    # Fallback to positional (dev writes coords then faces).
    if coords is None:
        coords = np.asarray(gii.darrays[0].data)
    if faces is None and len(gii.darrays) > 1:
        faces = np.asarray(gii.darrays[1].data)
    return coords, faces


@pytest.fixture(scope="module")
def context():
    driver = _load_driver()
    root = _resolve_deriv_root()
    if root is None:
        pytest.skip(
            "published Branch-B cut (sourcedata/freesurfer + anat inflated GIFTIs) "
            "not found (set MARVI_DERIVATIVES_ROOT)")
    return driver, root


@pytest.mark.parametrize("subject", SUBJECTS)
@pytest.mark.parametrize("hemi_fs,hemi_bids", HEMIS)
def test_inflated_conversion_reproduces_published(context, subject, hemi_fs, hemi_bids):
    import numpy as np
    import nibabel as nib

    driver, root = context
    freesurfer_dir = root / "sourcedata" / "freesurfer"

    fs_path = driver.freesurfer_inflated_path(freesurfer_dir, subject, hemi_fs)
    pub_path = driver.inflated_output_path(root, subject, hemi_bids)
    if not (fs_path.exists() and pub_path.exists()):
        pytest.skip(f"missing {subject} {hemi_bids}: fs={fs_path.exists()} pub={pub_path.exists()}")

    got = driver.inflated_gifti_from_freesurfer(fs_path)
    got_coords = np.asarray(got.darrays[0].data)
    got_faces = np.asarray(got.darrays[1].data)

    ref_coords, ref_faces = _pointset_and_triangle(nib.load(str(pub_path)))

    assert got_coords.shape == ref_coords.shape, (
        f"{subject} {hemi_bids}: coord shape {got_coords.shape} != {ref_coords.shape}")
    assert got_faces.shape == ref_faces.shape, (
        f"{subject} {hemi_bids}: face shape {got_faces.shape} != {ref_faces.shape}")

    coord_d = float(np.max(np.abs(got_coords.astype(np.float64) - ref_coords.astype(np.float64))))
    face_d = int(np.max(np.abs(got_faces.astype(np.int64) - ref_faces.astype(np.int64))))
    assert coord_d <= ATOL, f"{subject} {hemi_bids}: coord max|Δ| {coord_d:.3e} > {ATOL:.1e}"
    assert face_d == 0, f"{subject} {hemi_bids}: face max|Δ| {face_d} != 0"
