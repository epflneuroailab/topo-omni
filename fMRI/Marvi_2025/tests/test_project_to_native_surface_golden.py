"""Golden master for Branch B step 09 — volumetric maps -> native surface (PLAN §6 Tier 1).

Re-runs the pial-vertex trilinear projection of the published T1w concatenated-GLM stat
maps (`concatenated_glm/…_concat_space-T1w_<map>.nii.gz`) and asserts it reproduces the
published surface projections (`native_surface_projections/…_hemi-{L,R}_<map>.func.gii`)
**bitwise**.

Why bitwise: the projection is pure numpy + `scipy.ndimage.map_coordinates(order=1)` — no
nilearn, no FreeSurfer CLI, no resampling of the source grid. Calibration (2026-07-07, base
python: numpy 1.26.4 / nibabel 5.3.2 / scipy 1.11.1) measured **max|Δ| = 0.0** across
sub-kaneff01 × {visual,auditory} × {faces_vs_objects, objects_vs_words,
false_belief_vs_false_photo, math_vs_theory_of_mind} × {tmap, pval, signed_log_p} × {L, R}
(24 maps). Tolerance is therefore frozen at ATOL = 0.0 (exact), the Pernet step-02 regime.

Feeding the *published* GLM maps + *published* pial surfaces isolates the projection
boundary — the test never re-runs the concatenated GLM (dev 08, heavy, not golden-mastered).

DATA-GATED (nibabel + numpy + scipy + the published derivatives cut): needs
<deriv>/<subj>/anat/<subj>_hemi-*_pial.surf.gii, <deriv>/concatenated_glm/, and
<deriv>/native_surface_projections/. Skips cleanly when unavailable. Point the derivatives
root with MARVI_DERIVATIVES_ROOT.
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

SUBJECT = "sub-kaneff01"

# Frozen: the projection reproduces the published .func.gii exactly (calibrated max|Δ| = 0.0).
ATOL = 0.0

# Representative sample: both modalities, both hemi-orderings, all 3 map types, and
# contrasts spanning the visual/auditory contrast families.
CASES = [
    ("visual", "faces_vs_objects"),
    ("visual", "objects_vs_words"),
    ("auditory", "false_belief_vs_false_photo"),
    ("auditory", "math_vs_theory_of_mind"),
]
MAP_TYPES = ("tmap", "pval", "signed_log_p")
HEMIS = ("L", "R")


def _has_deps():
    try:
        import nibabel  # noqa: F401
        import numpy  # noqa: F401
        import scipy  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _has_deps(), reason="nibabel + numpy + scipy not installed")


def _load_driver():
    path = _DATASET / "analysis" / "project_to_native_surface.py"
    spec = importlib.util.spec_from_file_location("marvi_project_to_native_surface", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _resolve_deriv_root():
    """First candidate root with the published GLM maps + pial surfaces + projections."""
    for root in _CANDIDATE_DERIV_ROOTS:
        if not root:
            continue
        root = Path(root)
        subj = root / SUBJECT
        if (
            (subj / "anat").exists()
            and (root / "concatenated_glm" / SUBJECT).exists()
            and (root / "native_surface_projections" / SUBJECT).exists()
        ):
            return root
    return None


@pytest.fixture(scope="module")
def context():
    driver = _load_driver()
    root = _resolve_deriv_root()
    if root is None:
        pytest.skip(
            "published Branch-B cut (concatenated_glm/ + native_surface_projections/ + "
            "anat pial) not found (set MARVI_DERIVATIVES_ROOT)")
    return driver, root


@pytest.mark.parametrize("modality,contrast", CASES)
@pytest.mark.parametrize("map_type", MAP_TYPES)
def test_projection_reproduces_published(context, modality, contrast, map_type):
    import numpy as np
    import nibabel as nib

    driver, root = context
    glm_dir = root / "concatenated_glm"
    pub_dir = root / "native_surface_projections"

    vol = driver.volume_map_path(glm_dir, SUBJECT, modality, contrast, map_type)
    if not vol.exists():
        pytest.skip(f"published GLM volume missing: {vol.name}")

    checked = 0
    for hemi in HEMIS:
        pial = driver.pial_surface_path(root, SUBJECT, hemi)
        pub = driver.surface_output_path(
            pub_dir, SUBJECT, modality, contrast, hemi, map_type)
        if not (pial.exists() and pub.exists()):
            continue

        got = driver.project_volume_to_surface_data(vol, pial)
        ref = np.asarray(nib.load(str(pub)).darrays[0].data, dtype=np.float32)

        assert got.shape == ref.shape, (
            f"{modality}/{contrast}/{map_type}/{hemi}: vertex count "
            f"{got.shape} != published {ref.shape}")
        max_abs = float(np.max(np.abs(got.astype(np.float64) - ref.astype(np.float64))))
        assert max_abs <= ATOL, (
            f"{modality}/{contrast}/{map_type}/{hemi}: max|Δ| {max_abs:.3e} > "
            f"ATOL {ATOL:.1e} vs published projection")
        checked += 1

    assert checked > 0, (
        f"{modality}/{contrast}/{map_type}: no published hemisphere projection found")
