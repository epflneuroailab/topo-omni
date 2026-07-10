"""Unit tests for core.surface (docs/DESIGN.md §6 Tier 2).

Self-contained: a synthetic volume + a hand-built mesh (no fsaverage fetch, no real
data). Needs nilearn for the projection itself (skipped if absent). The heavy
end-to-end golden master (published Pernet npz reproduced bitwise) lives data-gated in
Pernet_2015/tests/test_surface_projection_golden.py.
"""
import numpy as np
import pytest

from core import surface as surf


def _has_nilearn():
    try:
        import nilearn  # noqa: F401
        return True
    except ImportError:
        return False


requires_nilearn = pytest.mark.skipif(not _has_nilearn(), reason="nilearn not installed")


def test_projection_kwargs_are_frozen():
    # The golden master depends on these exact values (atol=0). Guard against drift.
    assert surf.PROJECTION_KWARGS == {"radius": 0, "kind": "auto", "n_samples": 1}


def test_binarize_sulcal_map():
    sulc = np.array([-2.0, -1e-9, 0.0, 1e-9, 5.0])
    out = surf.binarize_sulcal_map(sulc)
    # sulci (>0) -> gray; gyri and zero (<=0) -> white
    assert np.allclose(out, [1.0, 1.0, 1.0, 179 / 255, 179 / 255])


@requires_nilearn
def test_black_viridis_zero_is_black():
    cmap = surf.black_viridis_cmap()
    r, g, b, a = cmap(0.0)
    assert (r, g, b) == (0.0, 0.0, 0.0)


@requires_nilearn
def test_vol_to_surf_samples_constant_volume():
    import nibabel as nib

    # Constant-value volume: every in-FOV vertex must project to that constant.
    affine = np.eye(4)
    vol = np.full((10, 10, 10), 7.0, dtype=np.float64)
    img = nib.Nifti1Image(vol, affine)

    # A few vertices well inside the FOV (voxel == world coords under identity affine).
    coords = np.array([[3.0, 3.0, 3.0], [5.0, 5.0, 5.0], [6.0, 4.0, 7.0]])
    faces = np.array([[0, 1, 2]])  # one triangle; enough for a valid mesh

    out = surf.vol_to_surf(img, (coords, faces))
    assert out.shape == (3,)
    assert np.allclose(out, 7.0)


@requires_nilearn
def test_vol_to_surf_overrides_reach_nilearn():
    # Deliberate deviation is possible without touching the frozen defaults.
    import nibabel as nib

    img = nib.Nifti1Image(np.full((8, 8, 8), 3.0), np.eye(4))
    coords = np.array([[4.0, 4.0, 4.0], [3.0, 3.0, 3.0], [5.0, 5.0, 5.0]])
    faces = np.array([[0, 1, 2]])
    out = surf.vol_to_surf(img, (coords, faces), n_samples=2)
    assert out.shape == (3,)
    assert np.allclose(out, 3.0)
