"""Unit tests for the pure-numpy core of Branch B step 12 (parcels -> surface contours).

Step 12 itself is FreeSurfer-CLI heavy (mri_vol2vol / mri_vol2surf) → Stage-0-ish, not
golden-mastered (its parcel routing + CLI flags are validated by faithful-port review + a
provenance spot-check vs the published native_surface_parcels/ when a container is available).
But the *contour tracer* (`create_contour_from_mask` / `contour_from_surface_masks`) is pure
numpy and fully deterministic, so it is unit-tested here on a hand-built mesh (TDD-style — no
FreeSurfer, no dataset). Also checks the parcel-routing tables (transform per subdir + the
multi-subdir overrides) that decide which inverse warp each parcel uses.

DATA-GATED only on numpy.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

_DATASET = Path(__file__).resolve().parent.parent
if str(_DATASET) not in sys.path:
    sys.path.insert(0, str(_DATASET))


def _has_deps():
    try:
        import numpy  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _has_deps(), reason="numpy not installed")


def _load_driver():
    path = _DATASET / "analysis" / "project_parcels_to_surface.py"
    spec = importlib.util.spec_from_file_location("marvi_project_parcels_to_surface", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_create_contour_marks_boundary_triangles():
    import numpy as np

    drv = _load_driver()
    # 5 vertices, 3 triangles in a strip.
    faces = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]], dtype=np.int32)
    # Vertices 0,1 inside the ROI; 2,3,4 outside.
    mask = np.array([1, 1, 0, 0, 0], dtype=np.float32)

    contour = drv.create_contour_from_mask(mask, faces)

    # Triangle [0,1,2] spans a boundary -> marks 0,1,2. [1,2,3] spans -> marks 1,2,3.
    # [2,3,4] is all-outside -> marks nothing. Vertex 4 stays 0.
    assert contour.tolist() == [1, 1, 1, 1, 0]


def test_create_contour_empty_when_uniform():
    import numpy as np

    drv = _load_driver()
    faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
    # Entire mesh inside -> no boundary triangles -> no contour.
    mask = np.ones(4, dtype=np.float32)
    contour = drv.create_contour_from_mask(mask, faces)
    assert contour.sum() == 0


def test_contour_from_surface_masks_thresholds_and_unions():
    import numpy as np

    drv = _load_driver()
    faces = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]], dtype=np.int32)
    n_vertices = 5
    # Two per-parcel surface samples; >0.5 is "inside". Parcel A: vert 0; parcel B: vert 1.
    surf_a = np.array([0.9, 0.1, 0.0, 0.0, 0.0])  # -> vert 0
    surf_b = np.array([0.2, 0.8, 0.3, 0.0, 0.0])  # -> vert 1
    # Union inside = {0,1}; same as the boundary case above.
    contour = drv.contour_from_surface_masks([surf_a, surf_b], faces, n_vertices)
    assert contour.tolist() == [1, 1, 1, 1, 0]


def test_contour_from_surface_masks_empty_returns_zeros():
    import numpy as np

    drv = _load_driver()
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    # Nothing survives the >0.5 threshold -> empty combined mask -> zeros (caller skips).
    surf = np.array([0.1, 0.2, 0.3])
    contour = drv.contour_from_surface_masks([surf], faces, 3)
    assert contour.sum() == 0


def test_parcel_routing_tables_match_dev():
    drv = _load_driver()
    # Transform per subdir (dev 12 SUBDIR_TRANSFORM).
    assert drv.SUBDIR_TRANSFORM["julian"] == "cvs"
    assert drv.SUBDIR_TRANSFORM["vwfa"] == "cvs_mni"
    assert drv.SUBDIR_TRANSFORM["md"] == "cvs_mni"
    assert drv.SUBDIR_TRANSFORM["language"] == "affine"
    assert drv.SUBDIR_TRANSFORM["speech"] == "affine"
    assert drv.SUBDIR_TRANSFORM["tom"] == "affine"
    # Multi-subdir overrides (dev 12 PARCEL_SUBDIR_OVERRIDE).
    assert drv.PARCEL_SUBDIR_OVERRIDE["lh.vwfa"] == "vwfa"   # CVS-MNI152, not julian/CVS
    assert drv.PARCEL_SUBDIR_OVERRIDE["rh.sts"] == "julian"  # face-STS, not tom
    # words_vs_objects has no rh.vwfa; false_belief has only rh.tpj (L empty).
    assert drv.CONTRAST_PARCELS["words_vs_objects"]["R"] == []
    assert drv.CONTRAST_PARCELS["false_belief_vs_false_photo"]["L"] == []


def test_find_parcel_file_honours_override(tmp_path):
    drv = _load_driver()
    # Build a fake parcels tree where lh.vwfa exists in BOTH julian/ and vwfa/.
    for sub in ("julian", "vwfa"):
        (tmp_path / sub).mkdir(parents=True)
        (tmp_path / sub / "lh.vwfa.nii.gz").write_bytes(b"")
    path, subdir = drv.find_parcel_file("lh.vwfa", tmp_path)
    # Override forces vwfa/ (CVS-MNI152), not the search-order julian/.
    assert subdir == "vwfa"
    assert path == tmp_path / "vwfa" / "lh.vwfa.nii.gz"
