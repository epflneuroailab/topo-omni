"""Fig. 1b parcel panel (analysis/plot_figure_1b_parcels.py).

Pure-numpy parts (contour tracer, colormap layout, region selection) are unit-tested on a
hand-built mesh. The label maps are pinned by per-region vertex counts on fsaverage6: at the
port (2026-09-24) the maps were identical, vertex for vertex, to the dev script's
(20251030_Marvi_2025_efficient_fMRI_localizer/src/visualize_roi_parcels_surface.py), and these
counts come from that run. Overlaps are resolved by ROI_DEFS order, so LOC takes one left
scenes vertex (483 -> 482).

DATA-GATED: the count test needs nilearn and the fsaverage6 surfaces already in the nilearn
data dir (it does not download them).
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_DATASET = Path(__file__).resolve().parent.parent
if str(_DATASET) not in sys.path:
    sys.path.insert(0, str(_DATASET))


def _has_deps():
    try:
        import matplotlib  # noqa: F401
        import nibabel  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _has_deps(), reason="matplotlib + nibabel not installed")


def _load():
    path = _DATASET / "analysis" / "plot_figure_1b_parcels.py"
    spec = importlib.util.spec_from_file_location("marvi_plot_figure_1b_parcels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Frozen vertex counts per region on fsaverage6: {tag: {hemi: {region: n}}}.
GOLDEN_COUNTS = {
    "scenes-loc-vwfa-faces": {
        "L": {"scenes": 482, "loc": 744, "vwfa": 572, "faces": 327},
        "R": {"scenes": 362, "loc": 512, "vwfa": 0, "faces": 630},
    },
    "scenes-vwfa-faces": {
        "L": {"scenes": 483, "vwfa": 572, "faces": 327},
        "R": {"scenes": 362, "vwfa": 0, "faces": 630},
    },
}


def test_select_rois_renumbers_after_dropping_loc():
    m = _load()
    full = m.select_rois(include_loc=True)
    assert m.roi_tag(full) == "scenes-loc-vwfa-faces"
    assert [r["value"] for r in full] == [1, 2, 3, 4]
    no_loc = m.select_rois(include_loc=False)
    assert m.roi_tag(no_loc) == "scenes-vwfa-faces"
    assert [r["value"] for r in no_loc] == [1, 2, 3]
    assert m.ROI_DEFS[1]["value"] == 2  # the module-level list is not mutated


def test_region_colours_follow_the_palette():
    m = _load()
    by_name = {r["name"]: r["color"] for r in m.ROI_DEFS}
    assert by_name == {
        "scenes": m.CATEGORY_COLORS["scenes"],
        "loc": m.CATEGORY_COLORS["objects"],
        "vwfa": m.CATEGORY_COLORS["words"],
        "faces": m.CATEGORY_COLORS["faces"],
    }


def test_discrete_cmap_maps_each_label_to_its_colour_and_contours_to_black():
    from matplotlib.colors import to_hex
    m = _load()
    rois = m.select_rois(include_loc=True)
    cmap, vmin, vmax, contour_value = m.make_discrete_cmap(rois)
    assert contour_value == len(rois) + 1
    for r in rois:
        frac = (r["value"] - vmin) / (vmax - vmin)
        assert to_hex(cmap(frac)).lower() == r["color"].lower()
    assert to_hex(cmap((contour_value - vmin) / (vmax - vmin))) == "#000000"


def test_contours_mark_region_vertices_on_mixed_triangles_only():
    m = _load()
    # Strip of 3 triangles over vertices 0..4: (0,1,2) (1,2,3) (2,3,4).
    faces = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])
    roi_map = np.array([1.0, 1.0, 1.0, np.nan, np.nan])
    contour = m.compute_roi_contours(roi_map, faces)
    # Triangle (0,1,2) is uniform; (1,2,3) and (2,3,4) mix label 1 with background.
    # Vertices 1 and 2 are region vertices on a mixed triangle; 3 and 4 are background.
    assert contour.tolist() == [False, True, True, False, False]


def test_every_parcel_resolves_in_the_vendored_parcels():
    m = _load()
    parcels = _DATASET / "data" / "PARCELS"
    for roi in m.ROI_DEFS:
        for name in roi["parcels_L"] + roi["parcels_R"]:
            assert m.find_parcel_path(name, parcels) is not None, name
    # The 3D vwfa/ copy wins over the 4D julian/ one.
    assert m.find_parcel_path("lh.vwfa", parcels).parent.name == "vwfa"


def _fsaverage6_cached():
    try:
        from nilearn.datasets import get_data_dirs
    except ImportError:
        return False
    return any((Path(d) / "fsaverage6").is_dir() for d in get_data_dirs())


@pytest.mark.skipif(not _fsaverage6_cached(), reason="nilearn or cached fsaverage6 surfaces unavailable")
@pytest.mark.parametrize("include_loc", [True, False])
def test_label_map_vertex_counts_golden(include_loc):
    import nibabel as nib
    from nilearn import datasets
    m = _load()
    fs = datasets.fetch_surf_fsaverage(mesh="fsaverage6")
    rois = m.select_rois(include_loc=include_loc)
    expected = GOLDEN_COUNTS[m.roi_tag(rois)]
    for hemi, side in (("L", "left"), ("R", "right")):
        n = len(nib.load(fs[f"sulc_{side}"]).darrays[0].data)
        roi_map = m.build_roi_map(rois, fs[f"pial_{side}"], n, hemi, _DATASET / "data" / "PARCELS")
        counts = {r["name"]: int(np.sum(roi_map == r["value"])) for r in rois}
        assert counts == expected[hemi], (hemi, counts)
