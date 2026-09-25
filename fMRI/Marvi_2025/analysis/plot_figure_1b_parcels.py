#!/usr/bin/env python3
"""Fig. 1b (top) — the EMFL visual parcels as filled regions on the fsaverage6 surface.

The human brain panel of Figure 1b ("Visual clusters on the cortical surface, defined by the
EMFL localizer"). Anatomical group parcels (MNI volumes, vendored in ``data/PARCELS/``) are
projected to fsaverage6 with ``nilearn.surface.vol_to_surf``, merged into one integer label
map, and drawn as flat colours over a binarized sulcal background, with black contours at the
region boundaries. No subject data and no t-maps: the panel is atlas-only, so it needs neither
the precomputed cut nor a GLM.

Regions (``ROI_DEFS`` order; where two overlap, the later one wins):
  1  Scenes — PPA + OPA + RSC (bilateral)
  2  LOC    — LOC (bilateral; dropped with --no-loc)
  3  VWFA   — VWFA (left only)
  4  Faces  — FFA + OFA + STS (bilateral)

Colours are the visual family of the Topo-Omni palette, in the version of 2026-09-25
(Badr AlKhamissi's swatch sheet: warm earth tones for the visual categories, teal -> navy for
the non-visual ones). Only the visual family is used here. Bodies (#BF8649) has no region.
Earlier versions: faces #E64B35, scenes #F1C232, objects #EFC94C then #A0522D, words #D4AC0D
then #9C6B00.

Port of the dev script ``20251030_Marvi_2025_efficient_fMRI_localizer/src/
visualize_roi_parcels_surface.py`` (untracked in that repo; copied 2026-09-24). Changes: the
paths are arguments (the parcels resolve via ``emfl.config.get_parcels_dir``), ``--no-contours``
restores the contour-less variant the dev script used to write, and the colours follow the
palette above. The label maps are identical to the dev script's (checked 2026-09-24).

Writes into --output-dir:
  roi_parcels_<tag>_{lh,rh}_{inflated,pial}.html  and  roi_parcels_<tag>_legend.png
with <tag> the active regions, e.g. ``scenes-loc-vwfa-faces``.

Render step — not golden-mastered; the label maps are pinned by vertex counts in
tests/test_plot_figure_1b_parcels.py. Requires nilearn (pinned 0.12.1), which fetches the
fsaverage6 surfaces on first use.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402

_DATASET_DIR = Path(__file__).resolve().parent.parent
if str(_DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(_DATASET_DIR))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Visual family of the Topo-Omni palette, keyed by EMFL condition. The one place to edit when
# the palette changes.
CATEGORY_COLORS = {
    'faces': '#A04D3B',    # was #E64B35
    'bodies': '#BF8649',   # was #F39B2F; no bodies region in this panel
    'scenes': '#CDB34C',   # was #F1C232
    'objects': '#E7DFA1',  # was #A0522D (before that #EFC94C)
    'words': '#877C26',    # was #9C6B00 (before that #D4AC0D)
}

# value: integer label in the surface map (1-based, consecutive; reassigned after --no-loc).
ROI_DEFS = [
    {
        'name': 'scenes',
        'label': 'Scenes (PPA/OPA/RSC)',
        'parcels_L': ['lh.ppa', 'lh.opa', 'lh.rsc'],
        'parcels_R': ['rh.ppa', 'rh.opa', 'rh.rsc'],
        'color': CATEGORY_COLORS['scenes'],
        'value': 1,
        'optional': False,
    },
    {
        'name': 'loc',
        'label': 'LOC',
        'parcels_L': ['lh.loc'],
        'parcels_R': ['rh.loc'],
        'color': CATEGORY_COLORS['objects'],
        'value': 2,
        'optional': True,
    },
    {
        'name': 'vwfa',
        'label': 'VWFA',
        'parcels_L': ['lh.vwfa'],
        'parcels_R': [],
        'color': CATEGORY_COLORS['words'],
        'value': 3,
        'optional': False,
    },
    {
        'name': 'faces',
        'label': 'Faces (FFA/OFA/STS)',
        'parcels_L': ['lh.ffa', 'lh.ofa', 'lh.sts'],
        'parcels_R': ['rh.ffa', 'rh.ofa', 'rh.sts'],
        'color': CATEGORY_COLORS['faces'],
        'value': 4,
        'optional': False,
    },
]

# 'vwfa' before 'julian': the vwfa/ copy is 3D, julian/lh.vwfa.nii.gz is 4D (256,256,256,1).
PARCEL_SUBDIRS = ['vwfa', 'julian', 'language', 'tom', 'md', 'speech']


def select_rois(include_loc: bool = True) -> list:
    """ROI_DEFS without the optional regions if requested, values renumbered 1..n."""
    rois = [r for r in ROI_DEFS if include_loc or not r['optional']]
    return [{**r, 'value': i} for i, r in enumerate(rois, start=1)]


def roi_tag(roi_defs: list) -> str:
    """Filename tag naming the active regions, e.g. 'scenes-loc-vwfa-faces'."""
    return '-'.join(r['name'] for r in roi_defs)


def find_parcel_path(parcel_name: str, parcels_dir: Path):
    for subdir in PARCEL_SUBDIRS:
        p = Path(parcels_dir) / subdir / f'{parcel_name}.nii.gz'
        if p.exists():
            return p
    return None


def binarize_sulcal_map(sulc_data: np.ndarray) -> np.ndarray:
    """Sulci (positive) -> 179/255 gray, gyri -> 1.0 white."""
    return np.where(sulc_data > 0, 179 / 255, 1.0)


def build_roi_map(roi_defs: list, surf_path, n_vertices: int, hemi: str,
                  parcels_dir: Path) -> np.ndarray:
    """Project the parcels of one hemisphere to the surface; integer labels, NaN background.

    Later regions overwrite earlier ones where they overlap.
    """
    from nilearn import surface

    parcel_key = 'parcels_L' if hemi == 'L' else 'parcels_R'
    roi_map = np.zeros(n_vertices, dtype=float)

    for roi in roi_defs:
        parcel_names = roi[parcel_key]
        if not parcel_names:
            continue

        merged = np.zeros(n_vertices, dtype=bool)
        for parcel_name in parcel_names:
            parcel_path = find_parcel_path(parcel_name, parcels_dir)
            if parcel_path is None:
                raise FileNotFoundError(f'Parcel {parcel_name} not found under {parcels_dir}')
            surf_data = surface.vol_to_surf(
                str(parcel_path),
                surf_path,
                interpolation='nearest_most_frequent',
                radius=3.0,
            )
            surf_data = np.asarray(surf_data).squeeze()  # guard against 4D inputs
            merged |= surf_data > 0

        roi_map[merged] = roi['value']
        logger.info(f'  {roi["label"]} ({hemi}H): {int(merged.sum())} vertices')

    return np.where(roi_map > 0, roi_map, np.nan)


def compute_roi_contours(roi_map: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Boolean mask of region vertices on a triangle whose vertices carry different labels."""
    labels = np.where(np.isnan(roi_map), 0, roi_map)
    v0, v1, v2 = faces[:, 0], faces[:, 1], faces[:, 2]
    mixed = ~((labels[v0] == labels[v1]) & (labels[v1] == labels[v2]))
    contour = np.zeros(len(roi_map), dtype=bool)
    contour[v0[mixed]] = True
    contour[v1[mixed]] = True
    contour[v2[mixed]] = True
    contour &= ~np.isnan(roi_map)  # contours on region vertices only, not on background
    return contour


def make_discrete_cmap(roi_defs: list):
    """ListedColormap of the region colours plus black for contours.

    Returns (cmap, vmin, vmax, contour_value); contour vertices get value n_rois + 1.
    """
    colors = [roi['color'] for roi in roi_defs] + ['#000000']
    cmap = ListedColormap(colors, name='roi_parcels')
    vmin = 0.5
    vmax = len(roi_defs) + 1.5
    contour_value = float(len(roi_defs) + 1)
    return cmap, vmin, vmax, contour_value


def save_legend_png(roi_defs: list, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(3, 0.4 * len(roi_defs) + 0.3))
    ax.axis('off')
    for i, roi in enumerate(roi_defs):
        ax.add_patch(plt.Rectangle((0, i), 0.3, 0.8, color=roi['color']))
        ax.text(0.35, i + 0.4, roi['label'], va='center', fontsize=11)
    ax.set_xlim(0, 3)
    ax.set_ylim(-0.1, len(roi_defs))
    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close(fig)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output-dir', required=True, help='Where to write the HTML maps + legend.')
    p.add_argument('--parcels-dir', default=None,
                   help='Anatomical parcels (default: emfl.config.get_parcels_dir(), i.e. data/PARCELS).')
    p.add_argument('--space', default='fsaverage6', choices=['fsaverage5', 'fsaverage6'],
                   help='Surface space (default: fsaverage6, as published).')
    p.add_argument('--no-loc', action='store_true', help='Leave out the LOC (objects) region.')
    p.add_argument('--no-contours', action='store_true', help='No black region boundaries.')
    return p


def main(argv=None) -> list:
    args = build_parser().parse_args(argv)
    from nilearn import datasets, plotting

    if args.parcels_dir is None:
        from emfl.config import get_parcels_dir
        parcels_dir = Path(get_parcels_dir())
    else:
        parcels_dir = Path(args.parcels_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    roi_defs = select_rois(include_loc=not args.no_loc)
    tag = roi_tag(roi_defs)
    logger.info(f'Regions: {", ".join(r["label"] for r in roi_defs)} | {args.space} | {parcels_dir}')

    fsaverage = datasets.fetch_surf_fsaverage(mesh=args.space)
    cmap, vmin, vmax, contour_value = make_discrete_cmap(roi_defs)
    legend_str = ' | '.join(r['label'] for r in roi_defs)

    written = []
    for hemi, side in (('L', 'left'), ('R', 'right')):
        sulc = nib.load(fsaverage[f'sulc_{side}']).darrays[0].data
        bg = binarize_sulcal_map(sulc)
        roi_map = build_roi_map(roi_defs, fsaverage[f'pial_{side}'], len(sulc), hemi, parcels_dir)
        if not args.no_contours:
            faces = nib.load(fsaverage[f'pial_{side}']).darrays[1].data
            roi_map[compute_roi_contours(roi_map, faces)] = contour_value

        for surf_name, key in (('inflated', f'infl_{side}'), ('pial', f'pial_{side}')):
            out = output_dir / f'roi_parcels_{tag}_{hemi.lower()}h_{surf_name}.html'
            view = plotting.view_surf(
                surf_mesh=fsaverage[key],
                surf_map=roi_map,
                bg_map=bg,
                threshold=0.5,  # hide NaN background -> the sulcal map shows through
                cmap=cmap,
                symmetric_cmap=False,
                vmin=vmin,
                vmax=vmax,
                title=f'ROI Parcels — {hemi}H ({surf_name}) | {legend_str}',
            )
            view.save_as_html(str(out))
            written.append(out)
            logger.info(f'  Saved: {out.name}')

    legend = output_dir / f'roi_parcels_{tag}_legend.png'
    save_legend_png(roi_defs, legend)
    written.append(legend)
    logger.info('Colours: ' + ' | '.join(f'{r["label"]}={r["color"]}' for r in roi_defs))
    return written


if __name__ == '__main__':
    main()
