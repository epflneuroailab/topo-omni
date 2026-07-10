"""Shared volume->surface projection + surface-map rendering (docs/DESIGN.md §4).

Used by Pernet Fig. 3b (voice-selective surface map), and — once ported — Marvi
Figs. 2 & 3 and Jung Fig. 6. Two responsibilities:

  1. `vol_to_surf` / `project_to_fsaverage` — project a volumetric statistical map
     onto an fsaverage mesh. This is the numeric spine and it is **version-robust**:
     calibrated on the Pernet lineage, nilearn 0.12.1 reproduces the published
     (0.10.4-produced) `surface_data_fsaverage6.npz` **bitwise** (max|Δ| = 0.0) with
     the pinned parameters below. That exactness is why this belongs in `core` and is
     golden-mastered at atol=0 (Pernet_2015/tests/test_surface_projection_golden.py).

  2. `plot_surface_map` — the consolidated "figure look" (inflated mesh, binarized
     sulcal background, black-at-zero viridis for positive-only maps). Rendering is
     matplotlib and NOT golden-masterable numerically; it is smoke-tested only.

⚠ nilearn-version-SENSITIVE (docs/DESIGN.md §4): nilearn is imported LAZILY inside every
function so that `import core.surface` never binds nilearn (Pernet 0.10.4 vs
Marvi/Jung 0.12.1). Top level stays numpy-only.

PORT NOTES (from Pernet dev-repo `src/02_surface_projection.py` @ f842b1a):
  * `PROJECTION_KWARGS` (radius=0, kind='auto', n_samples=1) are the exact params the
    published maps were produced with — do not change (golden-master at atol=0).
  * The non-paper visualization variants in the dev script (percentile thresholds,
    Destrieux A1 auditory overlay, bidirectional "complete" and secondary
    "thresholded" maps, the comprehensive multi-panel figure) are intentionally NOT
    ported — scope-lock to paper figures (docs/DESIGN.md §2.4, §8).
"""
from __future__ import annotations

import numpy as np

# NOTE: do NOT `import nilearn`/`matplotlib` at module scope — see the version note.

# Exact projection parameters the published Pernet surface data was produced with.
# Frozen: changing any of these breaks the atol=0 golden master.
PROJECTION_KWARGS = dict(radius=0, kind="auto", n_samples=1)

# fsaverage6 = 40,962 vertices per hemisphere (Pernet 2015 used ~82k bilateral).
DEFAULT_TEMPLATE = "fsaverage6"


def vol_to_surf(stat_map, surf_mesh, **overrides):
    """Project a volumetric map onto a surface mesh (thin, version-tolerant wrapper).

    Wraps ``nilearn.surface.vol_to_surf`` with the pinned ``PROJECTION_KWARGS``. Pass
    ``overrides`` only to deviate deliberately (e.g. a unit test probing another
    sampling); the paper pipeline uses the defaults.

    Parameters
    ----------
    stat_map : niimg-like
        Volumetric statistical map (path or Nifti1Image).
    surf_mesh : path or mesh
        Target surface (e.g. an fsaverage ``pial_left``).

    Returns
    -------
    np.ndarray
        1-D array of per-vertex values (fsaverage6 pial -> 40,962,).
    """
    from nilearn import surface  # lazy — see module docstring

    kwargs = {**PROJECTION_KWARGS, **overrides}
    return surface.vol_to_surf(stat_map, surf_mesh, **kwargs)


def project_to_fsaverage(stat_map, template=DEFAULT_TEMPLATE, fsaverage=None):
    """Project one volumetric map to both fsaverage hemispheres (pial surfaces).

    Returns ``{'lh': ndarray, 'rh': ndarray}`` matching the ``*_lh`` / ``*_rh`` keys
    of the published ``surface_data_<template>.npz``.
    """
    from nilearn import datasets  # lazy

    if fsaverage is None:
        fsaverage = datasets.fetch_surf_fsaverage(template)
    return {
        "lh": vol_to_surf(stat_map, fsaverage["pial_left"]),
        "rh": vol_to_surf(stat_map, fsaverage["pial_right"]),
    }


def black_viridis_cmap(n_colors=256):
    """viridis with index 0 mapped to pure black (the positive-only voice-map look).

    Contours/zeros render black while positive activation keeps the viridis gradient.
    Faithful to the Pernet dev renderer.
    """
    import matplotlib.pyplot as plt  # lazy
    from matplotlib.colors import LinearSegmentedColormap

    viridis = plt.cm.viridis
    colors = [[0.0, 0.0, 0.0, 1.0]]  # pure black at index 0
    for i in range(1, n_colors):
        colors.append(viridis(0.05 + (i / n_colors) * 0.95))
    return LinearSegmentedColormap.from_list("black_viridis", colors, N=n_colors)


def binarize_sulcal_map(sulc_data):
    """Binarize sulcal depth for crisp background: sulci -> 179/255 gray, gyri -> white."""
    return np.where(sulc_data > 0, 179 / 255, 1.0)


def plot_surface_map(surf_data, hemi, fsaverage, *, threshold=1e-6, vmin=0, vmax=16,
                     cmap=None, view="lateral", title="", figure=None):
    """Render one hemisphere's surface stat map with the shared publication look.

    Inflated mesh, binarized sulcal-depth background, black-at-zero viridis by default.
    Matplotlib rendering — smoke-tested, not golden-mastered.

    Parameters
    ----------
    surf_data : np.ndarray
        Per-vertex values (NaN where below threshold / not shown).
    hemi : {'left', 'right'}
    fsaverage : mapping
        A fetched nilearn fsaverage dict (needs ``infl_*`` and ``sulc_*``).

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt  # lazy
    import nibabel as nib
    from nilearn import plotting

    hemi_key = "left" if hemi == "left" else "right"
    if cmap is None:
        cmap = black_viridis_cmap()
    sulc = binarize_sulcal_map(nib.load(fsaverage[f"sulc_{hemi_key}"]).darrays[0].data)

    if figure is None:
        figure = plt.figure(figsize=(10, 8))
    plotting.plot_surf_stat_map(
        fsaverage[f"infl_{hemi_key}"],
        surf_data,
        hemi=hemi,
        view=view,
        threshold=threshold,
        bg_map=sulc,
        bg_on_data=True,
        colorbar=True,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        title=title,
        figure=figure,
    )
    return figure
