"""
Reconstruct the macaque V1 orientation map that TDANN compares against, so our
Delta-orientation-vs-distance curve can be plotted against a real reference rather than a
value read off a figure.

Where the "macaque data" actually comes from
-------------------------------------------
It is not a data file. TDANN's ``NauhausOrientationTissue``
(``spacetorch/maps/screenshot_maps.py``) reconstructs the map by **colour-decoding a
published figure**: it crops the orientation-map panel out of Fig. 2 of

    Nauhaus, Nielsen, Disney & Callaway (2012), "Orthogonal micro-organization of
    orientation and spatial frequency in primate V1", Nature Neuroscience 15:1683.

matches every pixel against the panel's own colour legend to recover a preferred
orientation, and converts pixels to millimetres with the panel's scale bar. Their own module
docstring is explicit about this: "Screenshot maps are constructed from portions of JPEGs
taken from published results". This module replicates that decoding exactly, so the curve is
the same object theirs is -- with the same limitation, which is that it is a colour
approximation of a printed figure, not the underlying recording.

The figure JPEG is a copyrighted publication figure, so it is **downloaded on demand and not
committed**. Only the derived numeric curve is cached (``data/macaque_delta_curve.npz``).

Binning matters for this comparison
-----------------------------------
TDANN's ``get_curves`` bins distance as ``linspace(0, sqrt(2*(4/3*Lambda)^2)/2, 10)`` --
9 bins reaching ~0.943 Lambda -- and reports distance as a fraction of Lambda. Since
``smoothness`` is ``(max - curve[0]) / max``, its value depends on how wide the first bin is,
so a curve binned differently is not comparable to their ~0.65 for macaque. ``tdann_bin_edges``
below is unit-agnostic (mm for macaque, sheet px for us), which is what puts both curves on a
common x axis.
"""

import urllib.request
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.spatial.distance import pdist, squareform

# Asset lives in the TDANN repo; see the module docstring for provenance.
NAUHAUS_URL = (
    "https://raw.githubusercontent.com/neuroailab/TDANN/main/"
    "notebooks/assets/F02/41593_2012_Article_BFnn3255_Fig2_HTML.jpg"
)

# Verbatim from NauhausOrientationTissue.metadata
NAUHAUS_META = {
    "map_box": (52, 118, 420, 370),
    "legend_box": (955, 15, 1010, 470),
    "scale_bar_px": 810 - 563,
    "scale_bar_len_mm": 0.5,
    "legend_positions": [[25, 25], [25, 90], [25, 160], [25, 225],
                         [25, 290], [25, 360], [25, 420]],
}

# TDANN's hypercolumn_widths["Macaque V1"]: "estimated spacing of same-orientation columns"
MACAQUE_COLUMN_SPACING_MM = 0.75


def tdann_bin_edges(column_spacing: float, n_bins: int = 9) -> np.ndarray:
    """TDANN's distance bins, in whatever unit ``column_spacing`` is given in.

    ``analysis_width = 4/3 * Lambda``; the largest distance inside a window that wide is its
    half-diagonal, ``sqrt(2 * analysis_width^2) / 2``, which is ~0.943 * Lambda.
    """
    analysis_width = column_spacing * (4.0 / 3.0)
    max_dist = np.sqrt(2.0 * analysis_width ** 2) / 2.0
    return np.linspace(0.0, max_dist, n_bins + 1)


def fetch_nauhaus_figure(dest: Path) -> Path:
    """Download the source figure if absent. Deliberately not stored in the repo."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(NAUHAUS_URL, headers={"Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=120) as resp, dest.open("wb") as fh:
        fh.write(resp.read())
    return dest


def decode_orientation_map(img_path: Path) -> Tuple[np.ndarray, float]:
    """Colour-decode the orientation panel into preferred orientations in degrees.

    Returns:
        angles : (H, W) preferred orientation in [0, 180)
        px_per_mm : scale from the panel's scale bar
    """
    from PIL import Image                      # local import: only needed for this path
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.spatial.distance import cdist

    raw = Image.open(img_path).convert("RGB")
    data = np.array(raw.crop(box=NAUHAUS_META["map_box"])) / 255.0
    legend = np.array(raw.crop(box=NAUHAUS_META["legend_box"])) / 255.0

    # Build a colormap from 7 sampled legend colours, then invert it by nearest-colour
    # lookup against 500 angles -- exactly TDANN's procedure.
    colors = [legend[y, x] for x, y in NAUHAUS_META["legend_positions"]]
    cmap = LinearSegmentedColormap.from_list("", colors)
    angles = np.linspace(0, 180, 500)
    dense = np.array([cmap(i) for i in np.linspace(0, 1, 500)])

    alpha = np.ones(data.shape[:-1])[..., None]
    flat = np.concatenate((data, alpha), axis=-1).reshape(-1, 4)
    nearest = np.argmin(cdist(flat, dense), axis=1)

    px_per_mm = NAUHAUS_META["scale_bar_px"] / NAUHAUS_META["scale_bar_len_mm"]
    return angles[nearest].reshape(data.shape[:-1]), float(px_per_mm)


# Spatial-frequency panel of the same figure (NauhausSFTissue). Its legend is decoded onto
# a log2 spatial-frequency scale, so macaque SF preferences are in log2(cpd) units while the
# model preferences below are SF *indices* -- TDANN mixes these units too, which is only
# defensible because `smoothness` is scale-free.
NAUHAUS_SF_META = {
    "map_box": (52, 613, 420, 865),
    "legend_box": (955, 510, 1050, 965),
    "scale_bar_px": 810 - 563,
    "scale_bar_len_mm": 0.5,
    "legend_positions": [[25, 30], [25, 75], [25, 125], [25, 175], [25, 225],
                         [25, 275], [25, 325], [25, 375], [25, 420]],
}


def decode_sf_map(img_path: Path) -> Tuple[np.ndarray, float]:
    """Colour-decode the spatial-frequency panel into log2 spatial frequency."""
    from PIL import Image
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.spatial.distance import cdist

    raw = Image.open(img_path).convert("RGB")
    data = np.array(raw.crop(box=NAUHAUS_SF_META["map_box"])) / 255.0
    legend = np.array(raw.crop(box=NAUHAUS_SF_META["legend_box"])) / 255.0

    colors = [legend[y, x] for x, y in NAUHAUS_SF_META["legend_positions"]]
    cmap = LinearSegmentedColormap.from_list("", colors)
    sfs = np.logspace(start=0, stop=3, base=2, num=500)          # 1 - 8, as TDANN does
    dense = np.array([cmap(i) for i in np.linspace(0, 1, 500)])

    alpha = np.ones(data.shape[:-1])[..., None]
    flat = np.concatenate((data, alpha), axis=-1).reshape(-1, 4)
    nearest = np.argmin(cdist(flat, dense), axis=1)

    px_per_mm = NAUHAUS_SF_META["scale_bar_px"] / NAUHAUS_SF_META["scale_bar_len_mm"]
    return np.log2(sfs[nearest]).reshape(data.shape[:-1]), float(px_per_mm)


def macaque_sf_delta_curve(
    cache: Path = Path("data/macaque_sf_delta_curve.npz"),
    figure_cache: Path = None,
    column_spacing_mm: float = MACAQUE_COLUMN_SPACING_MM,
    n_bins: int = 9,
    num_samples: int = 20,
    sample_size: int = 1000,
    shuffle: bool = False,
    overwrite: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Macaque Delta-spatial-frequency-vs-distance curve (TDANN Fig. 2G reference)."""
    cache = Path(cache)
    if shuffle:      # distance-shuffled control gets its own cache
        cache = cache.with_name(cache.stem + "_shuffled" + cache.suffix)
    if cache.exists() and not overwrite:
        z = np.load(cache)
        return z["x"], z["curves"]

    figure_cache = Path(figure_cache) if figure_cache else cache.parent / "_nauhaus_fig2.jpg"
    img = fetch_nauhaus_figure(figure_cache)
    sf_map, px_per_mm = decode_sf_map(img)
    rows, cols = sf_map.shape
    rr, cc = np.indices((rows, cols))
    positions = np.stack([cc.ravel(), rr.ravel()], axis=1) / px_per_mm

    edges = tdann_bin_edges(column_spacing_mm, n_bins=n_bins)
    curves = delta_over_distance(sf_map.ravel(), positions, edges,
                                 num_samples=num_samples, sample_size=sample_size,
                                 circular=False, shuffle=shuffle)
    x = (0.5 * (edges[:-1] + edges[1:])) / column_spacing_mm

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, x=x, curves=curves, column_spacing_mm=column_spacing_mm,
             source=NAUHAUS_URL)
    return x, curves


# ---------------------------------------------------------------------------------------
# Livingstone & Hubel 1984, Fig. 27B -- the colour ("cytochrome-oxidase blob") reference
# ---------------------------------------------------------------------------------------
# TDANN's LivingstoneColorTissue decodes this figure, but the asset is NOT in their public
# repo (404, and absent from the full tree), so unlike the Nauhaus panels it cannot be
# reproduced from their release. Supply the figure yourself and point this at it.
#
# Their decode: crop the blob field, take two pixel colours from inside the crop as the ends
# of a 2-colour ramp, and map every pixel to the nearest point on that ramp, giving a
# continuous "blobness" in [0, 1]. Pixel-to-mm comes from the figure's scale bar. Their
# hard-coded crop/scale-bar boxes are specific to their own copy of the image, so they are
# arguments here rather than constants -- a different scan or screenshot has different
# framing, and this calibration is what sets the x axis.
#
# Note the endpoint colours are sampled by percentile rather than at two fixed pixels: on a
# rescreened reproduction, two hard-coded coordinates can easily land on halftone dots.

# Cross-checked against TDANN's own rendering of this same data (their Fig. 2H "Macaque V1"
# chromatic-preference panel, captioned "reconstruction of cytochrome oxidase staining data
# adapted from Livingstone and Hubel"). Decoding their rendered panel at its own 200 um
# scale bar (420 px/mm) gives smoothness 0.557 against 0.533 from the raw figure here -- a
# 0.024 gap, i.e. the decode reproduces their processing of the same source.
#
# Their panel is NOT used as the reference, though, because it spans only 0.84 mm, about 1.1
# column spacings. Smoothness reads the short-distance rise and survives that (0.557 vs
# 0.533), but the plateau does not: the panel gives plateau/chance 1.140 versus 0.662 from a
# 3.8 mm crop, because at the outer bins the only available pairs are near-opposite corners.
# Its FFT confirms the problem -- the "dominant period" comes out as 0.838 mm, i.e. the panel
# width itself, so there is no resolvable periodicity inside it.

def decode_livingstone_blobs(
    img_path: Path,
    map_box: Tuple[int, int, int, int],
    lo_pct: float = 5.0,
    hi_pct: float = 95.0,
) -> np.ndarray:
    """Colour-decode the blob field into a continuous blobness map in [0, 1]."""
    from PIL import Image
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.spatial.distance import cdist

    raw = Image.open(img_path).convert("RGB")
    data = np.array(raw.crop(box=tuple(map_box))) / 255.0
    if min(data.shape[:2]) < 20:
        raise ValueError(f"crop {map_box} is only {data.shape[:2]} px; too small")

    # Endpoints of the ramp: robust dark (blob) and light (interblob) colours in the crop.
    lum = data.mean(axis=-1)
    dark = data[lum <= np.percentile(lum, lo_pct)].mean(axis=0)
    light = data[lum >= np.percentile(lum, hi_pct)].mean(axis=0)
    cmap = LinearSegmentedColormap.from_list("", [dark, light])

    values = np.linspace(0, 1, 500)
    dense = np.array([cmap(v) for v in values])
    alpha = np.ones(data.shape[:-1])[..., None]
    flat = np.concatenate((data, alpha), axis=-1).reshape(-1, 4)
    nearest = np.argmin(cdist(flat, dense), axis=1)
    return values[nearest].reshape(data.shape[:-1])


def livingstone_color_curve(
    img_path: Path,
    map_box: Tuple[int, int, int, int],
    scale_bar_px: float,
    scale_bar_len_mm: float = 1.0,
    column_spacing_mm: float = MACAQUE_COLUMN_SPACING_MM,
    cache: Path = Path("data/macaque_color_delta_curve.npz"),
    n_bins: int = 9,
    num_samples: int = 20,
    sample_size: int = 1000,
    shuffle: bool = False,
    overwrite: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Macaque colour Delta-preference-vs-distance curve (TDANN Fig. 2I reference).

    ``scale_bar_px`` is the scale bar's length in pixels *of the supplied image*, and
    ``scale_bar_len_mm`` its physical length. Both must be measured from that image.
    """
    cache = Path(cache)
    if shuffle:      # distance-shuffled control gets its own cache
        cache = cache.with_name(cache.stem + "_shuffled" + cache.suffix)
    if cache.exists() and not overwrite:
        z = np.load(cache)
        return z["x"], z["curves"]

    blobs = decode_livingstone_blobs(Path(img_path), map_box)
    px_per_mm = float(scale_bar_px) / float(scale_bar_len_mm)
    rows, cols = blobs.shape
    rr, cc = np.indices((rows, cols))
    positions = np.stack([cc.ravel(), rr.ravel()], axis=1) / px_per_mm

    edges = tdann_bin_edges(column_spacing_mm, n_bins=n_bins)
    curves = delta_over_distance(blobs.ravel(), positions, edges,
                                 num_samples=num_samples, sample_size=sample_size,
                                 circular=False, shuffle=shuffle)
    x = (0.5 * (edges[:-1] + edges[1:])) / column_spacing_mm

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, x=x, curves=curves, px_per_mm=px_per_mm, map_box=np.array(map_box),
             extent_mm=np.array([cols / px_per_mm, rows / px_per_mm]))
    return x, curves


# TDANN's own rendering of the Livingstone data (their Fig. 2H "Macaque V1" chromatic
# preference panel). Already a continuous grey map -- dark = stronger chromatic response --
# so no colour-ramp inversion is needed, unlike the raw halftone figure.
TDANN_MACAQUE_COLOR_META = {
    "map_box": (8, 8, 360, 360),        # the grey square, above the scale bar
    "scale_bar_px": 84.0,               # detected: y375-384, x270-354
    "scale_bar_len_mm": 0.2,            # labelled "200 um"
}


def tdann_macaque_color_curve(
    img_path: Path,
    map_box: Tuple[int, int, int, int] = None,
    scale_bar_px: float = None,
    scale_bar_len_mm: float = None,
    column_spacing_mm: float = MACAQUE_COLUMN_SPACING_MM,
    cache: Path = Path("data/tdann_macaque_color_curve.npz"),
    n_bins: int = 9,
    num_samples: int = 20,
    sample_size: int = 1000,
    shuffle: bool = False,
    overwrite: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Colour curve from TDANN's rendered macaque panel rather than the raw figure.

    Useful as a cross-check on ``livingstone_color_curve`` -- the two agree on smoothness
    (0.557 vs 0.533). Do NOT use it for the plateau: the panel spans 0.84 mm, about 1.1
    column spacings, so the outer distance bins only contain near-opposite-corner pairs and
    the plateau comes out at 1.140 against 0.662 from a 3.8 mm crop.
    """
    from PIL import Image

    meta = TDANN_MACAQUE_COLOR_META
    map_box = map_box or meta["map_box"]
    scale_bar_px = scale_bar_px or meta["scale_bar_px"]
    scale_bar_len_mm = scale_bar_len_mm or meta["scale_bar_len_mm"]

    cache = Path(cache)
    if shuffle:
        cache = cache.with_name(cache.stem + "_shuffled" + cache.suffix)
    if cache.exists() and not overwrite:
        z = np.load(cache)
        return z["x"], z["curves"]

    lum = np.array(Image.open(Path(img_path)).convert("L").crop(tuple(map_box)))
    pref = 1.0 - lum.astype(float) / 255.0          # dark = stronger chromatic response
    px_per_mm = float(scale_bar_px) / float(scale_bar_len_mm)

    rows, cols = pref.shape
    rr, cc = np.indices((rows, cols))
    positions = np.stack([cc.ravel(), rr.ravel()], axis=1) / px_per_mm

    edges = tdann_bin_edges(column_spacing_mm, n_bins=n_bins)
    curves = delta_over_distance(pref.ravel(), positions, edges,
                                 num_samples=num_samples, sample_size=sample_size,
                                 circular=False, shuffle=shuffle)
    x = (0.5 * (edges[:-1] + edges[1:])) / column_spacing_mm

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, x=x, curves=curves, px_per_mm=px_per_mm,
             extent_mm=np.array([cols / px_per_mm, rows / px_per_mm]))
    return x, curves


def macaque_tissue(img_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Decoded map as (positions_mm, preferences_deg), one entry per panel pixel."""
    angles, px_per_mm = decode_orientation_map(img_path)
    rows, cols = angles.shape
    rr, cc = np.indices((rows, cols))
    positions = np.stack([cc.ravel(), rr.ravel()], axis=1) / px_per_mm
    return positions, angles.ravel()


def delta_over_distance(
    preferences: np.ndarray, positions: np.ndarray, bin_edges: np.ndarray,
    num_samples: int = 20, sample_size: int = 1000, shuffle: bool = False, seed: int = 424,
    circular: bool = True,
) -> np.ndarray:
    """Mean |Delta preference| per distance bin, one row per resample.

    Mirrors ``SimpleTissue.difference_over_distance``: sample units, take all pairwise
    absolute differences and pairwise distances, average per bin.

    ``circular`` applies the mod-180 wrap that orientation needs. Spatial frequency is a
    linear variable, so it must be passed ``circular=False`` -- TDANN gates the same wrap on
    ``metric == angle_metric`` for exactly this reason.
    """
    rng = np.random.default_rng(seed)
    n_bins = len(bin_edges) - 1
    curves = np.full((num_samples, n_bins), np.nan)

    for s in range(num_samples):
        idx = rng.choice(len(positions), size=min(sample_size, len(positions)),
                         replace=False)
        p = preferences[idx][:, None]
        diffs = np.abs(p - p.T)
        if circular:
            diffs = np.where(diffs >= 90.0, 180.0 - diffs, diffs)
        dists = squareform(pdist(positions[idx]))

        tri = np.tril_indices(len(idx), k=-1)
        diffs, dists = diffs[tri], dists[tri]
        if shuffle:
            dists = rng.permutation(dists)

        which = np.digitize(dists, bin_edges) - 1
        for b in range(n_bins):
            sel = which == b
            if sel.any():
                curves[s, b] = np.nanmean(diffs[sel])
    return curves


def macaque_delta_curve(
    cache: Path = Path("data/macaque_delta_curve.npz"),
    figure_cache: Path = None,
    column_spacing_mm: float = MACAQUE_COLUMN_SPACING_MM,
    n_bins: int = 9,
    num_samples: int = 20,
    sample_size: int = 1000,
    shuffle: bool = False,
    overwrite: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Macaque Delta-orientation-vs-distance curve, x in units of column spacing.

    Cached as derived numbers so the copyrighted source figure is needed only once.

    Returns:
        x      : (n_bins,) bin midpoints as a fraction of the column spacing
        curves : (num_samples, n_bins) one curve per resample
    """
    cache = Path(cache)
    if shuffle:      # distance-shuffled control gets its own cache
        cache = cache.with_name(cache.stem + "_shuffled" + cache.suffix)
    if cache.exists() and not overwrite:
        z = np.load(cache)
        return z["x"], z["curves"]

    figure_cache = Path(figure_cache) if figure_cache else cache.parent / "_nauhaus_fig2.jpg"
    img = fetch_nauhaus_figure(figure_cache)
    positions, prefs = macaque_tissue(img)

    edges = tdann_bin_edges(column_spacing_mm, n_bins=n_bins)
    curves = delta_over_distance(prefs, positions, edges,
                                 num_samples=num_samples, sample_size=sample_size,
                                 shuffle=shuffle)
    x = (0.5 * (edges[:-1] + edges[1:])) / column_spacing_mm

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, x=x, curves=curves, column_spacing_mm=column_spacing_mm,
             source=NAUHAUS_URL)
    return x, curves


def estimate_macaque_column_spacing(img_path: Path) -> float:
    """Independent check on the 0.75 mm constant, from the decoded map's own spectrum.

    TDANN hard-codes the macaque column spacing and reads the millimetre scale from a scale
    bar cropped elsewhere in a multi-panel figure, so it is worth confirming the number the
    x axis is normalised by.
    """
    from src.experiments.orientation.metrics import column_spacing
    angles, px_per_mm = decode_orientation_map(img_path)
    lam_px, _, _ = column_spacing(angles, np.ones_like(angles, dtype=bool))
    return float(lam_px / px_per_mm)
