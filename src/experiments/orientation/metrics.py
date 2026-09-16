"""
Quantify the orientation map on the Topo-Omni cortical sheet with the metrics TDANN uses
for its V1 figure (Margalit et al., Fig. 2 / notebooks/F02_V1_Details).

Consumes what ``src.experiments.orientation.run`` already saved (``orientation_stats.npz``),
so this needs no GPU and no re-extraction.

Metrics, and where each comes from in TDANN
-------------------------------------------
circular variance (CV)
    ``spacetorch.datasets.sine_gratings.Responses._compute_circular_variance``:
    ``CV = 1 - |sum(curve * exp(2i*theta)) / sum(curve)|``. Identical to the resultant
    length ``run_orientation`` already computes, so ``CV == 1 - OSI``; recomputed here from
    the saved tuning curve so this module stands alone and to keep TDANN's naming.
column spacing (Lambda)
    TDANN *hard-codes* this per model (``hypercolumn_widths``, 0.75-3.5 mm). Our sheet has
    no physical scale to look up, so it is estimated from the data: the peak of the
    radially averaged power spectrum of ``exp(2i*theta)``, which is the standard way of
    reading off orientation-map periodicity. Override with ``--column-spacing``.
smoothed orientation map + circular sd
    ``spacetorch.maps.smoother.Smoother``: a sliding square window, aggregated with
    ``circmean`` (and ``circstd`` for the heterogeneity map). Our units already sit on a
    regular grid, so the same thing is a strided box filter -- implemented directly, which
    is exact and far faster than iterating windows.
winding numbers + pinwheel counting
    ``spacetorch.maps.pinwheel_detector.PinwheelDetector``: walk the 8 neighbours of each
    pixel, sum the circular increments, divide by 2*pi; a pinwheel is a connected region
    where ``|winding| > thresh`` and the circular sd is below ``var_thresh``.
pinwheel density
    ``count / num_hcol**2`` with ``num_hcol = extent / Lambda``, i.e. pinwheels per
    Lambda^2 -- dimensionless, so it is comparable to TDANN (~2.1) and macaque V1 (~3.0)
    even though our sheet has no millimetre scale.
Delta preferred orientation vs. distance, and smoothness
    ``V1Map.metric_difference_over_distance`` and ``spatial_utils.smoothness``:
    ``(max(curve) - curve[0]) / max(curve)``. ~1 for a smooth map, ~0 for a random one.

Two deliberate departures from TDANN's code, both flagged where they happen:
  * TDANN passes ``high=high-1`` (179) to ``scipy.stats.circmean``/``circstd`` for data whose
    period is 180. That is an off-by-one; this module uses the correct period of 180.
  * TDANN's ``increments`` takes 7 increments across the 8 neighbours, leaving the loop open,
    which biases winding numbers low. ``closed_loop=True`` (the default here) adds the
    closing increment; pass ``--open-loop`` to reproduce TDANN exactly.
Also note TDANN's ``circstd`` aggregation divides by the number of units in the window, so
its ``var_thresh=0.3`` is tied to their unit density. This module reports the circular sd in
degrees instead, with a threshold in degrees, and prints the distribution so the cut can be
sanity-checked.

Usage
-----
python -m src.experiments.orientation.metrics \
    --stats results/topo-omni/orientation/orientation_stats.npz
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.stats import sem
from skimage.measure import label

from src.experiments.orientation.run import (
    SHEET_DEPTH, SHEET_COLS, VISION_DEPTH_ROWS, VISION_SHEET_COLS, N_VISION_SLOTS,
)

from dotenv import load_dotenv
load_dotenv()

# The vision encoder's 33 slots tile the band's 165 rows, 5 rows each, so the row axis is
# depth while the column axis is tangential. See ``band_anisotropy``. Slot 0 is the
# `patch_embed` conv front-end and slots 1..32 are the transformer blocks, so a "layer
# index" here is one greater than the corresponding `visual.blocks` index.
N_VISION_LAYERS = N_VISION_SLOTS
ROWS_PER_VISION_LAYER = VISION_DEPTH_ROWS // N_VISION_LAYERS


def layer_band_mask(layers) -> np.ndarray:
    """(165, 256) boolean mask selecting given vision-encoder slot indices.

    Slot ``L`` owns sheet depth rows ``5L .. 5L+4``. ``band_grid`` rotates the sheet, which
    *reverses* the depth axis -- band row ``i`` holds depth row ``164 - i`` -- so slot 0
    (the earliest) lands at the BOTTOM of the band grid and slot 32 at the top. Hence early
    slots are the lower rows, which is where the selectivity map is strongest.

    Slot 0 is the `patch_embed` front-end, so slot ``L`` is `visual.blocks[L - 1]`.
    """
    layers = np.atleast_1d(np.asarray(layers, dtype=int))
    if layers.min() < 0 or layers.max() >= N_VISION_LAYERS:
        raise ValueError(f"layer indices must lie in [0, {N_VISION_LAYERS - 1}]")
    mask = np.zeros((VISION_DEPTH_ROWS, VISION_SHEET_COLS), dtype=bool)
    for lay in layers:
        depth_rows = np.arange(lay * ROWS_PER_VISION_LAYER,
                               (lay + 1) * ROWS_PER_VISION_LAYER)
        mask[VISION_DEPTH_ROWS - 1 - depth_rows, :] = True
    return mask


def parse_layers(spec: str) -> np.ndarray:
    """Parse a layer selection: ``all``, ``0-7``, ``0,3,5``, or a mix of ranges and indices."""
    spec = str(spec).strip().lower()
    if spec in ("all", "", "none"):
        return np.arange(N_VISION_LAYERS)
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))       # inclusive, like "0-7" = 8 layers
        else:
            out.append(int(part))
    return np.unique(np.array(out, dtype=int))


# -------------------------------------------------
# Sheet -> vision-band grid
# -------------------------------------------------

def band_grid(values: np.ndarray) -> np.ndarray:
    """Smoothed-sheet vector -> the vision-encoder block as a 2D grid.

    Deliberately does not call ``run_orientation.vision_band``: that helper can be toggled
    to return the whole rotated sheet for display, and every metric here (column spacing,
    pinwheel density, map smoothness) is only meaningful within one retinotopic band.
    """
    grid = np.asarray(values, dtype=float).reshape(SHEET_COLS, SHEET_DEPTH)
    grid = np.rot90(grid, k=1)                    # (309, 512), depth on rows, reversed
    band = grid[SHEET_DEPTH - VISION_DEPTH_ROWS:, :VISION_SHEET_COLS]
    assert band.shape == (VISION_DEPTH_ROWS, VISION_SHEET_COLS), band.shape
    return band


# -------------------------------------------------
# Circular variance (TDANN: _compute_circular_variance)
# -------------------------------------------------

def circular_variance(curve: np.ndarray, ori_deg: np.ndarray,
                      baseline: str = "min") -> np.ndarray:
    """CV = 1 - |R|, the TDANN orientation tuning-strength measure.

    curve : (n_ori, n_units) mean response per orientation
    Low CV = sharply orientation tuned; CV -> 1 = untuned. Panel B of their figure shows
    example curves at CV 0.19 / 0.41 / 0.66 / 0.98.

    ``baseline`` mirrors ``run_orientation.orientation_preference``: TDANN's responses are
    post-ReLU and non-negative, so they divide by ``sum(curve)`` directly. Sheet activations
    are roughly zero-centred, which would make that denominator meaningless (or negative),
    so the curve is shifted by its per-unit minimum first.
    """
    if baseline == "min":
        w = curve - curve.min(axis=0, keepdims=True)
    elif baseline == "zero":
        w = np.clip(curve, 0.0, None)
    else:
        raise ValueError(f"baseline must be 'min' or 'zero' (got {baseline!r})")

    theta2 = 2.0 * np.deg2rad(np.asarray(ori_deg, dtype=float))
    num = np.abs((w * np.exp(1j * theta2)[:, None]).sum(axis=0))
    den = w.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.where(den > 0, num / den, 0.0)
    return 1.0 - np.clip(R, 0.0, 1.0)


# -------------------------------------------------
# Column spacing (Lambda) from the map's power spectrum
# -------------------------------------------------

def column_spacing(pref_ori_2d: np.ndarray, valid: np.ndarray = None,
                   min_period_px: float = 4.0,
                   max_period_px: float = None) -> Tuple[float, np.ndarray, np.ndarray]:
    """Typical orientation-column spacing, in pixels, from the map's power spectrum.

    Orientation is axial, so the map is analysed as the complex field ``exp(2i*theta)``,
    whose spatial periodicity is the hypercolumn periodicity. The radially averaged power
    spectrum peaks at 1/Lambda.

    Returns:
        lam    : estimated spacing in pixels
        freqs  : radial frequency bins (cycles/pixel)
        power  : radially averaged power
    """
    theta = np.asarray(pref_ori_2d, dtype=float)
    if valid is None:
        valid = np.isfinite(theta)
    z = np.where(valid, np.exp(2j * np.deg2rad(np.nan_to_num(theta))), 0.0)

    # Remove the mean over valid pixels, else a global orientation bias dominates the DC
    # bin and leaks into low frequencies.
    if valid.any():
        z = np.where(valid, z - z[valid].mean(), 0.0)

    # Hann window in both axes: the band is a hard-edged rectangle, and without tapering
    # the edges spray power across all frequencies.
    ny, nx = z.shape
    z = z * np.hanning(ny)[:, None] * np.hanning(nx)[None, :]

    power = np.abs(np.fft.fftshift(np.fft.fft2(z))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(ny))          # cycles/pixel
    fx = np.fft.fftshift(np.fft.fftfreq(nx))
    kr = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)

    # Radial average onto bins of |k|
    nbins = int(min(ny, nx) // 2)
    edges = np.linspace(0, kr.max(), nbins + 1)
    idx = np.clip(np.digitize(kr.ravel(), edges) - 1, 0, nbins - 1)
    tot = np.bincount(idx, weights=power.ravel(), minlength=nbins)
    cnt = np.bincount(idx, minlength=nbins)
    radial = np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0)
    centres = 0.5 * (edges[:-1] + edges[1:])

    # Restrict to periods the grid can actually resolve
    hi_f = 1.0 / float(min_period_px)
    lo_f = 1.0 / float(max_period_px) if max_period_px else 1.5 / min(ny, nx)
    band = (centres >= lo_f) & (centres <= hi_f) & (radial > 0)
    if not band.any():
        return float("nan"), centres, radial
    peak = centres[band][np.argmax(radial[band])]
    return float(1.0 / peak), centres, radial


# -------------------------------------------------
# Smoothed map + circular sd (TDANN: Smoother, agg=circmean/circstd)
# -------------------------------------------------

def smoothed_orientation_map(
    pref_ori_2d: np.ndarray, valid: np.ndarray, width: int, stride: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Circular mean and circular sd of orientation in sliding square windows.

    Equivalent to TDANN's ``Smoother`` with ``agg='circmean'`` / ``'circstd'``, specialised
    to a regular grid. Circular statistics are done on the doubled angle, so the period is
    180 deg (TDANN passes 179 to scipy here, an off-by-one this avoids).

    Returns:
        ori   : (H', W') circular-mean preferred orientation, degrees in [0, 180)
        sd    : (H', W') circular sd, in degrees of orientation
        n     : (H', W') how many valid units contributed to each window
    """
    theta = np.asarray(pref_ori_2d, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    z = np.where(valid, np.exp(2j * np.deg2rad(np.nan_to_num(theta))), 0.0)

    ny, nx = theta.shape
    rows = range(0, ny - width + 1, stride)
    cols = range(0, nx - width + 1, stride)

    ori = np.full((len(rows), len(cols)), np.nan)
    sd = np.full((len(rows), len(cols)), np.nan)
    n = np.zeros((len(rows), len(cols)), dtype=int)

    # Summed-area tables make this O(H*W) instead of O(H*W*width^2).
    cz = np.pad(np.cumsum(np.cumsum(z, axis=0), axis=1), ((1, 0), (1, 0)))
    cv = np.pad(np.cumsum(np.cumsum(valid.astype(float), axis=0), axis=1), ((1, 0), (1, 0)))

    def box(cum, r, c):
        return (cum[r + width, c + width] - cum[r, c + width]
                - cum[r + width, c] + cum[r, c])

    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            cnt = box(cv, r, c)
            if cnt < 1:
                continue
            zbar = box(cz, r, c) / cnt
            n[i, j] = int(round(cnt))
            ori[i, j] = (np.rad2deg(np.angle(zbar)) / 2.0) % 180.0
            # Circular sd of the doubled angle, halved back to orientation degrees.
            rlen = min(abs(zbar), 1.0)
            if rlen > 1e-12:
                sd[i, j] = np.rad2deg(np.sqrt(-2.0 * np.log(rlen))) / 2.0
            else:
                sd[i, j] = 180.0
    return ori, sd, n


# -------------------------------------------------
# Winding numbers + pinwheels (TDANN: PinwheelDetector)
# -------------------------------------------------

def circdiff(x, y):
    """Circular difference of two orientations, wrapped into [-90, 90)."""
    return (np.asarray(x, dtype=float) - np.asarray(y, dtype=float) + 90.0) % 180.0 - 90.0


def winding_numbers(smoothed: np.ndarray, closed_loop: bool = True) -> np.ndarray:
    """Winding number of the orientation field around every interior pixel.

    Walks the 8 neighbours in TDANN's order (NW, N, NE, E, SE, S, SW, W), sums the circular
    increments and divides by 2*pi. A pinwheel is a point where orientation turns through
    180 deg, i.e. |winding| = 0.5.

    ``closed_loop`` adds the final W->NW increment. TDANN omits it (their ``increments``
    returns n-1 values for n neighbours), which leaves the contour open and biases the
    winding number low; set False to reproduce their numbers exactly.
    """
    s = np.asarray(smoothed, dtype=float)
    # neighbours in walk order, as shifted views
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    stack = np.stack([np.roll(np.roll(s, -dr, axis=0), -dc, axis=1) for dr, dc in offsets])

    pairs = list(zip(range(len(offsets) - 1), range(1, len(offsets))))
    if closed_loop:
        pairs.append((len(offsets) - 1, 0))
    total = sum(np.deg2rad(circdiff(stack[b], stack[a])) for a, b in pairs)
    wn = total / (2.0 * np.pi)

    wn[0, :] = wn[-1, :] = wn[:, 0] = wn[:, -1] = 0.0     # border has no full neighbourhood
    wn[~np.isfinite(wn)] = 0.0
    return wn


def count_pinwheels(
    wn: np.ndarray, sd: np.ndarray, thresh: float = 0.3,
    sd_thresh_deg: float = 30.0, min_px_count: int = 2,
) -> Tuple[List[int], List[np.ndarray]]:
    """Connected regions of high |winding| in homogeneous parts of the map.

    Mirrors ``PinwheelDetector.count_pinwheels``: positive and negative pinwheels are
    labelled separately, and a region must span at least ``min_px_count`` pixels.
    ``sd_thresh_deg`` replaces TDANN's ``var_thresh``, which was a per-window circular sd
    divided by the window's unit count and so depended on their unit density.
    """
    ok = np.isfinite(sd) & (sd < sd_thresh_deg)
    counts, centres = [0, 0], []
    for mask in (ok & (wn > thresh), ok & (wn < -thresh)):
        islands = label(mask)
        found = []
        for lab in np.unique(islands):
            if lab == 0:
                continue
            px = np.argwhere(islands == lab)
            if len(px) < min_px_count:
                continue
            found.append(px.mean(axis=0))
        counts[len(centres)] = len(found)
        centres.append(np.array(found).reshape(-1, 2))
    return counts, centres


def pinwheel_sensitivity(
    ori_band: np.ndarray, valid: np.ndarray, lam: float,
    fracs=(0.10, 0.15, 0.25, 0.5, 1.0), sd_threshs=(30.0, 45.0, 60.0, 1e9),
    thresh: float = 0.3,
) -> List[dict]:
    """Pinwheel count/density across smoothing windows and circular-sd cuts.

    Reported alongside the headline number because it is *not* robust: the density falls by
    two orders of magnitude as the smoothing window grows from 0.15 to 1.0 column spacings,
    since a window approaching Lambda averages the 180 deg turn at a pinwheel away entirely.
    A single density with no window stated is close to meaningless.
    """
    rows = []
    for frac in fracs:
        width = max(3, int(round(frac * lam)))
        stride = max(1, int(round(0.3 * width)))
        osm, sdm, _ = smoothed_orientation_map(ori_band, valid, width, stride)
        wn = winding_numbers(osm)
        entry = {"frac": frac, "width": width, "stride": stride,
                 "px_per_lambda": lam / stride, "shape": osm.shape,
                 "sd_median": float(np.nanmedian(sdm[np.isfinite(sdm)])), "counts": {}}
        for sd_t in sd_threshs:
            c, _ = count_pinwheels(wn, sdm, thresh=thresh, sd_thresh_deg=sd_t)
            entry["counts"][sd_t] = (sum(c),
                                     pinwheel_density(sum(c), osm.shape, stride, lam))
        rows.append(entry)
    return rows


def band_anisotropy(ori_band: np.ndarray, valid: np.ndarray,
                    rows_per_layer: int = ROWS_PER_VISION_LAYER) -> dict:
    """Check whether the band's row axis can be read as tangential cortical space.

    The vision band's rows are the *depth* axis -- 32 transformer layers at
    ``rows_per_layer`` rows each -- while its columns are a genuine tangential axis. Column
    spacing and pinwheel density are defined for a 2D sheet whose axes are both tangential,
    so this quantifies how far the band departs from that: the dominant period along each
    axis, and whether neighbouring rows are more similar within a layer than across a layer
    boundary (if they are, the row axis is not homogeneous and the density is suspect).
    """
    z = np.where(valid, np.exp(2j * np.deg2rad(np.nan_to_num(ori_band))), 0.0)
    if valid.any():
        z = np.where(valid, z - z[valid].mean(), 0.0)

    def peak_period(axis, min_p=4.0):
        n = z.shape[axis]
        w = np.hanning(n)
        w = w[:, None] if axis == 0 else w[None, :]
        P = (np.abs(np.fft.fft(z * w, axis=axis)) ** 2).mean(axis=1 - axis)
        f = np.fft.fftfreq(n)
        keep = (f > 1.5 / n) & (f <= 1.0 / min_p)
        return float(1.0 / f[keep][np.argmax(P[keep])])

    def row_delta(r0, r1):
        m = valid[r0] & valid[r1]
        if m.sum() < 20:
            return np.nan
        dd = np.abs(ori_band[r0][m] - ori_band[r1][m])
        return float(np.where(dd >= 90, 180 - dd, dd).mean())

    within, across = [], []
    for r in range(ori_band.shape[0] - 1):
        (across if ((r + 1) % rows_per_layer == 0) else within).append(row_delta(r, r + 1))

    return {
        "period_rows_px": peak_period(0),
        "period_cols_px": peak_period(1),
        "delta_within_layer_deg": float(np.nanmean(within)),
        "delta_across_layer_deg": float(np.nanmean(across)),
    }


def pinwheel_density(n_pinwheels: int, map_shape: Tuple[int, int],
                     stride: int, lam_px: float) -> float:
    """Pinwheels per Lambda^2 (TDANN: ``total / num_hcol**2``).

    ``map_shape`` is the smoothed map's shape, so its extent in original pixels is
    ``shape * stride``; ``num_hcol`` is that extent measured in column spacings.
    """
    ny, nx = map_shape
    num_hcol_y = (ny * stride) / lam_px
    num_hcol_x = (nx * stride) / lam_px
    return float(n_pinwheels / (num_hcol_y * num_hcol_x))


# -------------------------------------------------
# Delta orientation vs distance, and smoothness
# -------------------------------------------------

def metric_difference_over_distance(
    pref_ori: np.ndarray, positions: np.ndarray, distance_cutoff: float,
    num_samples: int = 50, sample_size: int = 1000, num_bins: int = 20,
    shuffle: bool = False, seed: int = 0, bin_edges: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean |delta preferred orientation| as a function of pairwise distance.

    TDANN's ``V1Map.metric_difference_over_distance``: repeatedly sample units, take all
    pairwise absolute differences (wrapped so 180 deg apart means identical) and pairwise
    distances, and average the differences within distance bins. ``shuffle`` permutes the
    distances to give the chance-level curve.

    Returns:
        midpoints : (num_bins,) bin centres
        curves    : (num_samples, num_bins) one curve per resample
    """
    rng = np.random.default_rng(seed)
    pref_ori = np.asarray(pref_ori, dtype=float)
    positions = np.asarray(positions, dtype=float)

    # Bin edges span the distances actually observed, not [0, cutoff]: units sit on an
    # integer grid, so no pair is closer than 1 px and a [0, 1) first bin would always be
    # empty -- which would put a NaN at curve[0] and make `smoothness` truncate to nothing.
    # TDANN gets this for free by deriving edges with histogram_bin_edges on the first
    # sample and reusing them; same thing here.
    # Explicit edges let this adopt TDANN's binning (9 bins to 0.943*Lambda), which is
    # required for the smoothness value to be comparable to theirs -- `smoothness` is
    # (max - curve[0]) / max, so it depends on how wide the first bin is.
    edges = None if bin_edges is None else np.asarray(bin_edges, dtype=float)
    if edges is not None:
        num_bins = len(edges) - 1
    curves = np.full((num_samples, num_bins), np.nan)

    for s in range(num_samples):
        idx = rng.choice(len(positions), size=min(sample_size, len(positions)),
                         replace=False)
        p = pref_ori[idx][:, None]
        diffs = np.abs(p - p.T)
        diffs = np.where(diffs >= 90.0, 180.0 - diffs, diffs)      # circular wrap
        dists = squareform(pdist(positions[idx]))

        tri = np.tril_indices(len(idx), k=-1)
        diffs, dists = diffs[tri], dists[tri]
        if shuffle:
            dists = rng.permutation(dists)

        keep = dists <= distance_cutoff
        diffs, dists = diffs[keep], dists[keep]
        if dists.size == 0:
            continue
        if edges is None:
            edges = np.linspace(dists.min(), distance_cutoff, num_bins + 1)
        which = np.clip(np.digitize(dists, edges) - 1, 0, num_bins - 1)
        for b in range(num_bins):
            sel = which == b
            if sel.any():
                curves[s, b] = np.nanmean(diffs[sel])

    if edges is None:
        raise ValueError(f"no unit pairs within distance_cutoff={distance_cutoff}")
    return 0.5 * (edges[:-1] + edges[1:]), curves


def smoothness(curve: np.ndarray) -> float:
    """TDANN's ``spatial_utils.smoothness``: (max - first) / max, truncated at the first NaN.

    ~1 when nearby units share an orientation and the curve climbs with distance (a smooth
    map); ~0 when the curve starts at its ceiling (a salt-and-pepper map).
    """
    curve = np.asarray(curve, dtype=float)
    nans = np.where(np.isnan(curve))[0]
    if len(nans) > 0:
        curve = curve[: nans[0]]
    if curve.size == 0:
        return float("nan")
    if np.ptp(curve) == 0:
        return 1.0
    return float((np.max(curve) - curve[0]) / np.max(curve))


# -------------------------------------------------
# Plots
# -------------------------------------------------

def plot_cv_distribution(cv, savepath=None, figsize=(5, 3.4)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(cv[np.isfinite(cv)], bins=np.linspace(0, 1, 41), color="0.35",
            edgecolor="white", linewidth=0.5)
    ax.set_xlabel("circular variance (0 = sharply tuned, 1 = untuned)")
    ax.set_ylabel("units")
    ax.set_title(f"Orientation tuning strength (median CV {np.nanmedian(cv):.3f})")
    sns.despine(ax=ax)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_delta_vs_distance(mid, curves, sh_curves, lam_px, savepath=None,
                           macaque=None, tdann=None, label="Topo-Omni",
                           figsize=(5.6, 3.9)):
    """Panel E of the TDANN figure: distance on the x axis in % of column spacing.

    ``macaque`` is an optional ``(x_in_column_spacings, curves)`` pair from
    ``macaque_map.macaque_delta_curve``. Both axes are distance as a fraction of each map's
    own column spacing, which is what makes the two commensurable given that our sheet has
    no millimetre scale.
    """
    x = 100.0 * mid / lam_px
    m, s = np.nanmean(curves, axis=0), sem(curves, axis=0, nan_policy="omit")
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, m, color="#5B2C9E", lw=1.9, label=label)
    ax.fill_between(x, m - s, m + s, color="#5B2C9E", alpha=0.25, linewidth=0)

    for ref, colour, name in ((macaque, "#1B7F3B", "Macaque V1"),
                              (tdann, "#C25E00", "TDANN")):
        if ref is None:
            continue
        rx, rc = ref
        rm, rs = np.nanmean(rc, axis=0), sem(rc, axis=0, nan_policy="omit")
        ax.plot(100.0 * rx, rm, color=colour, lw=1.9, label=name)
        ax.fill_between(100.0 * rx, rm - rs, rm + rs, color=colour,
                        alpha=0.22, linewidth=0)

    ax.plot(x, np.nanmean(sh_curves, axis=0), color="0.5", lw=1.4, ls="--",
            label="distance-shuffled")
    ax.axhline(45.0, color="0.75", ls=":", lw=1.0)
    ax.set_xlabel("pairwise distance (% column spacing)")
    ax.set_ylabel("$\\Delta$ preferred orientation (deg)")
    bits = [f"{label.replace(chr(10), ' ')} {smoothness(m):.3f}"]
    if macaque is not None:
        bits.append(f"Macaque {smoothness(np.nanmean(macaque[1], axis=0)):.3f}")
    if tdann is not None:
        bits.append(f"TDANN {smoothness(np.nanmean(tdann[1], axis=0)):.3f}")
    ax.set_title("Smoothness: " + " | ".join(bits), fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    sns.despine(ax=ax)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_pinwheels(ori_map, centres, savepath=None, figsize=(7, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(ori_map, cmap="hsv", vmin=0, vmax=180, interpolation="nearest")
    if len(centres) and centres[0].size:
        ax.scatter(centres[0][:, 1], centres[0][:, 0], c="k", s=14, linewidths=0,
                   label=f"+{len(centres[0])}")
    if len(centres) > 1 and centres[1].size:
        ax.scatter(centres[1][:, 1], centres[1][:, 0], c="w", s=14,
                   edgecolors="k", linewidths=0.4, label=f"-{len(centres[1])}")
    n = sum(c.shape[0] for c in centres)
    ax.set_title(f"Smoothed orientation map — {n} pinwheels")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_power_spectrum(freqs, power, lam_px, savepath=None, figsize=(5, 3.2)):
    fig, ax = plt.subplots(figsize=figsize)
    keep = freqs > 0
    ax.plot(freqs[keep], power[keep], color="0.25", lw=1.4)
    ax.axvline(1.0 / lam_px, color="crimson", ls="--", lw=1.2,
               label=f"peak: $\\Lambda$ = {lam_px:.1f} px")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("spatial frequency (cycles/pixel)")
    ax.set_ylabel("radially averaged power")
    ax.set_title("Orientation-map power spectrum")
    ax.legend(frameon=False, fontsize=8)
    sns.despine(ax=ax)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig, ax


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--stats", type=str,
                    default="results/topo-omni/orientation/orientation_stats.npz")
    ap.add_argument("--outdir", type=str, default=None,
                    help="defaults to a 'metrics' folder beside --stats")
    ap.add_argument("--column-spacing", type=float, default=None,
                    help="override the estimated column spacing, in sheet pixels")
    ap.add_argument("--smooth-window-frac", type=float, default=0.15,
                    help="smoothing window width as a fraction of the column spacing. Keep "
                         "this well below 1: a window approaching Lambda averages away the "
                         "180 deg turn at a pinwheel (see --sweep output)")
    ap.add_argument("--winding-thresh", type=float, default=0.3)
    ap.add_argument("--sd-thresh-deg", type=float, default=30.0)
    ap.add_argument("--bank", type=str, default="ori_bias",
                    help="which grating bank the responses came from (ori_bias | unbiased); "
                         "selects the matching TDANN reference curve so both sides are "
                         "driven by the same stimuli")
    ap.add_argument("--layers", type=str, default="all",
                    help="restrict the CV histogram and the Delta-orientation-vs-distance "
                         "curve to these vision-encoder layers (0 = earliest): 'all', "
                         "'0-3', '0,3,5'. Column spacing and the pinwheel metrics always "
                         "use the full band -- a few layers is far too thin vertically to "
                         "estimate a 2D periodicity or count defects in.")
    ap.add_argument("--open-loop", action="store_true",
                    help="omit the closing increment, reproducing TDANN's winding numbers")
    args = ap.parse_args()

    stats_path = Path(args.stats)
    outdir = Path(args.outdir) if args.outdir else stats_path.parent / "metrics"
    outdir.mkdir(parents=True, exist_ok=True)

    d = np.load(stats_path, allow_pickle=True)
    pref_ori, curve, ori_deg = d["pref_ori"], d["tuning_curve"], d["ori_deg"]
    gate = d["gate"] if "gate" in d.files else d["tuned"]
    print(f"> Loaded {stats_path}  ({pref_ori.size} units, "
          f"criterion '{d['criterion'] if 'criterion' in d.files else 'fdr'}')")

    # ---- circular variance
    cv = circular_variance(curve, ori_deg)
    band_gate = band_grid(gate.astype(float)) > 0.5
    cv_band = band_grid(cv)[band_gate]   # full band, for the summary below
    print(f"\n> Circular variance (vision band, gated: {band_gate.sum()} units)")
    print(f"    median {np.nanmedian(cv_band):.3f} | "
          f"percentiles 5/25/75/95 "
          f"{np.round(np.nanpercentile(cv_band, [5, 25, 75, 95]), 3)}")
    print(f"    (TDANN panel B shows example curves at CV 0.19 / 0.41 / 0.66 / 0.98)")

    # ---- the map, restricted to the vision band
    ori_band = band_grid(pref_ori)
    valid = band_gate & np.isfinite(ori_band)

    # Layer restriction applies to the unit-level statistics only. Lambda, the smoothed map
    # and the pinwheel counts stay on the full band: 4 layers is 20 sheet rows, well under
    # one column spacing tall, so a 2D periodicity or defect count from that strip would be
    # meaningless.
    sel_layers = parse_layers(args.layers)
    lay_mask = layer_band_mask(sel_layers)
    valid_curve = valid & lay_mask
    layer_tag = ("all layers" if len(sel_layers) == N_VISION_LAYERS
                 else f"layers {sel_layers.min()}-{sel_layers.max()}")
    if len(sel_layers) < N_VISION_LAYERS:
        rows = np.nonzero(lay_mask.any(axis=1))[0]
        print(f"\n> Restricted to {layer_tag} ({len(sel_layers)}/{N_VISION_LAYERS}): "
              f"{valid_curve.sum()} of {valid.sum()} units, band rows "
              f"{rows.min()}-{rows.max()} ({len(rows)} rows tall)")

    # ---- column spacing
    lam, freqs, power = column_spacing(ori_band, valid)
    if args.column_spacing:
        print(f"\n> Column spacing: using override {args.column_spacing:.2f} px "
              f"(estimate was {lam:.2f})")
        lam = args.column_spacing
    else:
        print(f"\n> Column spacing Lambda = {lam:.2f} sheet px "
              f"(peak of the radially averaged power spectrum)")
        print(f"    band is {ori_band.shape[0]}x{ori_band.shape[1]} px "
              f"= {ori_band.shape[0]/lam:.1f} x {ori_band.shape[1]/lam:.1f} hypercolumns")

    # ---- is the band even a valid 2D cortical sheet?
    aniso = band_anisotropy(ori_band, valid)
    print(f"\n> Band geometry caveat: the row axis is DEPTH "
          f"({ROWS_PER_VISION_LAYER} rows per vision layer x 32 layers), not tangential "
          f"cortex; only the column axis is tangential.")
    print(f"    dominant period along rows (depth) {aniso['period_rows_px']:.1f} px vs "
          f"columns (tangential) {aniso['period_cols_px']:.1f} px "
          f"-> anisotropy {aniso['period_rows_px']/aniso['period_cols_px']:.2f}")
    print(f"    neighbouring-row Delta orientation: "
          f"{aniso['delta_within_layer_deg']:.2f} deg within a layer vs "
          f"{aniso['delta_across_layer_deg']:.2f} deg across a layer boundary")
    if abs(aniso["delta_within_layer_deg"] - aniso["delta_across_layer_deg"]) < 1.0:
        print(f"    -> indistinguishable, so the depth axis at least behaves homogeneously "
              f"and a 2D reading is defensible")
    else:
        print(f"    -> they differ; the row axis is NOT homogeneous and Lambda / pinwheel "
              f"density should be treated as descriptive only")

    # ---- smoothed map, winding numbers, pinwheels
    width = max(3, int(round(args.smooth_window_frac * lam)))
    stride = max(1, int(round(0.3 * width)))
    ori_sm, sd_sm, n_sm = smoothed_orientation_map(ori_band, valid, width, stride)
    print(f"\n> Smoothed map {ori_sm.shape} (window {width} px, stride {stride} px, "
          f"{lam/stride:.1f} px per Lambda); "
          f"{np.median(n_sm[n_sm > 0]):.0f} units/window (median)")
    print(f"    circular sd (deg): percentiles 5/25/50/75/95 "
          f"{np.round(np.nanpercentile(sd_sm[np.isfinite(sd_sm)], [5,25,50,75,95]), 1)}"
          f"  -> cut at {args.sd_thresh_deg}")
    if lam / stride < 8:
        print(f"    WARNING: only {lam/stride:.1f} px per column spacing. A pinwheel's "
              f"180 deg turn cannot be resolved by a 3x3 winding loop at this resolution; "
              f"lower --smooth-window-frac.")

    wn = winding_numbers(ori_sm, closed_loop=not args.open_loop)
    counts, centres = count_pinwheels(wn, sd_sm, thresh=args.winding_thresh,
                                      sd_thresh_deg=args.sd_thresh_deg)
    total = sum(counts)
    density = pinwheel_density(total, ori_sm.shape, stride, lam)
    print(f"\n> Pinwheels: {counts[0]} positive + {counts[1]} negative = {total}")
    print(f"    density {density:.2f} per Lambda^2  "
          f"[TDANN ~2.1, macaque V1 ~3.0, hand-crafted SOM ~4.0]")
    print(f"    winding loop: {'open (TDANN-exact)' if args.open_loop else 'closed'}")
    print(f"    |winding| reaches {np.nanmax(np.abs(wn)):.2f} "
          f"({100*np.mean(np.abs(wn) > args.winding_thresh):.1f}% of pixels above "
          f"{args.winding_thresh}) -- 0.5 is a full pinwheel")

    # ---- and how much that density depends on the smoothing, which is a lot
    print(f"\n> Pinwheel density is strongly smoothing-dependent; never quote it without "
          f"the window. Rows are window/Lambda, columns are the circular-sd cut:")
    sweep = pinwheel_sensitivity(ori_band, valid, lam)
    print(f"    {'w/Λ':>5} {'win':>4} {'strd':>5} {'px/Λ':>6} {'sd50':>6} | "
          + " ".join(f"{('sd<%g' % s) if s < 1e8 else 'no cut':>11}" for s in (30., 45., 60., 1e9)))
    for e in sweep:
        cells = " ".join(f"{e['counts'][s][0]:4d}/{e['counts'][s][1]:5.2f}"
                         for s in (30., 45., 60., 1e9))
        print(f"    {e['frac']:5.2f} {e['width']:4d} {e['stride']:5d} "
              f"{e['px_per_lambda']:6.1f} {e['sd_median']:6.1f} |  {cells}")
    print("    cells are count/density-per-Lambda^2")

    # ---- delta orientation vs distance + smoothness
    # Use TDANN's own binning (9 bins reaching 0.943*Lambda) rather than an arbitrary
    # 20 bins to Lambda: `smoothness` is (max - curve[0]) / max, so it moves with the width
    # of the first bin, and only matched binning makes our value comparable to the macaque
    # curve and to their published number.
    from src.experiments.orientation.macaque_map import tdann_bin_edges, macaque_delta_curve
    edges = tdann_bin_edges(lam)
    yy, xx = np.nonzero(valid_curve)
    positions = np.stack([xx, yy], axis=1).astype(float)
    if len(sel_layers) < N_VISION_LAYERS:
        # With only a few layers the strip is far wider than it is tall, so pairs out at
        # 0.943*Lambda are almost all separated along the tangential axis. That is arguably
        # the better comparison to macaque, whose axes are both tangential -- but it does
        # mean this curve is not the same object as the full-band one.
        vspan = yy.max() - yy.min()
        print(f"    curve sampled over a {vspan + 1} x {xx.max() - xx.min() + 1} px strip; "
              f"bins reach {edges[-1]:.1f} px, so pairs beyond {vspan} px are purely "
              f"tangential")
    mid, curves = metric_difference_over_distance(
        ori_band[valid_curve], positions, distance_cutoff=edges[-1], bin_edges=edges)
    _, sh_curves = metric_difference_over_distance(
        ori_band[valid_curve], positions, distance_cutoff=edges[-1], bin_edges=edges,
        shuffle=True)
    print(f"\n> Distance bins: TDANN scheme, {len(edges)-1} bins reaching "
          f"{edges[-1]:.1f} px = {edges[-1]/lam:.3f} column spacings")
    sm_vals = np.array([smoothness(c) for c in curves])
    sh_vals = np.array([smoothness(c) for c in sh_curves])
    print(f"\n> Map smoothness {np.nanmean(sm_vals):.3f} +/- {sem(sm_vals, nan_policy='omit'):.3f}"
          f"  (distance-shuffled control {np.nanmean(sh_vals):.3f})")
    print(f"    [macaque V1 ~0.65 in TDANN panel E; higher = smoother map]")

    # ---- figures
    plot_cv_distribution(cv_band, savepath=outdir / "circular_variance.png")
    plot_power_spectrum(freqs, power, lam, savepath=outdir / "power_spectrum.png")
    plot_pinwheels(ori_sm, centres, savepath=outdir / "pinwheels.png")
    # Macaque V1 reference curve, decoded from the Nauhaus 2012 figure the same way TDANN
    # does it. Optional: needs one network fetch on first run.
    try:
        mac_x, mac_curves = macaque_delta_curve()
        mac_smoo = np.nanmean([smoothness(c) for c in mac_curves])
        print(f"> Macaque V1 reference: smoothness {mac_smoo:.3f} "
              f"(TDANN report ~0.65); ours {np.nanmean(sm_vals):.3f}")
        macaque = (mac_x, mac_curves)
    except Exception as err:                                  # noqa: BLE001
        print(f"> Macaque reference unavailable ({type(err).__name__}: {err}); "
              f"plotting without it")
        macaque = None

    # TDANN's own model, run on these same gratings (see tdann_model.py). Optional: needs
    # a one-off checkpoint download.
    try:
        from src.experiments.orientation.tdann_model import tdann_delta_curve
        td_x, td_curves = tdann_delta_curve(bank=args.bank)
        td_smoo = np.nanmean([smoothness(c) for c in td_curves])
        print(f"> TDANN reference (bank '{args.bank}'): smoothness {td_smoo:.3f} "
              f"(published ~0.72); "
              f"ours {np.nanmean(sm_vals):.3f}")
        tdann = (td_x, td_curves)
    except Exception as err:                                  # noqa: BLE001
        print(f"> TDANN reference unavailable ({type(err).__name__}: {err})")
        tdann = None

    plot_delta_vs_distance(mid, curves, sh_curves, lam, macaque=macaque, tdann=tdann,
                           label=f"Topo-Omni ({layer_tag})",
                           savepath=outdir / "delta_orientation_vs_distance.png")

    np.savez(
        outdir / "orientation_metrics.npz",
        cv=cv, column_spacing_px=lam, smoothed_map=ori_sm, smoothed_sd=sd_sm,
        winding_numbers=wn, pinwheel_counts=np.array(counts),
        pinwheel_density=density, pinwheel_centers_pos=centres[0],
        pinwheel_centers_neg=centres[1] if len(centres) > 1 else np.zeros((0, 2)),
        distance_midpoints=mid, delta_curves=curves, shuffled_curves=sh_curves,
        smoothness=sm_vals, smoothness_shuffled=sh_vals,
        smooth_window_px=width, smooth_stride_px=stride,
        anisotropy=np.array([aniso[k] for k in sorted(aniso)]),
        anisotropy_keys=np.array(sorted(aniso)),
        sweep_fracs=np.array([e["frac"] for e in sweep]),
        sweep_density_sd30=np.array([e["counts"][30.0][1] for e in sweep]),
        sweep_density_nocut=np.array([e["counts"][1e9][1] for e in sweep]),
    )
    print(f"\n> Saved metrics + figures to: {outdir}")


if __name__ == "__main__":
    main()
