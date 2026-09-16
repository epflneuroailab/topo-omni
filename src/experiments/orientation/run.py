"""
Plot orientation preference and orientation selectivity on the Topo-Omni
cortical sheet, masking units that are not significantly orientation-tuned.

Drives the model with the TDANN "orientation bias" sine-grating bank (Margalit et al.),
hosted on OSF at https://osf.io/64qv3 under
``tdann_data/datasets/sine_grating_images_ori_bias`` (510 full-field gratings), and renders:

  * preferred orientation   -- circular (axial, mod-180 deg) vector average of each unit's
                               orientation tuning curve, drawn with a cyclic colormap
  * orientation selectivity -- resultant length of that same vector average (OSI):
                               0 = flat/unoriented, 1 = perfectly orientation tuned.
                               Computed on the baseline-subtracted tuning curve; see
                               ``orientation_preference`` for why that beats rectifying at 0.
  * the distribution of preferred orientations over tuned units, against the (deliberately
    non-uniform) orientation distribution of the stimulus bank

Stimulus filenames encode the parameters as
``fullfield_<ori>deg_<sf>sf_<phase>phase_<bw|rg>.jpg``. Spatial frequency, phase and colour
are nuisance dimensions: they are averaged over *within* each orientation, so a unit's
tuning curve has exactly one entry per unique orientation.

NOTE -- this bank is orientation *biased* by construction: cardinal orientations are
oversampled relative to the obliques (0 deg: 79 stimuli, 90 deg: 80, against 45 deg: 47 and
135 deg: 48). Averaging within each orientation *before* the circular average is what keeps
that sampling bias out of the preferred-orientation estimate -- a vector average taken
directly over the 510 raw stimuli would be pulled toward the cardinals. For the same reason
the orientation groups are kept as a list of unequal-length blocks rather than reshaped into
a rectangular (n_ori, n_exemplars, n_units) array the way the retinotopy/tonotopy scripts do.

Which units count as tuned
--------------------------
``stats.criterion`` picks the statistic that gates and shades the maps. All three are
computed and saved every run, so the figures can be re-cut on a different one cheaply:

``reliability`` (default)
    Correlation between the orientation tuning curves measured on two halves of the stimuli,
    where the halves are split *along* a nuisance dimension (``stats.reliability_split``:
    ``sf``, ``phase``, ``colour``, or ``random``). Asks whether a unit's orientation
    preference *generalises* across that dimension -- the question these maps actually rest
    on. Use a structured split, not ``random``: random halves draw the same mixture and so
    only measure sampling noise, scoring r ~ 0.97 even for a unit whose preferred
    orientation shifts 45 deg between low and high spatial frequency.
``eta2``
    Fraction of the unit's response variance explained by orientation. An effect size, so
    it does not grow with the stimulus count.
``fdr``
    The FDR-corrected ANOVA p-value.

Prefer one of the first two. A p-value is a poor gate here, and not because of the
multiple-comparisons correction: measured on this sheet, the entire span from uncorrected p
to Bonferroni is 5.2 orders of magnitude (necessarily log10 of the 155648 tests), so no
choice of correction moves the threshold far -- even Bonferroni leaves ~82% of the vision
band "significant" at 0.05. Two things actually inflate the p-values. The model is
deterministic, so the ANOVA error term is nuisance variance rather than trial noise and
"the orientation effect is exactly zero" is essentially never true; and the test runs on
the *smoothed* sheet, which buys ~13 orders of magnitude at the median (49% -> 99% of the
band significant) because near-silent units inherit their neighbours' tuning. Corrections
that model the smoothing-induced dependence (RFT, cluster permutation) make the threshold
*more* lenient, not less. Reaching for a tiny alpha to compensate is really an effect-size
filter in disguise -- so use an effect size.

Inputs
------
pref_ori    : (n_units,) float  preferred orientation in degrees, in [0, 180)
osi         : (n_units,) float  orientation selectivity index in [0, 1]
tuned       : (n_units,) bool   passed the FDR-corrected ANOVA at ``stats.alpha``
qvals       : (n_units,) float  FDR-corrected p-values
eta2        : (n_units,) float  variance explained by orientation
reliability : (n_units,) float  split-half tuning-curve reliability
gate        : (n_units,) bool   passed the *selected* criterion
opacity     : (n_units,) float  per-unit alpha from the selected criterion, so every map
                                fades each unit by its tuning strength instead of showing a
                                flat pass/fail mask (see ``criterion_opacity``)
coords      : (n_units, 2)      unit positions on the sheet, (depth_row, sheet_col)

The bank is downloaded from OSF on first run into ``$STIMULI_DIR/sine_grating_ori_bias``
(idempotent -- files already present at the right size are not re-fetched).

Usage
-----
python -m src.experiments.orientation.run --config src/configs/eval/orientation.yml
"""

import os
import re
import json
import time
import shutil
import random
import threading
import urllib.error
import urllib.request
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.models.spatial_utils import DEFAULT_SHEET_LAYOUT as _SHEET
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import seaborn as sns
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from scipy.stats import f_oneway, false_discovery_control

from qwen_omni_utils import process_mm_info
from src.core.model_loading import load_topo_omni
from src.utils.smoothing import NeuronSmoothingConv

from dotenv import load_dotenv
load_dotenv()

# -------------------------------------------------
# OSF stimulus bank (https://osf.io/64qv3)
# -------------------------------------------------

OSF_NODE = "64qv3"
# tdann_data/datasets/sine_grating_images_ori_bias
OSF_ORI_BIAS_FOLDER = "6466435cf9c6ce0b5fd4b8fd"

# The OSF node hosts two grating banks under tdann_data/datasets. Both use the same
# filename grammar; they differ only in how orientations are sampled:
#   ori_bias  510 files, cardinals oversampled (0 deg: 79, 90 deg: 80, vs 45 deg: 47,
#             135 deg: 48) -- the "orientation bias" set.
#   unbiased  638 files, ~uniform (79-80 per orientation; 2 short of the full 8x8x5x2 grid)
#             -- sine_grating_images_20190507, the set TDANN's published figures use.
GRATING_BANKS = {
    "ori_bias": (OSF_ORI_BIAS_FOLDER, "sine_grating_ori_bias"),
    "unbiased": ("646640c78338be0d4412b069", "sine_grating_unbiased"),
}
OSF_LIST_URL = (
    "https://api.osf.io/v2/nodes/{node}/files/osfstorage/{folder}/?page%5Bsize%5D=100"
)

# fullfield_22.5deg_18.6sf_0.0phase_bw.jpg
STIM_RE = re.compile(r"fullfield_([\d.]+)deg_([\d.]+)sf_([\d.]+)phase_(bw|rg)\.jpg$")

_ORI_TICKS = [0, 45, 90, 135, 180]

# The unified sheet is (n_tokens, 309, 512); the vision encoder occupies depth rows 0-164
# and sheet columns 0-255 (the audio encoder shares those rows in columns 256-511, the
# thinker/LM takes rows 165-308). Each encoder band is 33 slots of 5 rows: its conv
# front-end (vision `patch_embed`, audio `conv2`) as slot 0, then its 32 blocks.
# Read from the layout rather than hard-coded, so a geometry change raises instead of
# silently reshaping into a scrambled map. See src/training/init_coords.py.
SHEET_DEPTH, SHEET_COLS = _SHEET.n_row, _SHEET.n_col
VISION_DEPTH_ROWS, VISION_SHEET_COLS = _SHEET.encoder_rows, _SHEET.n_col // 2
N_VISION_SLOTS = _SHEET.num_encoder_layers


def _osf_get_json(url: str, retries: int = 4, timeout: int = 60) -> dict:
    """GET a JSON document from the OSF API, retrying transient failures."""
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as err:
            last = err
            time.sleep(2 ** attempt)
    raise RuntimeError(f"could not fetch {url}: {last}")


def list_osf_folder(node: str = OSF_NODE, folder: str = OSF_ORI_BIAS_FOLDER) -> List[dict]:
    """List every file in an OSF folder as {name, size, download}, following pagination.

    OSF's pagination is not stably ordered, so the same file can surface on two pages
    (the ori-bias folder reports 511 entries for 510 distinct files). Deduplicate by
    file id so the stimulus count is right.
    """
    url = OSF_LIST_URL.format(node=node, folder=folder)
    seen: Dict[str, dict] = {}
    while url:
        page = _osf_get_json(url)
        for item in page["data"]:
            if item["attributes"]["kind"] != "file":
                continue
            seen[item["id"]] = {
                "name": item["attributes"]["name"],
                "size": int(item["attributes"]["size"]),
                "download": item["links"]["download"],
            }
        url = page["links"].get("next")
    return sorted(seen.values(), key=lambda e: e["name"])


def _fetch_one(entry: dict, dest: Path, overwrite: bool) -> bool:
    """Download a single stimulus; returns True if it was actually fetched."""
    out = dest / entry["name"]
    if not overwrite and out.is_file() and out.stat().st_size == entry["size"]:
        return False

    # Unique temp name per worker so concurrent retries cannot collide.
    tmp = out.with_suffix(out.suffix + f".part{os.getpid()}-{threading.get_ident()}")
    try:
        # OSF rate-limits parallel downloads with HTTP 429, so back off and retry rather
        # than failing the whole bank. Honour Retry-After when the server sends it.
        last = None
        for attempt in range(8):
            try:
                req = urllib.request.Request(entry["download"], headers={"Accept": "*/*"})
                with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as fh:
                    shutil.copyfileobj(resp, fh)
                break
            except urllib.error.HTTPError as err:
                last = err
                if err.code not in (429, 500, 502, 503, 504):
                    raise
                wait = float(err.headers.get("Retry-After") or 0) or min(2 ** attempt, 60)
                time.sleep(wait + random.uniform(0, 1.0))       # jitter, avoid lockstep
            except (urllib.error.URLError, TimeoutError) as err:
                last = err
                time.sleep(min(2 ** attempt, 60) + random.uniform(0, 1.0))
        else:
            raise RuntimeError(f"gave up after retries: {last}")

        got = tmp.stat().st_size
        if got != entry["size"]:
            raise ValueError(
                f"size mismatch for {entry['name']}: "
                f"OSF says {entry['size']}, downloaded {got}"
            )
        tmp.replace(out)          # atomic, so an interrupted run never leaves a partial jpg
    finally:
        tmp.unlink(missing_ok=True)
    return True


def fetch_grating_bank(
    dest: Path, bank: str = "ori_bias", overwrite: bool = False, workers: int = 8
) -> List[Path]:
    """Download one of the OSF grating banks (see ``GRATING_BANKS``) into ``dest``.

    Idempotent: a file already on disk with the size OSF reports is left alone, so an
    interrupted download resumes cheaply. Also writes a ``manifest.csv`` alongside the
    images, matching the other stimulus banks in this repo.

    The 510 stimuli are ~13 KB each, so this is entirely round-trip-latency bound (OSF
    redirects every download to a CDN); fetching them serially takes well over an hour,
    hence the thread pool.
    """
    if bank not in GRATING_BANKS:
        raise ValueError(f"data.bank must be one of {sorted(GRATING_BANKS)} (got {bank!r})")
    dest.mkdir(parents=True, exist_ok=True)
    entries = list_osf_folder(folder=GRATING_BANKS[bank][0])
    print(f"> OSF lists {len(entries)} grating stimuli in bank '{bank}'")

    todo = [
        e for e in entries
        if overwrite or not (dest / e["name"]).is_file()
        or (dest / e["name"]).stat().st_size != e["size"]
    ]
    print(f"> {len(entries) - len(todo)} already present, fetching {len(todo)} "
          f"with {workers} workers")

    def _missing(entries_):
        return [e for e in entries_
                if not (dest / e["name"]).is_file()
                or (dest / e["name"]).stat().st_size != e["size"]]

    # OSF throttles per IP over a time window, so no amount of lowering concurrency avoids
    # 429 once the window is spent -- the fix is to keep resuming. Each pass is idempotent,
    # so progress accumulates; give up only when a pass fetches nothing twice running.
    fetched, stalled = 0, 0
    for round_no in range(1, 13):
        pending = todo if (round_no == 1 and overwrite) else _missing(entries)
        if not pending:
            break
        got_this_round = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_fetch_one, e, dest, overwrite): e for e in pending}
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=f"downloading stimuli (pass {round_no})"):
                try:
                    got_this_round += bool(fut.result())
                except Exception:                      # noqa: BLE001 - retried next pass
                    pass
        fetched += got_this_round
        remaining = len(_missing(entries))
        print(f"    pass {round_no}: +{got_this_round}, {remaining} still missing")
        if remaining == 0:
            break
        stalled = stalled + 1 if got_this_round == 0 else 0
        if stalled >= 2:
            raise RuntimeError(
                f"download stalled with {remaining} files missing; OSF is rate-limiting. "
                f"Re-run later to resume (already-fetched files are kept).")
        time.sleep(20.0)                               # let the rate-limit window recover

    still = _missing(entries)
    if still:
        raise RuntimeError(f"{len(still)} stimuli missing after retries, e.g. "
                           f"{[e['name'] for e in still[:3]]}")

    print(f"> Fetched {fetched} new stimuli ({len(entries) - fetched} already present) -> {dest}")

    manifest = dest / "manifest.csv"
    rows = ["filename,orientation_deg,spatial_freq,phase,color"]
    for entry in entries:
        m = STIM_RE.search(entry["name"])
        if m:
            rows.append(
                f"{entry['name']},{float(m.group(1))},{float(m.group(2))},"
                f"{float(m.group(3))},{m.group(4)}"
            )
    manifest.write_text("\n".join(rows) + "\n")

    return sorted(dest.glob("*.jpg"))


def parse_stimuli(
    stimuli_root: Path, color: str = "all"
) -> Tuple[List[str], np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Collect grating paths, their orientations, and their nuisance parameters.

    Returns:
        paths   : list of image paths, grouped so that all stimuli of one orientation are
                  contiguous and orientations ascend
        ori_deg : (n_ori,) the unique orientations present, in degrees, ascending
        counts  : (n_ori,) how many stimuli each orientation contributes (unequal by design)
        meta    : dict of (n_stimuli,) arrays 'sf', 'phase', 'colour', aligned with ``paths``.
                  Kept so reliability can be split *along* a nuisance dimension rather than
                  at random -- see ``split_half_reliability``.
    """
    if color not in ("all", "bw", "rg"):
        raise ValueError(f"data.color must be one of 'all', 'bw', 'rg' (got {color!r})")

    rows = []
    for path in sorted(stimuli_root.glob("*.jpg")):
        m = STIM_RE.search(path.name)
        if m is None:
            continue
        if color != "all" and m.group(4) != color:
            continue
        rows.append((float(m.group(1)), float(m.group(2)), float(m.group(3)),
                     m.group(4), str(path)))

    if not rows:
        raise FileNotFoundError(
            f"no gratings matching {STIM_RE.pattern!r} (colour {color!r}) under {stimuli_root}"
        )

    # Order must stay byte-for-byte what earlier runs used, or a cached cortical_sheets.npy
    # would be silently re-labelled: sort by filename, then *stably* by orientation, which
    # keeps filename order inside each orientation exactly as before.
    rows.sort(key=lambda r: r[4])
    rows.sort(key=lambda r: r[0])

    ori_all = np.array([r[0] for r in rows], dtype=float)
    ori_deg = np.unique(ori_all)
    counts = np.array([(ori_all == o).sum() for o in ori_deg], dtype=int)
    paths = [r[4] for r in rows]
    meta = {
        "sf": np.array([r[1] for r in rows], dtype=float),
        "phase": np.array([r[2] for r in rows], dtype=float),
        "colour": np.array([r[3] for r in rows]),
    }
    return paths, ori_deg, counts, meta


# -------------------------------------------------
# Orientation statistics (axial / mod-180 circular)
# -------------------------------------------------

def orientation_tuning_curve(
    responses: np.ndarray, counts: np.ndarray
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Average responses within each orientation.

    responses : (n_stimuli, n_units) with stimuli ordered by ascending orientation
    counts    : (n_ori,) stimuli per orientation, summing to n_stimuli

    Returns:
        curve  : (n_ori, n_units) mean response per orientation
        groups : list of (n_i, n_units) per-orientation response blocks, for the ANOVA
    """
    if responses.shape[0] != counts.sum():
        raise ValueError(
            f"{responses.shape[0]} responses does not match {counts.sum()} stimuli"
        )
    groups = np.split(responses, np.cumsum(counts)[:-1], axis=0)
    curve = np.stack([g.mean(axis=0) for g in groups], axis=0)
    return curve, groups


def orientation_preference(
    curve: np.ndarray, ori_deg: np.ndarray, baseline: str = "min"
) -> Tuple[np.ndarray, np.ndarray]:
    """Circular vector average of an orientation tuning curve.

    Orientation is axial (a grating at 10 deg and one at 190 deg are identical), so the
    average is taken over the doubled angle 2*theta and halved again -- the standard
    orientation-preference / OSI estimator.

    A vector average needs non-negative weights, but sheet activations are roughly
    zero-centred (about 45% of units have at least one negative entry), so the curve has to
    be made non-negative first. ``baseline`` picks how:

    ``"min"`` (default)
        Subtract each unit's least-preferred response. This is a *constant* shift, and the
        8 orientations here are evenly spaced over 180 deg, so ``sum(exp(2j*theta)) == 0``
        and the shift cancels exactly out of the vector sum: the preferred orientation is
        then identical to the one the raw (unclipped) curve would give, and only the
        resultant length is affected. Modulation depth is preserved in full.
    ``"zero"``
        Rectify at zero, appropriate when 0 is a meaningful floor (as for firing rates, and
        as the tonotopy best-frequency estimator does). Note this is a *nonlinear* clip, so
        for units with negative entries it distorts the curve's shape and moves the
        preferred orientation -- by up to ~50 deg on this sheet.

    curve    : (n_ori, n_units) mean response per orientation
    ori_deg  : (n_ori,) orientations in degrees
    baseline : "min" | "zero"

    Returns:
        pref_ori : (n_units,) preferred orientation in degrees, in [0, 180)
        osi      : (n_units,) resultant length in [0, 1]
    """
    if baseline == "min":
        w = curve - curve.min(axis=0, keepdims=True)
    elif baseline == "zero":
        w = np.clip(curve, 0.0, None)
    else:
        raise ValueError(f"stats.baseline must be 'min' or 'zero' (got {baseline!r})")
    total = w.sum(axis=0)

    theta2 = 2.0 * np.deg2rad(np.asarray(ori_deg, dtype=float))
    cos_term = (w * np.cos(theta2)[:, None]).sum(axis=0)
    sin_term = (w * np.sin(theta2)[:, None]).sum(axis=0)

    pref_ori = np.rad2deg(0.5 * np.arctan2(sin_term, cos_term)) % 180.0
    with np.errstate(invalid="ignore", divide="ignore"):
        osi = np.where(total > 0, np.hypot(cos_term, sin_term) / total, 0.0)

    # A unit that never responds above zero has no defined preference.
    pref_ori = np.where(total > 0, pref_ori, np.nan)
    return pref_ori, osi


def anova_orientation(
    groups: List[np.ndarray], alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-unit one-way ANOVA of response against orientation, FDR-corrected.

    ``groups`` holds one (n_i, n_units) block per orientation; the blocks have unequal
    n_i for this bank, which f_oneway handles natively.

    Returns:
        tuned : (n_units,) bool, corrected p-value below ``alpha``
        qvals : (n_units,) the corrected p-values themselves, so the maps can grade each
                unit by the strength of its tuning evidence rather than just pass/fail
    """
    _, p_values = f_oneway(*groups)
    p_values = np.where(np.isnan(p_values), 1.0, p_values)
    qvals = false_discovery_control(p_values)
    return qvals < alpha, qvals


def eta_squared(
    responses: np.ndarray, counts: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Fraction of each unit's response variance explained by orientation.

    A one-way effect size, so unlike a p-value it does not grow with the number of stimuli.
    That matters here: with 510 deterministic stimuli per unit (no trial-to-trial noise, so
    the ANOVA's error term is nuisance variance from spatial frequency / phase / colour
    rather than measurement noise), "the orientation effect is exactly zero" is essentially
    never true and significance is close to guaranteed. Effect size asks the useful question
    instead -- how *much* of this unit's response is about orientation.

    responses : (n_stimuli, n_units), stimuli ordered by ascending orientation
    counts    : (n_ori,) stimuli per orientation

    Returns:
        eta2   : (n_units,) SS_between / SS_total, in [0, 1]
        omega2 : (n_units,) the bias-corrected variant (can go slightly negative)
    """
    n_obs = int(counts.sum())
    k = len(counts)
    grand = responses.mean(axis=0)

    groups = np.split(responses, np.cumsum(counts)[:-1], axis=0)
    means = np.stack([g.mean(axis=0) for g in groups])                  # (n_ori, n_units)
    ss_b = (counts[:, None] * (means - grand) ** 2).sum(axis=0)
    # SS_total via the algebraic identity, to avoid a second full-size temporary.
    ss_t = np.einsum("ij,ij->j", responses, responses) - n_obs * grand ** 2
    ss_w = np.maximum(ss_t - ss_b, 0.0)
    ms_w = ss_w / (n_obs - k)

    with np.errstate(invalid="ignore", divide="ignore"):
        eta2 = np.where(ss_t > 0, ss_b / ss_t, 0.0)
        omega2 = np.where(ss_t + ms_w > 0, (ss_b - (k - 1) * ms_w) / (ss_t + ms_w), 0.0)
    return np.clip(eta2, 0.0, 1.0), omega2


def _curve_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Correlate two (n_ori, n_units) tuning curves per unit, on shape not amplitude."""
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    den = np.sqrt((a ** 2).sum(axis=0) * (b ** 2).sum(axis=0))
    return np.where(den > 0, (a * b).sum(axis=0) / np.maximum(den, 1e-30), 0.0)


def split_half_reliability(
    responses: np.ndarray,
    counts: np.ndarray,
    split_key: np.ndarray = None,
    n_rep: int = 10,
    seed: int = 0,
) -> np.ndarray:
    """Correlation between orientation tuning curves built from two disjoint stimulus halves.

    ``split_key`` decides what the number means, and the difference is not cosmetic:

    ``None`` -- random halves
        Measures reliability against *sampling noise only*. A random split cannot detect a
        preference that shifts systematically with a nuisance variable, because both halves
        draw from the same mixture and therefore estimate the same mixed tuning curve. A
        unit whose preferred orientation moves 45 deg between low and high spatial frequency
        still scores r ~ 0.97 here.
    ``(n_stimuli,)`` array of nuisance labels -- structured halves
        Splits within each orientation by that label (its two distinct values, or a median
        split if numeric with more than two), so the correlation measures whether the
        orientation preference *generalises across that nuisance dimension*. This is the
        question an orientation-preference map really rests on, and no p-value answers it: a
        unit can be overwhelmingly significant while its preferred orientation depends on
        spatial frequency, which is not orientation tuning in any useful sense. Deterministic,
        so ``n_rep`` is ignored.

    Returns:
        r : (n_units,) reliability in [-1, 1]
    """
    bounds = np.concatenate([[0], np.cumsum(counts)])
    n_ori = len(counts)

    if split_key is not None:
        split_key = np.asarray(split_key)
        if split_key.shape[0] != responses.shape[0]:
            raise ValueError(
                f"split_key has {split_key.shape[0]} entries for "
                f"{responses.shape[0]} stimuli")
        half_a, half_b = [], []
        for i in range(n_ori):
            idx = np.arange(bounds[i], bounds[i + 1])
            keys = split_key[idx]
            uniq = np.unique(keys)
            if uniq.size < 2:
                raise ValueError(
                    f"orientation index {i} has a single value of the split key "
                    f"({uniq!r}); cannot split along it")
            if uniq.size == 2:
                sel = keys == uniq[0]
            else:                                    # numeric with many levels -> median split
                sel = keys.astype(float) <= np.median(keys.astype(float))
            if sel.all() or not sel.any():
                raise ValueError(f"degenerate split at orientation index {i}")
            half_a.append(responses[idx[sel]].mean(axis=0))
            half_b.append(responses[idx[~sel]].mean(axis=0))
        return _curve_correlation(np.stack(half_a), np.stack(half_b))

    rng = np.random.default_rng(seed)
    total = np.zeros(responses.shape[1], dtype=np.float64)
    for _ in range(n_rep):
        half_a, half_b = [], []
        for i in range(n_ori):
            idx = rng.permutation(np.arange(bounds[i], bounds[i + 1]))
            h = len(idx) // 2
            half_a.append(responses[idx[:h]].mean(axis=0))
            half_b.append(responses[idx[h:2 * h]].mean(axis=0))
        total += _curve_correlation(np.stack(half_a), np.stack(half_b))
    return total / n_rep


# Criteria for "is this unit orientation tuned", all oriented so that larger = stronger.
CRITERIA = ("fdr", "eta2", "reliability")
_CRITERION_DEFAULT_THRESHOLD = {"eta2": 0.10, "reliability": 0.5}   # fdr uses stats.alpha
_CRITERION_LABEL = {
    "fdr": "$-\\log_{10}$(FDR q)",
    "eta2": "$\\eta^2$ explained by orientation",
    "reliability": "split-half tuning reliability",
}


def tuning_strength(
    criterion: str, alpha: float, threshold: float = None, *,
    qvals: np.ndarray = None, eta2: np.ndarray = None, reliability: np.ndarray = None,
) -> Tuple[np.ndarray, float, str]:
    """Pick the per-unit tuning-strength statistic and its cutoff.

    Everything is returned on a "larger = stronger" scale so one opacity ramp serves all
    three. For ``fdr`` that means -log10(q), which also makes the ramp logarithmic in q --
    necessary, because corrected p-values here span ~160 orders of magnitude.

    Returns:
        strength  : (n_units,) larger = better tuned
        cutoff    : units at or below this are not shown
        label     : axis annotation describing the scale
    """
    if criterion not in CRITERIA:
        raise ValueError(f"stats.criterion must be one of {CRITERIA} (got {criterion!r})")

    if criterion == "fdr":
        strength = -np.log10(np.clip(qvals, 1e-300, None))   # q can underflow to exactly 0
        cutoff = -np.log10(alpha)
    elif criterion == "eta2":
        strength = np.asarray(eta2, dtype=float)
        cutoff = _CRITERION_DEFAULT_THRESHOLD["eta2"] if threshold is None else threshold
    else:
        strength = np.asarray(reliability, dtype=float)
        cutoff = (_CRITERION_DEFAULT_THRESHOLD["reliability"]
                  if threshold is None else threshold)

    return strength, float(cutoff), _CRITERION_LABEL[criterion]


def criterion_opacity(
    strength: np.ndarray,
    cutoff: float,
    floor: float = 0.15,
    saturate_pct: float = 95.0,
    sel: np.ndarray = None,
) -> np.ndarray:
    """Map a tuning-strength statistic to a per-unit opacity in [0, 1].

    Stronger tuning -> more opaque; weaker evidence fades toward the grey background. Units
    at or below ``cutoff`` get opacity 0, so they drop out of the figure exactly as a
    boolean mask would drop them.

    Opacity rises linearly from ``floor`` at the cutoff to 1.0 at a saturation point taken
    as a percentile of the units being displayed, so the ramp adapts to the run instead of
    hard-coding a scale. (For the FDR criterion the strength is already -log10(q), so this
    linear ramp is logarithmic in q.)

    strength     : (n_units,) larger = better tuned
    cutoff       : strength at or below this -> fully transparent
    floor        : opacity of a unit sitting just above the cutoff, so it stays visible
    saturate_pct : percentile of strength (over ``sel``) that reaches opacity 1
    sel          : optional bool mask choosing which units set the saturation point; pass
                   the band being plotted so its own dynamic range drives the ramp
    """
    strength = np.asarray(strength, dtype=float)
    shown = strength > cutoff

    pool = strength[(sel & shown) if sel is not None else shown]
    hi = float(np.percentile(pool, saturate_pct)) if pool.size else cutoff + 1.0
    hi = max(hi, cutoff + 1e-12)                     # never divide by zero

    t = np.clip((strength - cutoff) / (hi - cutoff), 0.0, 1.0)
    return np.where(shown & np.isfinite(strength), floor + (1.0 - floor) * t, 0.0)


# -------------------------------------------------
# Sheet layout
# -------------------------------------------------

def vision_band(values: np.ndarray) -> np.ndarray:
    """Reshape a smoothed sheet vector to the vision-encoder band, as a 2D array.

    NeuronSmoothingConv rasterises by (y=coords[:, 1], x=coords[:, 0]), i.e. its grid is
    (512, 309) with the depth axis on the columns. Rotating a quarter turn puts depth back
    on the rows -- reversed, so depth row 0 (the earliest vision layer) lands on the last
    row -- which is why the vision band is the bottom-left block. Same transform as
    run_retinotopy_angle.py, so the maps stay comparable panel to panel.
    """
    grid = np.asarray(values, dtype=float).reshape(SHEET_COLS, SHEET_DEPTH)
    grid = np.rot90(grid, k=1)                       # -> (309, 512), depth on rows, flipped
    return grid
    return grid[SHEET_DEPTH - VISION_DEPTH_ROWS:, :VISION_SHEET_COLS]


def vision_mask() -> np.ndarray:
    """Boolean mask over smoothed-sheet units selecting the vision-encoder band.

    Same indexing as ``vision_band``: for smoother-grid flat index i, the depth coordinate
    is ``i % 309`` and the sheet column is ``i // 309``. Images never drive the audio band,
    so summary statistics over the whole sheet are dominated by units that saw no input --
    restrict them with this.
    """
    i = np.arange(SHEET_COLS * SHEET_DEPTH)
    return ((i % SHEET_DEPTH) < VISION_DEPTH_ROWS) & ((i // SHEET_DEPTH) < VISION_SHEET_COLS)


# -------------------------------------------------
# Plots
# -------------------------------------------------

def _ori_tick_cbar(cbar, label="preferred orientation (deg)"):
    cbar.set_ticks(_ORI_TICKS)
    cbar.set_ticklabels([str(t) for t in _ORI_TICKS])
    cbar.set_label(label)


def _band_labels(arr: np.ndarray) -> Tuple[str, str]:
    """Title/axis wording for whatever ``vision_band`` actually returned.

    ``vision_band`` can be toggled between the vision-encoder block and the whole rotated
    sheet, so derive the labels from the array shape rather than hard-coding "vision band"
    and mislabelling the figure when it is showing all three bands.
    """
    if arr.shape == (VISION_DEPTH_ROWS, VISION_SHEET_COLS):
        return "vision band", "vision encoder depth (deepest at top)"
    return "unified sheet", "sheet row (thinker top; vision/audio bottom)"


def _draw_band_edges(ax, arr: np.ndarray) -> None:
    """On a full-sheet view, mark where the three component bands meet."""
    if arr.shape != (SHEET_DEPTH, SHEET_COLS):
        return
    # After the quarter turn: rows 0-143 are the thinker band, 144+ vision/audio;
    # columns 0-255 are vision (and thinker), 256+ audio.
    ax.axhline(SHEET_DEPTH - VISION_DEPTH_ROWS - 0.5, color="0.35", lw=0.8, ls="--")
    ax.plot([VISION_SHEET_COLS - 0.5, VISION_SHEET_COLS - 0.5],
            [SHEET_DEPTH - VISION_DEPTH_ROWS - 0.5, SHEET_DEPTH - 0.5],
            color="0.35", lw=0.8, ls="--")


def _rgba_layer(values, opacity, cmap, vmin, vmax):
    """Colour-map ``values`` and put ``opacity`` in the alpha channel.

    Per-element alpha is why these maps use ``imshow`` on an explicit RGBA array rather
    than ``sns.heatmap``: seaborn takes only a scalar alpha and a boolean mask, so it can
    hide a unit but not fade one. Non-finite values (units with no defined preference) are
    forced fully transparent.
    """
    norm = Normalize(vmin=vmin, vmax=vmax)
    finite = np.isfinite(values)
    rgba = plt.get_cmap(cmap)(norm(np.where(finite, values, vmin)))
    rgba[..., 3] = np.clip(np.where(finite, opacity, 0.0), 0.0, 1.0)
    return rgba, norm


def plot_orientation_map(pref_ori, opacity, cmap="hsv", opacity_label="tuning strength",
                         figsize=(7, 6), savepath=None):
    """Preferred-orientation map over the vision band.

    Each unit's opacity encodes its FDR-corrected p-value, so strongly tuned units read
    solid and marginal ones fade into the grey background; units above the significance
    threshold arrive here with opacity 0 and vanish entirely.
    """
    ori_2d = vision_band(pref_ori)
    op_2d = vision_band(opacity)

    # cyclic colormap: orientation wraps at 180 deg
    rgba, norm = _rgba_layer(ori_2d, op_2d, cmap, 0.0, 180.0)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("0.92")                         # untuned units -> light-grey tissue
    ax.imshow(rgba, interpolation="nearest", aspect="equal")
    _ori_tick_cbar(fig.colorbar(
        ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap)), ax=ax, shrink=0.6))
    _draw_band_edges(ax, op_2d)
    # Count within the region actually drawn, not over the whole sheet.
    region, ylab = _band_labels(op_2d)
    n_shown = int((op_2d > 0).sum())
    ax.set_title(f"Orientation preference — {region} "
                 f"({n_shown}/{op_2d.size} tuned)")
    ax.set_xlabel(f"sheet column  ·  opacity $\\propto$ {opacity_label}")
    ax.set_ylabel(ylab)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_osi_map(osi, opacity, cmap="magma", opacity_label="tuning strength",
                 figsize=(7, 6), savepath=None):
    """Orientation-selectivity-index map over the vision band, faded by corrected p-value."""
    osi_2d = vision_band(osi)
    op_2d = vision_band(opacity)

    vmax = float(np.nanmax(osi_2d)) if np.isfinite(osi_2d).any() else 1.0
    rgba, norm = _rgba_layer(osi_2d, op_2d, cmap, 0.0, vmax)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("0.92")
    ax.imshow(rgba, interpolation="nearest", aspect="equal")
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap)),
                        ax=ax, shrink=0.6)
    cbar.set_label("orientation selectivity index")
    _draw_band_edges(ax, op_2d)
    region, ylab = _band_labels(op_2d)
    ax.set_title(f"Orientation selectivity — {region}")
    ax.set_xlabel(f"sheet column  ·  opacity $\\propto$ {opacity_label}")
    ax.set_ylabel(ylab)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_orientation_scatter(pref_ori, opacity, coords, cmap="hsv",
                             opacity_label="tuning strength",
                             figsize=(7, 6), point_size=8, savepath=None):
    """Continuous layout: preferred orientation at each unit's sheet position.

    Point opacity encodes the FDR-corrected p-value, same scale as the band maps.
    """
    coords = np.asarray(coords, dtype=float)
    xy = np.stack([coords[:, 1], coords[:, 0]], axis=1)   # (sheet col, depth row)
    opacity = np.asarray(opacity, dtype=float)
    shown = opacity > 0

    rgba, norm = _rgba_layer(pref_ori, opacity, cmap, 0.0, 180.0)

    sns.set_style("white")
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("0.92")
    ax.scatter(xy[~shown, 0], xy[~shown, 1], c="0.8", s=point_size * 0.7, linewidths=0)
    # Pass explicit RGBA per point: `alpha=` as an array is newer matplotlib only.
    ax.scatter(xy[shown, 0], xy[shown, 1], c=rgba[shown], s=point_size, linewidths=0)
    _ori_tick_cbar(fig.colorbar(
        ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap)), ax=ax, shrink=0.6))
    ax.set_aspect("equal")
    ax.set_title(f"Orientation preference — unified sheet "
                 f"({int(shown.sum())}/{shown.size} units tuned)")
    ax.set_xlabel(f"sheet column  ·  opacity $\\propto$ {opacity_label}")
    ax.set_ylabel("depth row")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_orientation_histogram(pref_ori, tuned, ori_deg, counts, n_bins=24,
                               figsize=(7, 4), savepath=None):
    """Distribution of preferred orientations over tuned units.

    Overlays the stimulus bank's own orientation distribution: this bank oversamples the
    cardinals, so the overlay is what distinguishes "the sheet prefers cardinals" from
    "the sheet inherited the bank's sampling bias".
    """
    vals = pref_ori[tuned & np.isfinite(pref_ori)]
    bins = np.linspace(0.0, 180.0, n_bins + 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(vals, bins=bins, density=True, color="0.35",
            edgecolor="white", linewidth=0.6, label="tuned units")

    # Stimulus sampling density, normalised the same way (per-degree density). Each
    # orientation's segment is *centred* on it, spanning +/- half the sampling step, and
    # drawn again shifted by +/-180 deg so the wrap-around at 0/180 is covered too (the
    # axis limits clip the overhang).
    step = 180.0 / len(ori_deg)
    stim_density = counts / counts.sum() / step
    for shift in (-180.0, 0.0, 180.0):
        for o, dens in zip(np.asarray(ori_deg, dtype=float) + shift, stim_density):
            ax.hlines(dens, o - step / 2.0, o + step / 2.0,
                      color="crimson", linewidth=1.6)
    ax.plot([], [], color="crimson", linewidth=1.6, label="stimulus bank")

    ax.axhline(1.0 / 180.0, color="0.6", linestyle=":", linewidth=1.2, label="uniform")
    ax.set_xticks(_ORI_TICKS)
    ax.set_xlim(0, 180)
    ax.set_xlabel("preferred orientation (deg)")
    ax.set_ylabel("density")
    ax.set_title(f"Preferred orientation over {vals.size} tuned vision-band units")
    ax.legend(frameon=False, fontsize=8)
    sns.despine(ax=ax)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig, ax


# -------------------------------------------------
# Feature extraction
# -------------------------------------------------

@torch.no_grad()
def extract_features_for_images(
    model,
    processor,
    image_paths: List[str],
    device: str = "cuda",
):
    """Returns cortical_sheets of shape (n_stimuli, n_units).

    One image per forward pass: ``out.unified_sheet`` is (n_tokens, 309, 512), so the mean
    over dim 0 is a mean over tokens and yields exactly one sheet per call. Batching here
    would average across images too, which is why the other run_* scripts pin batch_size=1.
    """
    model.eval()
    processor.tokenizer.padding_side = "right"

    cortical_sheets = []
    for image_path in tqdm(image_paths, desc="extracting sheets"):
        chats = [[{"role": "user", "content": [{"type": "image", "image": image_path}]}]]

        text = processor.apply_chat_template(
            chats,
            add_generation_prompt=False,
            tokenize=False,
        )

        audios, images, videos = process_mm_info(chats, use_audio_in_video=False)

        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        ).to(device)

        out = model(**inputs, return_dict=True)

        unified_sheet = out.unified_sheet.mean(dim=0)      # (309, 512), mean over tokens
        cortical_sheets.append(unified_sheet.float().numpy().reshape(-1))

    return np.stack(cortical_sheets, axis=0)


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/configs/eval/orientation.yml",
                        help="Path to YAML config for orientation-preference analysis.")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Overwrite existing outputs (plots, cached cortical sheets).")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    run_title      = str(cfg.run.run_title)
    category_name  = str(cfg.run.category_name)
    output_root    = str(cfg.run.output_root)
    do_pretrained  = bool(cfg.run.get("do_pretrained", False))
    # No pretrained weights at all (TDANN's "Unoptimized" analogue), as opposed to
    # do_pretrained, which keeps the pretrained backbone and only neutralises the adaptors.
    do_random_init = bool(cfg.run.get("do_random_init", False))

    model_override = cfg.model.get("model", None)
    device         = str(cfg.model.device)

    bank           = str(cfg.data.get("bank", "ori_bias"))
    if bank not in GRATING_BANKS:
        raise ValueError(f"data.bank must be one of {sorted(GRATING_BANKS)} (got {bank!r})")
    if cfg.data.get("stimuli_root", None):
        stimuli_root = Path(cfg.data.stimuli_root).resolve()
    else:
        stimuli_root = (Path(os.environ["STIMULI_DIR"]) / GRATING_BANKS[bank][1]).resolve()
    color          = str(cfg.data.get("color", "all"))
    download       = bool(cfg.data.get("download", True))
    workers        = int(cfg.data.get("download_workers", 8))

    alpha          = float(cfg.stats.alpha)
    baseline       = str(cfg.stats.get("baseline", "min"))
    criterion      = str(cfg.stats.get("criterion", "reliability"))
    criterion_threshold = cfg.stats.get("criterion_threshold", None)
    criterion_threshold = None if criterion_threshold is None else float(criterion_threshold)
    reliability_split = str(cfg.stats.get("reliability_split", "sf"))
    reliability_reps = int(cfg.stats.get("reliability_reps", 10))
    reliability_seed = int(cfg.stats.get("reliability_seed", 0))
    plot_cfg       = cfg.get("plot", {})
    opacity_floor  = float(plot_cfg.get("opacity_floor", 0.15))
    opacity_saturate_pct = float(plot_cfg.get("opacity_saturate_pct", 95.0))
    fwhm_mm        = float(cfg.stats.fwhm_mm)
    resolution_mm  = float(cfg.stats.resolution_mm)
    coords_path    = str(cfg.stats.coords_path)

    outdir = Path(output_root) / run_title / category_name
    outdir.mkdir(parents=True, exist_ok=True)

    if download:
        fetch_grating_bank(stimuli_root, bank=bank, overwrite=False, workers=workers)

    image_paths, ori_deg, counts, meta = parse_stimuli(stimuli_root, color=color)
    print(f"> Bank '{bank}': {len(image_paths)} gratings at {len(ori_deg)} orientations "
          f"(colour={color}) from {stimuli_root}")
    for ori, n in zip(ori_deg, counts):
        print(f"    {ori:6.1f} deg -> {n:3d} stimuli")
    if len(set(counts.tolist())) > 1:
        print("  (orientation-biased bank: averaging within orientation before the "
              "circular average keeps this sampling bias out of the estimate)")

    cache_path = outdir / "cortical_sheets.npy"

    if not os.path.exists(cache_path) or args.overwrite:
        model, processor, _ = load_topo_omni(
            model=model_override, device=device, baseline=do_pretrained,
            random_init=do_random_init,
        )
        cortical_sheets = extract_features_for_images(
            model, processor, image_paths, device=device
        )
        print(f"> Extracted cortical sheets with shape: {cortical_sheets.shape}")
        np.save(cache_path, cortical_sheets)
    else:
        print(f"> Loading cached cortical sheets from: {cache_path}")
        cortical_sheets = np.load(cache_path)
        if cortical_sheets.shape[0] != len(image_paths):
            raise ValueError(
                f"cached sheets hold {cortical_sheets.shape[0]} stimuli but the bank has "
                f"{len(image_paths)}; rerun with --overwrite"
            )

    coords = np.load(coords_path)

    smoother = NeuronSmoothingConv(fwhm_mm=fwhm_mm, resolution_mm=resolution_mm)
    smoothed_sheet = smoother(coords, cortical_sheets)
    print(f"> Smoothed sheet shape: {smoothed_sheet.shape} (fwhm={fwhm_mm} mm)")

    curve, groups = orientation_tuning_curve(smoothed_sheet, counts)
    pref_ori, osi = orientation_preference(curve, ori_deg, baseline=baseline)
    tuned, qvals = anova_orientation(groups, alpha=alpha)

    # All three criteria are computed every run and saved, so the figures can be re-cut on a
    # different one without re-extracting or even re-smoothing.
    eta2, omega2 = eta_squared(smoothed_sheet, counts)
    if reliability_split == "random":
        split_key = None
    elif reliability_split in meta:
        split_key = meta[reliability_split]
    else:
        raise ValueError(
            f"stats.reliability_split must be 'random' or one of {sorted(meta)} "
            f"(got {reliability_split!r})")
    reliability = split_half_reliability(
        smoothed_sheet, counts, split_key=split_key,
        n_rep=reliability_reps, seed=reliability_seed)
    print(f"> Reliability split: {reliability_split}"
          + ("" if split_key is None else
             f" (halves defined by {reliability_split}, so this tests generalisation "
             f"across it)"))

    strength, cutoff, opacity_label = tuning_strength(
        criterion, alpha, criterion_threshold,
        qvals=qvals, eta2=eta2, reliability=reliability,
    )
    gate = strength > cutoff

    # Report on the vision band and the rest separately: images leave the audio band
    # (depth rows 0-159, sheet columns 256-511) undriven, so whole-sheet medians are
    # dominated by units that saw no input.
    vis = vision_mask()
    print(f"> Criterion '{criterion}' (cutoff {cutoff:.4g}); FDR alpha {alpha:g} shown "
          f"alongside for reference")
    for name, sel in (("vision band", vis), ("rest of sheet", ~vis)):
        sel_gate = gate & sel
        osi_sel = osi[sel_gate]
        print(f"> {name}: {int(sel_gate.sum())}/{int(sel.sum())} units pass "
              f"({100.0 * sel_gate.sum() / sel.sum():.1f}%), "
              f"median OSI {np.median(osi_sel) if osi_sel.size else float('nan'):.3f}")
        print(f"    median eta^2 {np.median(eta2[sel]):.3f} | "
              f"median reliability {np.median(reliability[sel]):.3f} | "
              f"FDR q<alpha {100.0 * (tuned & sel).sum() / sel.sum():.1f}%")

    # Per-unit opacity from the chosen criterion. The vision band sets the saturation point
    # so the band maps use their own dynamic range; the scatter reuses the same scale, which
    # keeps the figures directly comparable.
    opacity = criterion_opacity(
        strength, cutoff, floor=opacity_floor,
        saturate_pct=opacity_saturate_pct, sel=vis,
    )
    shown = opacity > 0
    print(f"> Opacity: {int(shown.sum())} units drawn, "
          f"median {np.median(opacity[shown]) if shown.any() else float('nan'):.2f}, "
          f"{100.0 * np.mean(opacity[shown] >= 0.999) if shown.any() else 0.0:.1f}% fully opaque "
          f"(floor {opacity_floor}, saturating at the {opacity_saturate_pct:g}th pct of "
          f"the criterion in the vision band)")

    plot_orientation_map(
        pref_ori, opacity, opacity_label=opacity_label,
        savepath=outdir / "orientation_preference_sheet.png",
    )
    plot_osi_map(
        osi, opacity, opacity_label=opacity_label,
        savepath=outdir / "orientation_selectivity_sheet.png",
    )
    plot_orientation_scatter(
        pref_ori, opacity, coords, opacity_label=opacity_label,
        savepath=outdir / "orientation_preference_scatter.png",
    )
    plot_orientation_histogram(
        pref_ori, gate & vis, ori_deg, counts,
        savepath=outdir / "orientation_preference_histogram.png",
    )

    # Keep the alternative baseline alongside, so the choice can be audited without a rerun.
    other = "zero" if baseline == "min" else "min"
    pref_alt, osi_alt = orientation_preference(curve, ori_deg, baseline=other)

    np.savez(
        outdir / "orientation_stats.npz",
        pref_ori=pref_ori, osi=osi, tuned=tuned, baseline=baseline,
        qvals=qvals, opacity=opacity, alpha=alpha,
        eta2=eta2, omega2=omega2, reliability=reliability,
        criterion=criterion, criterion_cutoff=cutoff, gate=gate,
        reliability_split=reliability_split,
        pref_ori_alt=pref_alt, osi_alt=osi_alt, baseline_alt=other,
        vision_band=vis, ori_deg=ori_deg, counts=counts, tuning_curve=curve,
    )

    print(f"> Saved orientation maps + stats to: {outdir}")


if __name__ == "__main__":
    main()
