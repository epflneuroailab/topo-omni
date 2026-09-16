"""
Run TDANN's own trained model on the same gratings, so its
Delta-orientation-vs-distance curve can be plotted next to ours and macaque's.

Why this exists
---------------
TDANN publishes checkpoints and unit positions, but the ``features`` and ``results`` folders
on their OSF node are empty, so there is no precomputed curve to load. The alternative to
running the model would be digitising their plotted line, which is guesswork. This module
instead reproduces the model side directly.

Which model is "TDANN"
----------------------
From ``notebooks/F02_V1_Details``: ``"TDANN": tissues["SW_0pt25"][4]`` -- the spatial-weight
0.25 model, seed 4. Its analysis config
(``configs/analysis_configs/simclr/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3.yaml``)
gives the checkpoint directory and the shared position directory
(``simclr_spatial_resnet18_fuzzy_swappedon_SineGrating2019_lw0/...NBVER2``), and their
``hypercolumn_widths`` gives Lambda = 3.5 mm for this model.

Analysis choices copied from their notebook
-------------------------------------------
* layer ``layer2.0`` -- the V1-analogue layer used throughout their Fig. 2.
* ``set_unit_mask_by_ptp_percentile(metric, 75)`` -- only the top 25% of units by
  peak-to-peak response across orientations enter the curve. Skipping this would change the
  curve, so it is applied here too.
* TDANN's distance binning and Lambda normalisation, via ``macaque_map.tdann_bin_edges``.

``bank`` selects which OSF grating set drives the model (see ``run_orientation.GRATING_BANKS``),
and each bank gets its own cache, because the curve depends on the stimuli. Use the same bank
Topo-Omni was driven with so the curves are comparable. ``unbiased``
(``sine_grating_images_20190507``) is the set TDANN's published figures use, so it is the one
whose smoothness should be checked against their published ~0.72.

The checkpoint (92 MB) and positions (13.5 MB) are downloaded on demand into
``data/tdann/`` and are not committed.
"""

import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from src.experiments.orientation.macaque_map import tdann_bin_edges, delta_over_distance

from dotenv import load_dotenv
load_dotenv()

# OSF node 64qv3, resolved from the folder listing (see module docstring for provenance)
TDANN_CKPT_URL = "https://osf.io/download/dtnjw/"       # ..._isoswap_3_seed_4, phase199
TDANN_POS_URL = "https://osf.io/download/279sy/"        # layer2.0.npz positions

TDANN_LAYER = "layer2.0"
TDANN_COLUMN_SPACING_MM = 3.5      # hypercolumn_widths["SW_0pt25"]
TDANN_PTP_PERCENTILE = 75.0        # set_unit_mask_by_ptp_percentile(..., 75)
TDANN_PUBLISHED_SMOOTHNESS = 0.72  # read off Fig. 2E, for validation only

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _fetch(url: str, dest: Path, retries: int = 6) -> Path:
    """Download to ``dest`` unless already present, retrying OSF's 429 rate limit."""
    import random
    import time

    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "*/*"})
            with urllib.request.urlopen(req, timeout=900) as resp, tmp.open("wb") as fh:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
            tmp.replace(dest)
            return dest
        except urllib.error.HTTPError as err:
            last = err
            if err.code not in (429, 500, 502, 503, 504):
                raise
            wait = float(err.headers.get("Retry-After") or 0) or min(2 ** attempt, 60)
            time.sleep(wait + random.uniform(0, 1.0))
        except (urllib.error.URLError, TimeoutError) as err:
            last = err
            time.sleep(min(2 ** attempt, 60) + random.uniform(0, 1.0))
        finally:
            tmp.unlink(missing_ok=True)
    raise RuntimeError(f"could not download {url}: {last}")


def load_tdann_resnet18(ckpt_path: Path, device: str = "cuda"):
    """Load TDANN's VISSL checkpoint into a plain torchvision resnet18.

    The trunk state dict carries a ``base_model.``-prefixed copy using torchvision's own
    parameter names, so stripping that prefix is enough -- no VISSL install needed. Any
    leftover ``_feature_blocks.`` keys are VISSL's alternative view of the same weights and
    are dropped.
    """
    from torchvision.models import resnet18

    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    trunk = ck["classy_state_dict"]["base_model"]["model"]["trunk"]
    state = {k[len("base_model."):]: v for k, v in trunk.items()
             if k.startswith("base_model.")}
    if not state:
        raise RuntimeError("no 'base_model.'-prefixed weights in this checkpoint")

    model = resnet18()
    missing, unexpected = model.load_state_dict(state, strict=False)
    # fc is unused (SimCLR trunk has no classifier); anything else missing is a real problem.
    real_missing = [k for k in missing if not k.startswith("fc.")]
    if real_missing:
        raise RuntimeError(f"checkpoint did not populate {len(real_missing)} params, "
                           f"e.g. {real_missing[:5]}")
    return model.eval().to(device), {"missing": missing, "unexpected": unexpected}


@torch.no_grad()
def tdann_layer2_responses(model, image_paths: List[str], device: str = "cuda",
                           batch_size: int = 32, size: int = 224) -> np.ndarray:
    """Responses at ``layer2.0`` for each image, flattened to match the position file.

    ``dims`` in the position file is ``[128, 28, 28]``, i.e. C-order over
    (channel, row, col), which is exactly ``tensor.reshape(batch, -1)`` on the layer output.
    """
    from PIL import Image

    feats = {}
    handle = model.layer2[0].register_forward_hook(
        lambda _m, _i, out: feats.__setitem__("x", out))

    chunks = []
    try:
        for i in range(0, len(image_paths), batch_size):
            batch = []
            for path in image_paths[i:i + batch_size]:
                img = Image.open(path).convert("RGB").resize((size, size),
                                                             Image.BILINEAR)
                arr = (np.asarray(img, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
                batch.append(arr.transpose(2, 0, 1))
            x = torch.from_numpy(np.stack(batch)).to(device)
            model(x)
            chunks.append(feats["x"].reshape(x.shape[0], -1).float().cpu().numpy())
    finally:
        handle.remove()

    return np.concatenate(chunks, axis=0)


def tdann_delta_curve(
    cache: Path = None,
    data_dir: Path = Path("data/tdann"),
    bank: str = "ori_bias",
    stimuli_root: Path = None,
    device: str = "cuda",
    n_bins: int = 9,
    num_samples: int = 20,
    sample_size: int = 1000,
    baseline: str = "min",
    overwrite: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """TDANN's Delta-orientation-vs-distance curve, x in units of its column spacing.

    Cached as derived numbers, so the checkpoint is needed only on the first run.

    Returns:
        x      : (n_bins,) bin midpoints as a fraction of TDANN's column spacing
        curves : (num_samples, n_bins)
    """
    # One cache per stimulus bank: the curve depends on which gratings drove the model.
    if cache is None:
        cache = Path(f"data/tdann_delta_curve_{bank}.npz")
    cache = Path(cache)
    if cache.exists() and not overwrite:
        z = np.load(cache)
        return z["x"], z["curves"]

    import os
    from src.experiments.orientation.run import parse_stimuli, orientation_tuning_curve, \
        orientation_preference

    if stimuli_root is None:
        from src.experiments.orientation.run import GRATING_BANKS
        stimuli_root = Path(os.environ["STIMULI_DIR"]) / GRATING_BANKS[bank][1]
    paths, ori_deg, counts, _ = parse_stimuli(Path(stimuli_root), color="all")

    ckpt = _fetch(TDANN_CKPT_URL, Path(data_dir) / "tdann_seed4.torch")
    pos_file = _fetch(TDANN_POS_URL, Path(data_dir) / "layer2.0.npz")

    model, info = load_tdann_resnet18(ckpt, device=device)
    resp = tdann_layer2_responses(model, paths, device=device)

    z = np.load(pos_file, allow_pickle=True)
    positions = z["coordinates"].astype(float)
    if positions.shape[0] != resp.shape[1]:
        raise RuntimeError(f"{positions.shape[0]} positions vs {resp.shape[1]} units; "
                           f"layer or dims mismatch (dims={z['dims']})")

    curve, _ = orientation_tuning_curve(resp, counts)
    pref, _ = orientation_preference(curve, ori_deg, baseline=baseline)

    # Their unit mask: keep only the most responsive quarter of units.
    ptp = curve.max(axis=0) - curve.min(axis=0)
    keep = ptp >= np.percentile(ptp, TDANN_PTP_PERCENTILE)
    keep &= np.isfinite(pref)

    edges = tdann_bin_edges(TDANN_COLUMN_SPACING_MM, n_bins=n_bins)
    curves = delta_over_distance(pref[keep], positions[keep], edges,
                                 num_samples=num_samples, sample_size=sample_size)
    x = (0.5 * (edges[:-1] + edges[1:])) / TDANN_COLUMN_SPACING_MM

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, x=x, curves=curves, n_units=int(keep.sum()), bank=bank,
             n_stimuli=len(paths),
             column_spacing_mm=TDANN_COLUMN_SPACING_MM, layer=TDANN_LAYER)
    return x, curves
