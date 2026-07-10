"""
Unified topographic selectivity test for TopoQwen2VL.

Supports two modes, specified in a YAML config:

  1) TEXT MODE ("text"):
       stimuli_root/
         ON/   <some_text_file.txt>      # one stimulus per line
         OFF/  <some_text_file.txt>

     Computes selectivity on the LM cortical sheet only (words vs non-words type).

  2) IMAGE MODE ("image"):
       stimuli_root/
         ON/       (images: jpg/png/jpeg/webp/bmp...)
         OFF/      (images: jpg/png/jpeg/webp/bmp...)

     Computes selectivity on both LM and VIS cortical sheets (faces vs non-faces type).

The model is loaded via ``src.core.model_loading.load_topo_omni`` (HuggingFace
``epfl-neuroai/topo-omni`` by default; override with ``model.model`` in the config or
``$TOPO_OMNI_MODEL``). The script expects a YAML config file with keys:
      run:   {run_title, category_name, output_root, do_pretrained, overwrite}
      model: {device, model (optional)}
      data:  {mode, stimuli_root, batch_size, lm_reduce, vis_reduce, odd_or_even}
      stats: {alpha, smooth, fwhm_mm, resolution_mm, topk_pct}
"""

import json
import argparse
from pathlib import Path
from typing import List, Tuple

import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
from tqdm import tqdm

from qwen_omni_utils import process_mm_info
from src.core.model_loading import load_topo_omni, unified_grid_coords
from src.utils.island_morans_I import island_morans_I
from src.utils.smoothing import NeuronSmoothingConv


def dump_pickle(obj, fpath: Path):
    with fpath.open("wb") as f:
        pickle.dump(obj, f)

def load_pickle(fpath: Path):
    with fpath.open("rb") as f:
        obj = pickle.load(f)
    return obj  

def write_json(obj, fpath: Path):
    with fpath.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)

_HAVE_SMOOTH = True

# -------------------------------------------------
# Small helpers (shared)
# -------------------------------------------------

def pil_open(path: str):
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

def cohen_d(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    mean1, mean2 = x1.mean(axis=0), x2.mean(axis=0)
    std1, std2 = x1.std(axis=0, ddof=1), x2.std(axis=0, ddof=1)
    n1, n2 = x1.shape[0], x2.shape[0]
    pooled = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) /
                     max(n1 + n2 - 2, 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (mean1 - mean2) / pooled
        d[~np.isfinite(d)] = 0.0
    return d

def _bh_fallback(p: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg q-values if SciPy FDR not available."""
    p = np.asarray(p, dtype=float)
    n = p.size
    order = np.argsort(p)
    q = np.empty_like(p)
    cummin = 1.0
    for rank_from_end, idx in enumerate(order[::-1], start=1):
        rank = n - rank_from_end + 1
        val = p[idx] * n / rank
        cummin = min(cummin, val)
        q[idx] = cummin
    return np.minimum(q, 1.0)

def fdr_qvalues(p: np.ndarray) -> np.ndarray:
    try:
        return stats.false_discovery_control(p)
    except Exception:
        return _bh_fallback(p)

def attention_last_indices(attn_mask: torch.Tensor) -> torch.Tensor:
    lengths = attn_mask.sum(dim=1)
    return torch.clamp(lengths - 1, min=0).long()

def infer_grid_from_coords(coords: np.array) -> Tuple[int, int]:
    xs = coords[:, 0].astype(np.int64)
    ys = coords[:, 1].astype(np.int64)
    H = int(ys.max()) + 1
    W = int(xs.max()) + 1
    return H, W


def place_on_grid(values: np.ndarray, coords: np.array) -> np.ndarray:
    H, W = infer_grid_from_coords(coords)
    grid = np.full((H, W), np.nan, dtype=np.float32)
    ij = coords.astype(np.int64)
    grid[ij[:, 1], ij[:, 0]] = values
    return grid


def ensure_device_dtype(batch: dict, device: torch.device):
    for k, v in batch.items():
        if not isinstance(v, torch.Tensor):
            continue
        v = v.to(device)
        if device.type == "cuda" and v.is_floating_point():
            v = v.to(torch.bfloat16)
        batch[k] = v
    return batch

# -------------------------------------------------
# Data loading (ON/OFF)
# -------------------------------------------------

def load_texts(stimuli_root: Path) -> Tuple[List[str], List[str]]:
    """
    Expect:
      stimuli_root/
        ON/  <some_text_file.txt>
        OFF/ <some_text_file.txt>

    Each file: one stimulus per line (non-empty lines only).
    """
    on_dir = stimuli_root / "ON"
    off_dir = stimuli_root / "OFF"

    if not on_dir.exists() or not off_dir.exists():
        raise FileNotFoundError(f"Expecting 'ON' and 'OFF' directories in {stimuli_root}")

    def _read_folder(folder: Path) -> List[str]:
        txt_files = [p for p in folder.iterdir() if p.is_file()]
        if not txt_files:
            raise FileNotFoundError(f"No text files found in {folder}")
        txt_files = sorted(txt_files)
        lines = []
        for fpath in txt_files:
            if "theory_of_mind" in stimuli_root.as_posix() or "multiple_demand" in stimuli_root.as_posix() or "language" in stimuli_root.as_posix():
                with fpath.open("r", encoding="utf-8") as f:
                    stimuli_lines = f.read().split("\n\n")
                    lines.extend([line.strip() for line in stimuli_lines])
            else:
                with fpath.open("r", encoding="utf-8") as f:
                    for line in f:
                        s = line.strip()
                        if s:
                            lines.append(s)
        return lines

    on_texts = _read_folder(on_dir)
    off_texts = _read_folder(off_dir)
    return on_texts, off_texts

def load_images(stimuli_root: Path) -> Tuple[List[str], List[str]]:
    """
    Expect:
      stimuli_root/
        ON/  (images)
        OFF/ (images)
    """
    on_dir = stimuli_root / "ON"
    off_dir = stimuli_root / "OFF"
    if not on_dir.exists() or not off_dir.exists():
        raise FileNotFoundError(
            f"Expecting 'ON' and 'OFF' sub-directories under {stimuli_root}"
        )

    def _collect(dirpath: Path) -> List[str]:
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        files = []
        for p in sorted(dirpath.iterdir()):
            if p.is_file() and p.suffix.lower() in exts:
                files.append(p.as_posix())
        if not files:
            raise FileNotFoundError(f"No images found in {dirpath}")
        return files

    on = _collect(on_dir)
    off = _collect(off_dir)
    return on, off

def load_audio(stimuli_root: Path) -> Tuple[List[str], List[str]]:
    """
    Expect:
      stimuli_root/
        ON/  (audio)
        OFF/ (audio)
    """
    on_dir = stimuli_root / "ON"
    off_dir = stimuli_root / "OFF"
    if not on_dir.exists() or not off_dir.exists():
        raise FileNotFoundError(
            f"Expecting 'ON' and 'OFF' sub-directories under {stimuli_root}"
        )

    def _collect(dirpath: Path) -> List[str]:
        exts = {".wav"}
        files = []
        for p in sorted(dirpath.iterdir()):
            if p.is_file() and p.suffix.lower() in exts:
                files.append(p.as_posix())
        if not files:
            raise FileNotFoundError(f"No audio found in {dirpath}")
        return files

    on = _collect(on_dir)
    off = _collect(off_dir)
    return on, off

def load_video(stimuli_root: Path, odd_or_even=None) -> Tuple[List[str], List[str]]:
    """
    Expect:
      stimuli_root/
        ON/  (video)
        OFF/ (video)
    """
    on_dir = stimuli_root / "ON"
    off_dir = stimuli_root / "OFF"
    if not on_dir.exists() or not off_dir.exists():
        raise FileNotFoundError(
            f"Expecting 'ON' and 'OFF' sub-directories under {stimuli_root}"
        )

    def _collect(dirpath: Path) -> List[str]:
        exts = {".mp4", ".avi", ".mov", ".mkv"}
        files = []
        for p in sorted(dirpath.iterdir()):
            if p.is_file() and p.suffix.lower() in exts:
                file_path = p.as_posix()
                basename = os.path.basename(file_path)
                run_num = int(''.join(filter(str.isdigit, basename.split("_")[0])))
                if odd_or_even == "odd" and run_num % 2 == 0:
                    continue
                if odd_or_even == "even" and run_num % 2 == 1:
                    continue
                files.append(file_path)
        if not files:
            raise FileNotFoundError(f"No video found in {dirpath}")
        return files

    on = _collect(on_dir)
    off = _collect(off_dir)
    return on, off

# -------------------------------------------------
# Feature extraction
# -------------------------------------------------

@torch.no_grad()
def extract_features_for_texts(
    model,
    processor,
    texts: List[str],
    batch_size: int = 16,
    device: str = "cuda",
    prompt: str|None = None,
):
    """
    Returns: feats_lm (N, D). Only LM sheet is computed for text-only stimuli.
    """
    model.eval()
    dev = torch.device(device)
    processor.tokenizer.padding_side = "right"

    cortical_sheets = []

    for i in tqdm(range(0, len(texts), batch_size)):
        chunk = texts[i:i + batch_size]

        chats = [[{"role": "user", "content": [{"type": "text", "text": t}]}] for t in chunk]
        rendered = [processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=False) for chat in chats]

        batch = processor(text=rendered, images=None, return_tensors="pt", padding=True)
        batch = ensure_device_dtype(batch, dev)

        out = model(**batch, return_dict=True)

        unified_sheet = out.unified_sheet

        unified_sheet = unified_sheet.mean(dim=0)
        unified_sheet = unified_sheet.float().numpy()
        cortical_sheets.append(unified_sheet)
        
    cortical_sheets = np.stack(cortical_sheets, axis=0).reshape(len(cortical_sheets), -1)
    return cortical_sheets


@torch.no_grad()
def extract_features_for_images(
    model,
    processor,
    image_paths: List[str],
    batch_size: int = 8,
    device: str = "cuda",
    prompt: str|None = None,
):
    """
    Returns: feats_lm (N, D_lm) or None, feats_vis (N, D_vis) or None.
    Feeds images with NO accompanying text. Uses the chat template with <image> only.
    """
    model.eval()
    dev = torch.device(device)
    processor.tokenizer.padding_side = "right"

    cortical_sheets = []

    for i in tqdm(range(0, len(image_paths), batch_size)):
        chunk_paths = image_paths[i:i + batch_size]
        images = [pil_open(p) for p in chunk_paths]

        chats = [[{"role": "user", "content": [{"type": "image", "image": img}]}] for img in images]

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
        )

        inputs = ensure_device_dtype(inputs, dev)

        out = model(**inputs, return_dict=True)

        unified_sheet = out.unified_sheet

        unified_sheet = unified_sheet.mean(dim=0)
        unified_sheet = unified_sheet.float().numpy()
        cortical_sheets.append(unified_sheet)

    cortical_sheets = np.stack(cortical_sheets, axis=0).reshape(len(cortical_sheets), -1)
    return cortical_sheets


@torch.no_grad()
def extract_features_for_audio(
    model,
    processor,
    audio_paths: List[str],
    batch_size: int = 8,
    device: str = "cuda",
    prompt: str|None = None,
):
    """
    Returns: feats_lm (N, D_lm) or None, feats_vis (N, D_vis) or None.
    Feeds audio with NO accompanying text. Uses the chat template with <audio> only.
    """
    model.eval()
    dev = torch.device(device)
    processor.tokenizer.padding_side = "right"

    cortical_sheets = []

    for i in tqdm(range(0, len(audio_paths), batch_size)):
        chunk_paths = audio_paths[i:i + batch_size]

        chats = [[{"role": "user", "content": [{"type": "audio", "audio": audio}]}] for audio in chunk_paths]

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
        )

        inputs = ensure_device_dtype(inputs, dev)

        out = model(**inputs, return_dict=True)

        unified_sheet = out.unified_sheet

        unified_sheet = unified_sheet.mean(dim=0)
        unified_sheet = unified_sheet.float().numpy()
        cortical_sheets.append(unified_sheet)

    cortical_sheets = np.stack(cortical_sheets, axis=0).reshape(len(cortical_sheets), -1)
    return cortical_sheets


@torch.no_grad()
def extract_features_for_video(
    model,
    processor,
    video_paths: List[str],
    batch_size: int = 8,
    device: str = "cuda",
    use_audio_in_video: bool = True,
    prompt: str|None = None,
):
    """
    Returns: feats_lm (N, D_lm) or None, feats_vis (N, D_vis) or None.
    Feeds audio with NO accompanying text. Uses the chat template with <audio> only.
    """
    model.eval()
    dev = torch.device(device)
    processor.tokenizer.padding_side = "right"

    cortical_sheets = []

    for i in tqdm(range(0, len(video_paths), batch_size)):
        chunk_paths = video_paths[i:i + batch_size]

        if prompt is not None:
            chats = [[
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
                {"role": "user", "content": [{"type": "video", "video": video}]},
            ] for video in chunk_paths]
        else:
            chats = [[{"role": "user", "content": [{"type": "video", "video": video}]}] for video in chunk_paths]

        text = processor.apply_chat_template(
            chats, 
            add_generation_prompt=False, 
            tokenize=False,
        )

        audios, images, videos = process_mm_info(chats, use_audio_in_video=use_audio_in_video)

        inputs = processor(
            text=text, 
            audio=audios, 
            images=images, 
            videos=videos, 
            return_tensors="pt", 
            padding=True, 
            use_audio_in_video=use_audio_in_video, 
        )

        inputs = ensure_device_dtype(inputs, dev)

        out = model(**inputs, return_dict=True)

        unified_sheet = out.unified_sheet

        unified_sheet = unified_sheet.mean(dim=0)
        unified_sheet = unified_sheet.float().numpy()
        cortical_sheets.append(unified_sheet)

    cortical_sheets = np.stack(cortical_sheets, axis=0).reshape(len(cortical_sheets), -1)
    return cortical_sheets

# -------------------------------------------------
# Core stats logic
# -------------------------------------------------

def run_stats(pos: np.ndarray, neg: np.ndarray):
    tvals, pvals = stats.ttest_ind(pos, neg, axis=0, equal_var=False, alternative='greater')
    qvals = fdr_qvalues(pvals.flatten())
    dvals = cohen_d(pos, neg)
    return dict(
        t=tvals, 
        p=pvals, 
        q=qvals, 
        d=dvals, 
    )


def maybe_hrf_project(feats: np.ndarray,
                      coords: np.array,
                      smoother: NeuronSmoothingConv | None):
    """
    If smoothing is enabled, project per-neuron activations to a regular grid
    using the HRF Gaussian kernel. Returns (H, W, projected_feats).
    Otherwise returns (None, None, feats).
    """
    if (smoother is None) or (coords is None) or (feats is None):
        return None, None, feats
    xy = coords.astype(int)
    # positions = torch.from_numpy(xy).float().to(smoother.device)      # (N_neurons, 2)
    # activations = torch.from_numpy(feats).float().to(smoother.device)  # (N_samples, N_neurons)
    projected = smoother(xy, feats)   # projected: (N_samples, N_grid)
    # H, W = _hrf_grid_shape(gridx, gridy)
    H = smoother.height
    W = smoother.width
    return H, W, projected


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/configs/eval_marvi.yml",
                        help="Path to YAML config for selectivity analysis.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    run_title     = str(cfg.run.run_title)
    category_name = str(cfg.run.category_name)
    output_root   = str(cfg.run.output_root)
    do_pretrained = bool(cfg.run.get("do_pretrained", False))
    overwrite     = bool(cfg.run.get("overwrite", False))

    # HF repo id or local checkpoint dir; defaults to $TOPO_OMNI_MODEL / epfl-neuroai/topo-omni.
    model_override = cfg.model.get("model", None)
    device        = str(cfg.model.device)

    mode          = str(cfg.data.mode).lower()      # "text" or "image"
    stimuli_root  = Path(cfg.data.stimuli_root).resolve()
    batch_size    = int(cfg.data.batch_size)
    lm_reduce     = str(cfg.data.lm_reduce)
    vis_reduce    = str(cfg.data.vis_reduce) if mode == "image" and "vis_reduce" in cfg.data else "mean"
    odd_or_even   = cfg.data.get("odd_or_even", None)

    alpha         = float(cfg.stats.alpha)
    smooth        = bool(cfg.stats.smooth)
    fwhm_mm       = float(cfg.stats.fwhm_mm)
    resolution_mm = float(cfg.stats.resolution_mm)
    topk_pct      = float(cfg.stats.get("topk_pct", 0.0))

    outdir = Path(output_root) / run_title / category_name
    outdir.mkdir(parents=True, exist_ok=True)

    cache_dir = outdir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if odd_or_even is not None:
        category_name += f"_{odd_or_even}"

    dump_pickle_path = cache_dir / f"selectivity_{category_name}_features.pkl"

    # The model is only needed to (re)extract features; on a cache hit we plot from the
    # cached cortical sheets and never touch the GPU / HuggingFace.
    model = processor = None
    if not os.path.exists(dump_pickle_path) or overwrite:
        model, processor, _ = load_topo_omni(
            model=model_override, device=device, baseline=do_pretrained
        )

    if mode == "text":
        print("> Mode: TEXT")
        on_items, off_items = load_texts(stimuli_root)
    elif mode == "image":
        print("> Mode: IMAGE")
        on_items, off_items = load_images(stimuli_root)
    elif mode == "audio":
        print("> Mode: AUDIO")
        on_items, off_items = load_audio(stimuli_root)
    elif mode == "video":
        print("> Mode: VIDEO")
        on_items, off_items = load_video(stimuli_root, odd_or_even=odd_or_even)
    else:
        raise ValueError(f"Unknown mode: {mode}. Expected 'text' or 'image'.")

    print(f"ON  ({category_name})        : {len(on_items)}")
    print(f"OFF (non-{category_name})    : {len(off_items)}")

    smoother = None
    if smooth and _HAVE_SMOOTH:
        print(f"> Applying spatial smoothing … (FWHM={fwhm_mm}mm, res={resolution_mm}mm)")
        smoother = NeuronSmoothingConv(fwhm_mm=fwhm_mm, resolution_mm=resolution_mm)

    sns.set_context("notebook")    
    results = {}
    summary = {
        "mode": mode,
        "category": category_name,
        "n_on": len(on_items),
        "n_off": len(off_items),
        "alpha": alpha,
        "smooth": bool(smoother is not None),
        "lm_reduce": lm_reduce,
    }
    if mode == "image":
        summary["vis_reduce"] = vis_reduce

    extract_features = extract_features_for_texts if mode == "text" else (
        extract_features_for_images if mode == "image" else
        extract_features_for_audio if mode == "audio" else
        extract_features_for_video
    )

    if dump_pickle_path.exists() and not overwrite:
        print(f"> Loading cached features from {dump_pickle_path}")
        cached = load_pickle(dump_pickle_path)
        on_sheet, off_sheet = cached["on_sheet"], cached["off_sheet"]
        print(f"✓ loaded cached features from {dump_pickle_path}")
    else:
        print(f"> Extracting features and caching to {dump_pickle_path} …")
        if "marvi_videos_theory_of_mind" in cfg.data.stimuli_root:
            prompt_on  = "Question: Is the second statement consistent with the first?"
            prompt_off = "Question: Is the second statement consistent with the first?"
        elif "marvi_videos_multi_demand" in cfg.data.stimuli_root:
            prompt_on  = "Question: Is the given solution to the equation correct?"
            prompt_off = "Question: Is the second statement consistent with the first?"
        elif "marvi_videos_language" in cfg.data.stimuli_root:
            prompt_on  = "Question: Is the gender of the second speaker the same as the first speaker?"
            prompt_off = "Question: Is the gender of the second speaker the same as the first speaker?"
        else:
            prompt_on = None
            prompt_off = None

        
        print(f"> Using prompt for video stimuli: {prompt_on}")

        on_sheet = extract_features(
            model, processor, on_items,
            batch_size=batch_size, device=device, prompt=prompt_on
        )
        off_sheet = extract_features(
            model, processor, off_items,
            batch_size=batch_size, device=device, prompt=prompt_off
        )
        dump_pickle(
            {"on_sheet": on_sheet, "off_sheet": off_sheet},
            dump_pickle_path
        )
        print(f"✓ cached features to {dump_pickle_path}")

    # Unit coordinates are the fixed 304x512 grid (row-major), so we derive them
    # deterministically instead of reading a training-time coords.npy artifact.
    coords_lm = unified_grid_coords()

    category_name += f"_fwhm{fwhm_mm:.1f}" if smoother is not None else ""

    H_lm = None
    W_lm = None
    if (on_sheet is not None) and (coords_lm is not None):
        print(f"> Projecting LM decoder activations to grid")
        H_lm, W_lm, on_sheet  = maybe_hrf_project(on_sheet,  coords_lm, smoother)
        _,    _,   off_sheet = maybe_hrf_project(off_sheet, coords_lm, smoother)

    if (on_sheet is not None) and (off_sheet is not None) and (coords_lm is not None):
        print(f"> Computing LM decoder selectivity stats")
        stats_lm = run_stats(on_sheet, off_sheet)
        results[f"model"] = stats_lm

        if H_lm is not None and W_lm is not None:
            grid_t = stats_lm["t"].reshape(H_lm, W_lm)
            grid_q = stats_lm["q"].reshape(H_lm, W_lm)
        else:
            print("> Placing on Grid")
            grid_t = place_on_grid(stats_lm["t"], coords_lm)
            grid_q = place_on_grid(stats_lm["q"], coords_lm)

        print(f"> Computing Island Moran's I for LM decoder selectivity")
        island_morans_I_result = island_morans_I(p_map=grid_q, t_map=grid_t, p_threshold=alpha)

        results[f"model"][f"island_morans_I"] = {
            "I": island_morans_I_result["average_moran_I"],
            "num_components": island_morans_I_result["num_components"],
            "num_significant_components": island_morans_I_result["num_significant_components"]
        }

        write_json(
            island_morans_I_result["island_moran_values"],
            outdir / f"{category_name}_island_morans_I.json"
        )

        mask = stats_lm["t"].copy()
        mask[(stats_lm["q"] > alpha) | (stats_lm["t"] <= 0)] = np.nan

        if H_lm is not None and W_lm is not None:
            grid_lm_mask = mask.reshape(H_lm, W_lm)
        else:
            print("> Placing on Grid")
            grid_lm_mask = place_on_grid(mask, coords_lm)

    fig, ax = plt.subplots()

    # rotate grid_lm_mask 90 degrees to the left
    grid_lm_mask = np.rot90(grid_lm_mask, k=1)

    sns.heatmap(grid_lm_mask, cbar=True, cmap="viridis", ax=ax,
                linewidths=0, linecolor=None)

    # add horizontal dashed line at y = 160
    ax.axhline(y=304 - 160, color='gray', linestyle='--')

    # add vertical dashed line at x = 256 until y = 160
    ax.vlines(x=256, ymin=304 - 160, ymax=304, color='gray', linestyle='--')

    # remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    dump_pickle(
        stats_lm,
        outdir / f"{category_name}_selectivity_stats.pkl"
    )
    
    morans_I = island_morans_I_result["average_moran_I"]
    title_lm = f"{category_name} Selectivity | Island Moran's I: {morans_I:.4f}"
    
    ax.set_title(title_lm, pad=18)
    fig.tight_layout()

    if topk_pct <= 0:
        out_svg = outdir / f"{category_name}_selectivity_alpha{alpha:.1e}_fwhm={fwhm_mm:.1f}.svg"
        out_png = outdir / f"{category_name}_selectivity_alpha{alpha:.1e}_fwhm={fwhm_mm:.1f}.png"
    else:
        out_svg = outdir / f"{category_name}_selectivity_top{int(topk_pct)}_fwhm={fwhm_mm:.1f}.svg"
        out_png = outdir / f"{category_name}_selectivity_top{int(topk_pct)}_fwhm={fwhm_mm:.1f}.png"

    fig.savefig(out_svg.as_posix(), format="svg", bbox_inches="tight")
    fig.savefig(out_png.as_posix(), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ saved {out_svg}")
    print(f"✓ saved {out_png}")

    # counts
    m = stats_lm["t"].copy()

    if topk_pct <= 0.0:
        m[(stats_lm["q"] > alpha) | (stats_lm["t"] <= 0)] = np.nan
    else:
        m[(~stats_lm[f"top_{int(topk_pct)}_pct"])] = np.nan

    n_act = int(np.isfinite(m).sum())
    n_tot = int(m.size)
    summary[f"island_morans_I"] = results[f"model"][f"island_morans_I"]
    summary[f"alpha_counts"] = {f"{alpha:.2f}": {"active": n_act, "total": n_tot}}

    # ---- Write summary JSON ----
    summary_path = outdir / f"{category_name}_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ wrote summary → {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
