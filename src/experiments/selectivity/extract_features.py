from typing import List

import json
import numpy as np
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt

import os
import h5py
import torch
from PIL import Image
from tqdm import tqdm
from skimage import measure
import concurrent.futures

from qwen_omni_utils import process_mm_info
from src.core.model_loading import load_topo_omni, MODEL_TITLE
from src.models.spatial_utils import DEFAULT_SHEET_LAYOUT as SHEET

from dotenv import load_dotenv
load_dotenv()

STIMULI_DIR = os.getenv("STIMULI_DIR")
CKPT_DIR = os.getenv("CKPT_DIR")
SAVE_DIR = os.getenv("SAVE_DIR")

def read_pickle(filepath):
    with open(filepath, "rb") as f:
        data = pkl.load(f)
    return data

def dump_pickle(data, filepath):
    with open(filepath, "wb") as f:
        pkl.dump(data, f)

def write_json(data, filepath):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

def pil_open(path: str):
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

def save_image(image_data, output_path):
    # --- Handle channel ordering ---
    if image_data.ndim == 3 and image_data.shape[0] in (1, 3, 4):
        # Likely (C, H, W) — transpose to (H, W, C)
        image_data = np.transpose(image_data, (1, 2, 0))

    # --- Normalize to uint8 if needed ---
    if image_data.dtype != np.uint8:
        image_data = np.clip(image_data, 0, 255).astype(np.uint8)

    # --- Save with PIL ---
    image = Image.fromarray(image_data)
    image.save(output_path)

@torch.no_grad()
def extract_features_for_images(
    model,
    processor,
    image_paths: List[str],
    batch_size: int = 8,
    device: str = "cuda",
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
        ).to(dev)  # dict of tensors on the correct device

        out = model(**inputs, return_dict=True)

        unified_sheet = out.unified_sheet

        unified_sheet = unified_sheet.mean(dim=0)
        unified_sheet = unified_sheet.float().numpy()
        cortical_sheets.append(unified_sheet)

    cortical_sheets = np.stack(cortical_sheets, axis=0).reshape(len(cortical_sheets), -1)
    return cortical_sheets


def load_model(model=None, device_str="cuda", baseline=False):
    model, processor, _ = load_topo_omni(model=model, device=device_str, baseline=baseline)
    return model, processor

def dump_hdf5(data, hdf5_path):
    with h5py.File(hdf5_path, "w") as f:
        for key, value in data.items():
            if isinstance(value, dict):
                # Create a group for nested dicts
                grp = f.create_group(key)
                for subkey, subvalue in value.items():
                    grp.create_dataset(subkey, data=np.array(subvalue))
            elif isinstance(value, list):
                # Convert list to numpy array
                f.create_dataset(key, data=np.array(value))
            else:
                f.create_dataset(key, data=value)

def get_selectivity_mask(stats, modality, anatomical_constraint=False, p_value_threshold=0.001, top_k_pct=1):
    t_values = stats['t']  # shape: (num_units, )
    t_values[(stats["q"] >= p_value_threshold) | (stats["t"] <= 0)] = np.nan

    t_values = t_values.reshape(W, H)
    t_values = np.rot90(t_values, k=1)

    if anatomical_constraint:
        if modality in ["language", "cognitive"]:
            t_values[144:, :] = np.nan
        elif modality == "audio":
            t_values[:144, :] = np.nan
            t_values[144:, :256] = np.nan
        elif modality == "vision":
            t_values[:144, :] = np.nan
            t_values[144:, 256:] = np.nan

    if np.isnan(t_values).any():
        t_values = np.nan_to_num(t_values, nan=-np.inf)

    t_values = t_values.flatten()

    if anatomical_constraint:
        if modality in ["language", "cognitive"]:
            total_num_units = SHEET.thinker_rows * SHEET.n_col
        elif modality == "audio":
            total_num_units = SHEET.encoder_rows * (SHEET.n_col // 2)
        elif modality == "vision":
            total_num_units = SHEET.encoder_rows * (SHEET.n_col // 2)
    else:
        total_num_units = H * W

    if top_k_pct <= 0 or top_k_pct > 100:
        selectivity_mask = (stats["q"] < p_value_threshold) & (stats["t"] > 0)
        selectivity_mask = selectivity_mask.reshape(W, H)
        selectivity_mask = np.rot90(selectivity_mask, k=1)
    else:

        active_num_units = (top_k_pct / 100) * total_num_units

        t_values_indices_sorted = np.argsort(t_values)[::-1]
        top_k_pct_indices = t_values_indices_sorted[:int(active_num_units)]
        top_k_pct_mask = np.zeros_like(t_values, dtype=bool)
        top_k_pct_mask[top_k_pct_indices] = True

        selectivity_mask = top_k_pct_mask.reshape(H, W)

    return selectivity_mask


def parallelize_feature_extraction(
    model_name,
    selectivity,
    categories,
    top_k_pcts,
    anatomical_constraint,
    W, H,
    baseline=False,
    checkpoint=None,
    overwrite=False,
    save_batch_size=256,
    start_batch=0,
    end_batch=None,
):

    # WITHOUT `checkpoint`, load_topo_omni falls back to $TOPO_OMNI_MODEL -- a different
    # run -- while `model_name` only names the output directory, so the features land
    # under the right name describing the wrong model. Same trap the selectivity runner
    # documents; name the checkpoint explicitly.
    model, processor = load_model(model=checkpoint, baseline=baseline)
    # With baseline=True, resolve_model_id returns Qwen2.5-Omni-3B and IGNORES `checkpoint`,
    # so printing the checkpoint there would name a model that was not loaded.
    print(f"> Model: {'Qwen/Qwen2.5-Omni-3B (baseline, identity adaptors)' if baseline else (checkpoint or os.getenv('TOPO_OMNI_MODEL', '<default>'))}")

    hdf5_path = f"{STIMULI_DIR}/nsd_stimuli.hdf5"

    def save_image_worker(args):
        """Save a single image - runs in thread pool."""
        img_data, path = args
        save_image(img_data, path)
        return path

    def process_category(args):
        """Process a single category/top_k_pct combo - runs in thread pool."""
        top_k_pct, category, sheets, selectivity, W, H = args
        mask = selectivity[top_k_pct][category]
        category_key = f"{category}_top{top_k_pct}"
        features_list = []
        for sheet in sheets:
            sheet = sheet.reshape(W, H).copy()  # copy to avoid shared memory
            sheet = np.rot90(sheet, k=1)
            features = sheet[mask].astype(np.float16)
            features_list.append(features)
        return category_key, np.stack(features_list, axis=0)

    os.makedirs("tmp/nsd", exist_ok=True)
    os.makedirs(f"{SAVE_DIR}/{model_name}/nsd_features", exist_ok=True)

    # `imgBrick` is (73000, 425, 425, 3) uint8 = 39.6 GiB. Reading it with [:] materialises
    # the whole thing in RAM before the first forward pass, which needs a 40 GiB+ box and
    # buys nothing: each save-batch touches 256 contiguous rows and never looks back. Keep
    # the file open and slice per batch instead -- h5py reads only the requested rows.
    image_file = h5py.File(hdf5_path, "r")
    img_brick = image_file["imgBrick"]
    num_images = img_brick.shape[0]
    num_save_batches = (num_images + save_batch_size - 1) // save_batch_size

    # Shard over save-batches so the 73,000 images can fan out to several single-GPU jobs.
    # Each batch writes its own self-contained file and reads nothing from its neighbours,
    # so disjoint ranges are safe to run concurrently. Ranges must be DISJOINT: the
    # skip-if-exists check is not a lock, so two jobs given the same batch will both start
    # it and race on the write.
    lo = max(0, start_batch)
    hi = num_save_batches if end_batch is None else min(end_batch, num_save_batches)
    print(f"> Save-batches {lo}..{hi - 1} of {num_save_batches} "
          f"({save_batch_size} images each, {num_images} total)")

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as io_executor, \
         concurrent.futures.ThreadPoolExecutor(max_workers=len(top_k_pcts) * len(categories)) as cpu_executor:

        for batch_idx in tqdm(range(lo, hi)):

            start = batch_idx * save_batch_size
            num_files_in_batch = min(save_batch_size, num_images - start)
            save_path = f"{SAVE_DIR}/{model_name}/nsd_features/nsd_batch_idx={batch_idx}_anatomical={anatomical_constraint}.hdf5"

            # Resumability: 286 batches over 73,000 images is hours of GPU time, and the
            # loop had no way to pick up after a preemption. Each batch's output is
            # self-contained, so a finished one can simply be skipped.
            if os.path.exists(save_path) and not overwrite:
                print(f"> Batch {batch_idx + 1}/{num_save_batches} already done, skipping.")
                continue

            print(f"> Processing batch {batch_idx + 1}/{num_save_batches}...")

            # one contiguous read of this batch's rows, then hand the slices to the writers
            batch_images = img_brick[start:start + num_files_in_batch]

            # --- Parallel image saving (I/O-bound) ---
            save_args = [
                (batch_images[idx], f"tmp/nsd/tmp_{start + idx}.png")
                for idx in range(num_files_in_batch)
            ]
            image_paths = list(io_executor.map(save_image_worker, save_args))

            final_data = {
                "features": {},
                "ids": list(range(start, start + num_files_in_batch)),
            }

            # --- GPU feature extraction ---
            sheets = extract_features_for_images(model, processor, image_paths, batch_size=1, device="cuda")

            # --- Parallel post-processing across categories/top_k_pcts ---
            process_args = [
                (top_k_pct, category, sheets, selectivity, W, H)
                for top_k_pct in top_k_pcts
                for category in categories
            ]
            for category_key, stacked in cpu_executor.map(process_category, process_args):
                final_data["features"][category_key] = stacked

            # --- Cleanup temp images for this batch ---
            for path in image_paths:
                try:
                    os.remove(path)
                except OSError:
                    pass

            dump_hdf5(final_data, save_path)

    image_file.close()

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Extract NSD cortical-sheet features for the brain-alignment analysis.")
    parser.add_argument("--model_name", default=MODEL_TITLE,
                        help="run title under $SAVE_DIR. Names BOTH the selectivity stats "
                             "that define the unit masks and the nsd_features/ output dir.")
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint dir to extract with. WITHOUT this, load_topo_omni "
                             "falls back to $TOPO_OMNI_MODEL -- possibly a different run -- "
                             "while --model_name only sets the output path, so the features "
                             "land under the right name describing the wrong model.")
    parser.add_argument("--stats_fwhm_mm", default="3.0",
                        help="the FWHM tag in the selectivity-stats filename, i.e. "
                             "<cat>/<cat>_fwhm<F>_selectivity_stats.pkl. Pass 'none' for the "
                             "older unsuffixed <cat>_all_selectivity_stats.pkl layout.")
    parser.add_argument("--categories",
                        default="faces,scenes,bodies,vwfa,objects,speech",
                        help="comma-separated localizers whose masks to read out.")
    parser.add_argument("--top_k_pcts", default="1,5,10",
                        help="comma-separated top-k%% levels; the paper reports top-10%%.")
    parser.add_argument("--anatomical_constraint", action="store_true",
                        help="restrict each mask to its modality band. NOTE the band cuts "
                             "in get_selectivity_mask are still the hardcoded 144/256 "
                             "literals of the 309x512 strip sheet and are WRONG for any "
                             "other layout -- leave this off until they read from SHEET.")
    parser.add_argument("--p_value_threshold", type=float, default=0.001)
    parser.add_argument("--save_batch_size", type=int, default=256)
    parser.add_argument("--start_batch", type=int, default=0,
                        help="first save-batch index to process (inclusive).")
    parser.add_argument("--end_batch", type=int, default=None,
                        help="one past the last save-batch index. With --start_batch this "
                             "shards the run across jobs; keep the ranges DISJOINT, the "
                             "skip-if-exists check is not a lock.")
    parser.add_argument("--baseline", action="store_true",
                        help="extract with the untrained Qwen2.5-Omni-3B control instead.")
    parser.add_argument("--overwrite", action="store_true",
                        help="redo save-batches whose output file already exists. Default "
                             "is to skip them, so an interrupted run resumes.")

    args = parser.parse_args()

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    top_k_pcts = [int(x) for x in args.top_k_pcts.split(",") if x.strip()]

    model_name = args.model_name
    baseline = args.baseline
    dirpath = f"{SAVE_DIR}/{model_name}"

    anatomical_constraint = args.anatomical_constraint
    H, W = SHEET.n_row, SHEET.n_col
    p_value_threshold = args.p_value_threshold

    print(f"> Sheet: {SHEET.n_row}x{SHEET.n_col} "
          f"(TOPO_SHEET_LAYOUT={os.getenv('TOPO_SHEET_LAYOUT', 'strip')})")

    selectivity = {}
    for top_k_pct in top_k_pcts:
        selectivity[top_k_pct] = {}
        for category in categories:

            # The `_all_` form below is the ORIGINAL hardcoded path and matches nothing the
            # current selectivity runner writes -- it emits
            # <cat>_fwhm<F>_selectivity_stats.pkl. Keep both so old result trees still load.
            if str(args.stats_fwhm_mm).lower() == "none":
                selectivity_path = f"{dirpath}/{category}/{category}_all_selectivity_stats.pkl"
            else:
                selectivity_path = (f"{dirpath}/{category}/"
                                    f"{category}_fwhm{args.stats_fwhm_mm}_selectivity_stats.pkl")
            if not os.path.exists(selectivity_path):
                raise SystemExit(
                    f"no selectivity stats at {selectivity_path}. Run the selectivity "
                    f"sweep for '{category}' on {model_name} first, or pass a different "
                    f"--stats_fwhm_mm.")

            selectivity[top_k_pct][category] = get_selectivity_mask(
                stats=read_pickle(selectivity_path),
                modality=category,
                anatomical_constraint=anatomical_constraint,
                p_value_threshold=p_value_threshold,
                top_k_pct=top_k_pct,
            )
            print(f"> mask {category} top-{top_k_pct}%: "
                  f"{int(selectivity[top_k_pct][category].sum())} units")

    parallelize_feature_extraction(
        model_name=model_name,
        selectivity=selectivity,
        categories=categories,
        top_k_pcts=top_k_pcts,
        anatomical_constraint=anatomical_constraint,
        W=W,
        H=H,
        baseline=baseline,
        checkpoint=args.checkpoint,
        overwrite=args.overwrite,
        save_batch_size=args.save_batch_size,
        start_batch=args.start_batch,
        end_batch=args.end_batch,
    )
