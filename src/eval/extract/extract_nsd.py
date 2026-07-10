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
            total_num_units = 144 * 512
        elif modality == "audio":
            total_num_units = 160 * 256
        elif modality == "vision":
            total_num_units = 160 * 256
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
):

    save_batch_size = 256

    model, processor = load_model(baseline=baseline)

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

    with h5py.File(hdf5_path, "r") as image_data:
        num_images = len(image_data["imgBrick"])
        num_save_batches = (num_images + save_batch_size - 1) // save_batch_size
        img_brick = image_data["imgBrick"][:]

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as io_executor, \
         concurrent.futures.ThreadPoolExecutor(max_workers=len(top_k_pcts) * len(categories)) as cpu_executor:

        for batch_idx in tqdm(range(num_save_batches)):

            print(f"> Processing batch {batch_idx + 1}/{num_save_batches}...")

            start = batch_idx * save_batch_size
            num_files_in_batch = min(save_batch_size, num_images - start)
            save_path = f"{SAVE_DIR}/{model_name}/nsd_features/nsd_batch_idx={batch_idx}_anatomical={anatomical_constraint}.hdf5"

            # --- Parallel image saving (I/O-bound) ---
            save_args = [
                (img_brick[start + idx], f"tmp/nsd/tmp_{start + idx}.png")
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

if __name__ == "__main__":

    categories = [
        "faces",
        "scenes",
        "bodies",
        "vwfa",
        "objects",
        "speech",
    ]

    # Topographic model by default. For the non-topo control, set baseline=True below
    # and model_name=BASELINE_TITLE (its precomputed selectivity stats live under that title).
    model_name = MODEL_TITLE
    baseline = False
    dirpath = f"{SAVE_DIR}/{model_name}"

    anatomical_constraint = False
    H, W = 304, 512
    p_value_threshold = 0.001
    top_k_pcts = [1, 5, 10]

    selectivity = {}
    for top_k_pct in top_k_pcts:
        selectivity[top_k_pct] = {}
        for category in categories:
     
            selectivity_path = f"{dirpath}/{category}/{category}_all_selectivity_stats.pkl"

            selectivity[top_k_pct][category] = get_selectivity_mask(
                stats=read_pickle(selectivity_path),
                modality=category,
                anatomical_constraint=anatomical_constraint,
                p_value_threshold=p_value_threshold,
                top_k_pct=top_k_pct,
            )

    parallelize_feature_extraction(
        model_name=model_name,
        selectivity=selectivity,
        categories=categories,
        top_k_pcts=top_k_pcts,
        anatomical_constraint=anatomical_constraint,
        W=W,
        H=H,
        baseline=baseline,
    )



        
        

   


      


