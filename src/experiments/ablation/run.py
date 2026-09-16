import os
import json
import torch
import numpy as np
import pickle as pkl

import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image
from glob import glob
from tqdm import tqdm
from qwen_omni_utils import process_mm_info
from src.core.model_loading import load_topo_omni, MODEL_TITLE
from src.models.spatial_utils import DEFAULT_SHEET_LAYOUT as SHEET

from dotenv import load_dotenv
load_dotenv()

REPO_DIR = os.getenv("REPO_DIR")
CKPT_DIR = os.getenv("CKPT_DIR")
SAVE_DIR = os.getenv("SAVE_DIR")
STIMULI_DIR = os.getenv("STIMULI_DIR")

def pil_open(path: str):
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # resize image to 640x360
    img = img.resize((640, 360))
    return img

def read_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pkl.load(f)
    
def write_json(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def generate(model, processor, prompt, stimuli_path):

    stimuli = {"type": "video", "video": stimuli_path} if stimuli_path.endswith(".mp4") else {"type": "image", "image": pil_open(stimuli_path)}

    conversation = [
        # {
        #     "role": "system", 
        #     "content": [
        #         # {"type": "text", "text": "You are a helpful and precise assistant for visual recognition tasks. Select the correct answer from the options based on the image provided. If the image is unclear, make your best guess. Always choose one of the provided options and do not provide any explanations."},
        #         {"type": "text", "text": "You are a helpful and precise assistant for visual recognition tasks. Describe what is in the image. If the image is unclear, make your best guess."},
        #     ],
        # },
        {
            "role": "user",
            "content": [
                stimuli,
                {"type": "text", "text": prompt}
            ],
        }
    ]

    # set use audio in video
    USE_AUDIO_IN_VIDEO = True

    # Preparation for inference
    text = processor.apply_chat_template(
        conversation, 
        add_generation_prompt=True, 
        tokenize=False,
    )

    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = processor(
        text=text, 
        audio=audios, 
        images=images, 
        videos=videos, 
        return_tensors="pt", 
        padding=True, 
        use_audio_in_video=USE_AUDIO_IN_VIDEO, 
    )

    inputs = {
        k: v.to(model.device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    with torch.inference_mode():
        text_ids = model.generate(
            **inputs,
            use_audio_in_video=USE_AUDIO_IN_VIDEO,
            max_new_tokens=32, 
            min_new_tokens=1,
            do_sample=False,
        )

    input_len = inputs["input_ids"].shape[1]
    new_tokens = text_ids[:, input_len:]

    text_out = processor.batch_decode(
        new_tokens,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )[0]

    return text_out.split("<|im_end|>")[0].strip()

def make_cortical_hook(layer_idx: int, parent_module: str, selectivity_mask: torch.Tensor, mask_fn,
                       mean_activations=None, stimulate: bool = False, t_mask=None, baseline_activations=None):
    """
    Returns a forward hook for cortical_encode that applies a mask
    to its *output* (zero ablation where mask == 0) or replaces ablated
    units with mean activations when provided.
    """
    def hook(module, input, output):
        # output: tensor of shape [..., out_features]
        mask, t_vals_mask = mask_fn(layer_idx, parent_module, selectivity_mask, t_mask)

        # Ensure mask is on the right device/dtype & broadcastable
        mask = mask.to(device=output.device, dtype=output.dtype)
        # If mask is 1D [D], unsqueeze to match [..., D]
        if mask.dim() == 1 and output.dim() > 1:
            mask_expanded = mask.view(*(1,) * (output.dim() - 1), -1)
            t_vals_mask = t_vals_mask.view(*(1,) * (output.dim() - 1), -1)
        else:
            mask_expanded = mask

        t_vals_mask = t_vals_mask.to(device=output.device, dtype=output.dtype)
        H, W = 26, 46  # replace with actual grid
        B = 1 # border size to exclude when stimulating/ablating (e.g. if B=1, we only modify units in the center 25x45 region)

        center_indices = [
            r * W + c
            for r in range(B, H-B)
            for c in range(B, W-B)
        ]

        center_token_ids = torch.tensor(center_indices, dtype=torch.long)

        if mean_activations is not None:
            layer_key = (parent_module, layer_idx)
            layer_mean_actv = mean_activations.get(layer_key).view(*(1,) * (output.dim() - 1), -1)
            layer_baseline_actv = baseline_activations.get(layer_key).view(*(1,) * (output.dim() - 1), -1)
        
        if output.dim() == 2:
            if stimulate:
                # Stimulation is a fixed steering vector: the mean-activation difference on
                # the selected units, with NO scalar gain. The two additive forms below are
                # the parked alternative that would take one.
                #
                # There used to be a --stimulation_strength flag feeding a
                # STIMULATION_STRENGTH global here. Both lines were commented out, so the
                # flag reached the filename and the results metadata but never the forward
                # pass: a sweep over strengths 1/5/10 produced byte-identical outputs. It
                # has been removed. To reintroduce a real gain, thread a scalar through
                # make_cortical_hook and scale `steering_vector` below (or re-enable one of
                # these), and add it back to get_save_path so runs do not overwrite.
                # output[center_token_ids, :] = output[center_token_ids, :] + ((1-mask_expanded) * STRENGTH)
                # output = output + ((1-mask_expanded) * STRENGTH)
                steering_vector = (layer_mean_actv - layer_baseline_actv) * (1 - mask_expanded) if (layer_mean_actv is not None and layer_baseline_actv is not None) else 0
            else:
                # output[center_token_ids, :] = output[center_token_ids, :] * mask_expanded  # zero-ablate
                # output = output * mask_expanded  # zero-ablate
                steering_vector = (layer_baseline_actv - layer_mean_actv) * (1 - mask_expanded) if (layer_mean_actv is not None and layer_baseline_actv is not None) else 0
            
            output = output + steering_vector

            # if torch.any(mask_expanded == 0):
            #     output = output * torch.randn_like(output) * 0.1 * (1 - mask_expanded)  # noisy ablation

            return output
        elif output.dim() == 3:
            return output * mask_expanded # zero-ablate where mask_expanded == 0
        else:
            raise ValueError(f"Unexpected output dim: {output.dim()}")

        # if mean is None:
        #     return output * mask_expanded

        # mean = mean.to(device=output.device, dtype=output.dtype)
        # if mean.dim() == 1 and output.dim() > 1:
        #     mean_expanded = mean.view(*(1,) * (output.dim() - 1), -1)
        # else:
        #     mean_expanded = mean

        # return output * mask_expanded + mean_expanded * (1 - mask_expanded)
    
    return hook


def _encoder_cortical_slots(tower):
    """Yield ``(slot_idx, z_hook)`` for one encoder tower, front-end first.

    On a layout that carries a front-end slot, slot 0 is the conv front-end
    (`patch_embed` / `conv2`) and slots 1..L are the transformer blocks. On one that does
    not (`tiled32`), slots 0..L-1 are the blocks and there is no front-end module at all.
    Either way this is the order the tower appends them to its cortical sheet, so it is
    also the depth order `get_cortical_mask` indexes -- keep the two in step or ablation
    will mask the wrong units.
    """
    offset = 0
    if getattr(tower, "has_frontend_slot", True):
        yield 0, tower.frontend_cortical_adaptor.z_hook
        offset = 1
    for block_idx, adaptor in enumerate(tower.cortical_adaptors):
        yield block_idx + offset, adaptor.z_hook


def register_hooks(
        model, 
        mask: np.ndarray, 
        stimulate: bool = False, 
        mean_activations=None, 
        baseline_activations=None,
        t_mask=None
):
    hooks = []

    t_mask = torch.tensor(t_mask.copy(), dtype=torch.float32)
    for layer_idx, layer in enumerate(model.model.layers):

        cortical = model.model.cortical_adaptors[layer_idx].z_hook  # Linear(2048 -> 2048)
        h = cortical.register_forward_hook(
            make_cortical_hook(layer_idx, "language", mask, get_cortical_mask, mean_activations=mean_activations, stimulate=stimulate, t_mask=t_mask, baseline_activations=baseline_activations)
        )
        hooks.append(h)

    for slot_idx, cortical in _encoder_cortical_slots(model.visual):

        h = cortical.register_forward_hook(
            make_cortical_hook(slot_idx, "vision", mask, get_cortical_mask, mean_activations=mean_activations, stimulate=stimulate, t_mask=t_mask, baseline_activations=baseline_activations)
        )
        hooks.append(h)

    for slot_idx, cortical in _encoder_cortical_slots(model.audio_tower):

        h = cortical.register_forward_hook(
            make_cortical_hook(slot_idx, "audio", mask, get_cortical_mask, mean_activations=mean_activations, stimulate=stimulate, t_mask=t_mask, baseline_activations=baseline_activations)
        )
        hooks.append(h)

    return hooks 


def clean_hooks(hooks):
    for h in hooks:
        h.remove()  
    hooks.clear()


_LAYER_UNITS = None


def layer_unit_indices():
    """``(parent_module, layer_idx) -> global unit indices``, in sheet-write order.

    get_cortical_mask() used to derive this geometrically, slicing the sheet into
    full-width row strips of `rows_per_encoder_layer`. That is the STRIP layout. In TILED
    mode a layer owns a compact 2-D tile and `rows_per_encoder_layer` is the tile HEIGHT,
    so the strip formula asks for 64 * 288 = 18432 units per encoder layer where the band
    actually holds 147456 / 32 = 4608 -- a 4x overshoot that raised
    "cannot reshape array of size 147456 into shape (32,18432)".

    Rather than special-case the arithmetic, take the tiles from the same `_tile_blocks`
    the sheet was BUILT with, so this cannot drift from the real layout. Block order there
    is vision slots, then audio slots, then thinker layers -- matching the depth order
    `_encoder_cortical_slots` yields and the layer order of `model.model.layers`.
    """
    global _LAYER_UNITS
    if _LAYER_UNITS is not None:
        return _LAYER_UNITS

    n_col, out = SHEET.n_col, {}

    def flat(r0, r1, c0, c1):
        return np.array([r * n_col + c for r in range(r0, r1) for c in range(c0, c1)],
                        dtype=np.int64)

    if SHEET.mode == "tiled":
        from src.models.spatial_utils import _tile_blocks
        n_enc = SHEET.num_encoder_layers
        for b, (row_sl, col_sl) in enumerate(_tile_blocks(SHEET)):
            if b < n_enc:
                key = ("vision", b)
            elif b < 2 * n_enc:
                key = ("audio", b - n_enc)
            else:
                key = ("language", b - 2 * n_enc)
            out[key] = flat(row_sl.start, row_sl.stop, col_sl.start, col_sl.stop)
    else:
        half = n_col // 2
        for li in range(SHEET.num_encoder_layers):
            r0 = li * SHEET.rows_per_encoder_layer
            r1 = r0 + SHEET.rows_per_encoder_layer
            out[("vision", li)] = flat(r0, r1, 0, half)
            out[("audio", li)] = flat(r0, r1, half, n_col)
        for li in range(SHEET.num_thinker_layers):
            r0 = SHEET.encoder_rows + li * SHEET.rows_per_thinker_layer
            r1 = r0 + SHEET.rows_per_thinker_layer
            out[("language", li)] = flat(r0, r1, 0, n_col)

    _LAYER_UNITS = out
    return out


def sheet_to_unit_vector(sheet, coords):
    """Analysis-frame sheet -> one value per GLOBAL UNIT INDEX.

    The hook needs each unit's own value, and `build_sheet_coordinates` scatters a layer's
    units across its tile (only ~0.02% of tiled32 units land on their identity position),
    so reading the sheet positionally masks the wrong units. `place_on_grid` writes
    grid[col, row] and the analysis frame is rot90(grid, 1); undo that and read each unit
    at its TRAINED coordinate.
    """
    g = np.rot90(np.asarray(sheet), k=-1)          # (n_col, n_row) -> g[c, r]
    ij = np.asarray(coords).astype(np.int64)
    return g[ij[:, 1], ij[:, 0]]


def get_cortical_mask(layer_idx: int, parent_module: str, selectivity_mask, t_mask):
    """Mask for one layer, gathered from the per-unit vectors by that layer's unit indices.

    `selectivity_mask` / `t_mask` arrive as 1-D arrays indexed by global unit index (see
    sheet_to_unit_vector), not as 2-D sheets. Ablation keeps the UNSELECTED units, hence
    the 1 - mask.
    """
    idx = layer_unit_indices()[(parent_module, layer_idx)]
    keep = 1 - np.asarray(selectivity_mask).reshape(-1)[idx]
    tv = np.asarray(t_mask).reshape(-1)[idx]
    return torch.tensor(keep), torch.tensor(tv)


def register_mean_hooks(model, mean_stats):
    hooks = []

    def make_mean_hook(layer_idx: int, parent_module: str):
        def hook(module, input, output):
            out = output.detach()
            reduce_dims = tuple(range(out.dim() - 1)) if out.dim() > 1 else ()
            mean = out.mean(dim=reduce_dims) if reduce_dims else out
            key = (parent_module, layer_idx)
            if key not in mean_stats:
                mean_stats[key] = {
                    "sum": torch.zeros_like(mean),
                    "count": 0,
                }
            mean_stats[key]["sum"] += mean
            mean_stats[key]["count"] += 1
        return hook

    for layer_idx, layer in enumerate(model.model.layers):
        cortical = model.model.cortical_adaptors[layer_idx].z_hook
        hooks.append(cortical.register_forward_hook(make_mean_hook(layer_idx, "language")))

    for slot_idx, cortical in _encoder_cortical_slots(model.visual):
        hooks.append(cortical.register_forward_hook(make_mean_hook(slot_idx, "vision")))

    for slot_idx, cortical in _encoder_cortical_slots(model.audio_tower):
        hooks.append(cortical.register_forward_hook(make_mean_hook(slot_idx, "audio")))

    return hooks


def compute_mean_activations(model, processor, prompt, stimuli_paths):
    mean_stats = {}
    hooks = register_mean_hooks(model, mean_stats)
    try:
        for stimuli_path in tqdm(stimuli_paths, desc="Computing mean activations"):
            _ = generate(model, processor, prompt, stimuli_path)
    finally:
        clean_hooks(hooks)

    mean_activations = {}
    for key, stats in mean_stats.items():
        if stats["count"] == 0:
            continue
        mean_activations[key] = stats["sum"] / stats["count"]
    return mean_activations

def cache_mean_activations(model, processor, prompt, save_dir):
    # torch.save does not create parents, so every run computed all four categories on the
    # GPU and then died at the first save with "Parent directory ... does not exist".
    os.makedirs(save_dir, exist_ok=True)

    categories = ["faces", "objects", "scenes", "bodies"]

    for category in categories:
        stimuli_dir = f"{STIMULI_DIR}/marvi_{category}/ON"
        stimuli_paths = glob(f"{stimuli_dir}/*.jpg")
        stimuli_paths = [p for p in stimuli_paths if int(os.path.basename(p).split("_")[0][-1]) % 2 == 1] 

        mean_activations = compute_mean_activations(model, processor, prompt, stimuli_paths)
        save_path = f"{save_dir}/{category}_mean_activations.pt"
        torch.save(mean_activations, save_path)

def load_mean_activations(categories, save_dir):

    mean_activations = {}
    for category in categories:
        save_path = f"{save_dir}/{category}_mean_activations.pt"
        mean_activations[category] = torch.load(save_path)

    all_keys = set(k for cat in categories for k in mean_activations[cat].keys())
    for key in all_keys:
        category_means = [mean_activations[cat][key] for cat in categories if key in mean_activations[cat]]
        if category_means:
            mean_activations[key] = torch.stack(category_means, dim=0).mean(dim=0)    
    return mean_activations


def get_save_path(ABLATE, STIMULATE, STIMULI, LOCALIZER, MODE, percentage, num_units_masked, p_threshold):
    if not ABLATE and not STIMULATE:
        save_path = f"{SAVE_DIR}/ablation/ablation_stimuli={STIMULI}_localizer=no-ablation.json"
    else:
        if MODE == "top_k":
            save_path = f"{SAVE_DIR}/ablation/ablation_stimuli={STIMULI}_localizer={LOCALIZER}_k={num_units_masked}_stimulate={STIMULATE}.json"
        elif MODE == "top_p":
            save_path = f"{SAVE_DIR}/ablation/ablation_stimuli={STIMULI}_localizer={LOCALIZER}_perc={percentage:.2f}_stimulate={STIMULATE}.json"
        elif MODE == "p_value":
            save_path = f"{SAVE_DIR}/ablation/ablation_stimuli={STIMULI}_localizer={LOCALIZER}_p={p_threshold}_stimulate={STIMULATE}.json"
        elif MODE == "cluster":
            save_path = f"{SAVE_DIR}/ablation/ablation_stimuli={STIMULI}_localizer={LOCALIZER}_cluster{CLUSTER_ID}_perc={percentage:.1f}_stimulate={STIMULATE}.json"

    save_path = save_path.replace(".json", "_v4.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    return save_path

if __name__ == "__main__":

    seed = 42
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    import argparse
    parser = argparse.ArgumentParser(description="Run ablation experiments on Qwen2-5 OmniThinker.")
    parser.add_argument("--ablate", action="store_true", help="Whether to ablate selective units.")
    parser.add_argument("--stimulate", action="store_true", help="Whether to stimulate selective units.")
    parser.add_argument("--stimuli", type=str, default="bodies", help="Which stimuli to test on. Options: {faces, objects, scenes, bodies}")
    parser.add_argument("--localizer", type=str, default=None, help="Which localizer to use for selecting units to ablate. Options: {faces, objects, scenes, bodies}")
    parser.add_argument("--mode", type=str, default=None, help="How to select units to ablate. Options: {top_k, top_p, p_value, cluster}")
    parser.add_argument("--percentage", type=float, default=None, help="Percentage of vision-selective units to ablate (only for top_k and top_p modes).")
    parser.add_argument("--model_name", type=str, default=MODEL_TITLE,
                        help="run title under $SAVE_DIR: names both the selectivity stats "
                             "read and the ablation/ output written. Was hardcoded to "
                             "MODEL_TITLE (the strip model) while the WEIGHTS came from "
                             "$TOPO_OMNI_MODEL, so evaluating any other run silently mixed "
                             "one model's weights with another's stats and output path.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="checkpoint dir to load. Default: $TOPO_OMNI_MODEL.")
    parser.add_argument("--coords", type=str, default=None,
                        help="coords.npy the run was TRAINED with. Required in practice: "
                             "without it the sheet mask is applied to hidden units under "
                             "the identity-grid assumption, which holds for 0.02% of units "
                             "on a shuffled run. See unshuffle_sheet_to_unit_space().")
    parser.add_argument("--stats_fwhm_mm", type=float, default=None,
                        help="FWHM naming the selectivity-stats file, e.g. 3.0 for "
                             "'<loc>_even_fwhm3.0_selectivity_stats.pkl'. Omit for the "
                             "legacy unsmoothed filename.")
    args = parser.parse_args()

    assert not (args.ablate and args.stimulate), "Cannot both ablate and stimulate at the same time."
    assert args.localizer in ["faces", "objects", "scenes", "bodies", None], "Invalid localizer. Must be one of {faces, objects, scenes, bodies}."

    RANDOM_UNITS = False
    MEAN_ABLATE = False
    CLUSTER_ID = 0

    ABLATE = args.ablate
    STIMULATE = args.stimulate
    LOCALIZER = args.localizer
    STIMULI = args.stimuli
    MODE = args.mode
    PERCENTAGE = args.percentage

    print(f"ABALTE: {ABLATE} | LOCALIZER: {LOCALIZER} | STIMULI: {STIMULI} | MODE: {MODE} | PERCENTAGE: {PERCENTAGE}")

    model_name = args.model_name
    SAVE_DIR = f"{SAVE_DIR}/{model_name}"
    _stats_suffix = f"_fwhm{args.stats_fwhm_mm}" if args.stats_fwhm_mm is not None else ""
    _coords = np.load(args.coords) if args.coords else None
    if _coords is not None:
        print(f"> Unit coordinates: {args.coords} {_coords.shape}")
    else:
        print("> WARNING: no --coords given; the sheet mask will be applied under the "
              "identity-grid assumption, which is wrong for any shuffled run.")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    percentage = PERCENTAGE if ABLATE or STIMULATE else 0.0
    p_threshold = 0.001
    num_units_masked = 0

    if ABLATE or STIMULATE:

        model_stats_path = f"{SAVE_DIR}/{LOCALIZER}/{LOCALIZER}_even{_stats_suffix}_selectivity_stats.pkl"
        model_stats = read_pickle(model_stats_path)

        p_values = model_stats["p"].reshape(SHEET.n_col, SHEET.n_row)
        p_values = np.rot90(p_values, k=1)

        t_mask = model_stats["t"].reshape(SHEET.n_col, SHEET.n_row)
        t_mask = np.rot90(t_mask, k=1)
        t_mask[np.isnan(t_mask)] = 0

        vision_mask = np.zeros_like(t_mask)
        vision_mask[SHEET.thinker_rows:, :SHEET.n_col // 2] = 1  # vision cortical units

        if MODE in ["top_k", "top_p"]:
            t_mask = t_mask * vision_mask
            total_num_units = SHEET.encoder_rows * (SHEET.n_col // 2)
            # total_num_units = SHEET.num_units
            if MODE == "top_k":
                num_units_to_mask = int(percentage)
            elif MODE == "top_p":
                num_units_to_mask = int(total_num_units * (percentage / 100.0))

            flat_indices = np.argsort(t_mask.flatten(), axis=None)[-num_units_to_mask:]
            mask = np.zeros_like(t_mask).flatten()
            mask[flat_indices] = 1

            mask = mask.reshape(SHEET.n_row, SHEET.n_col)
            t_mask[vision_mask==0] = np.nan
        elif MODE == "p_value":
            vision_mask[vision_mask == 0] = np.inf
            p_values = p_values * vision_mask
            mask = (p_values < p_threshold) & (t_mask > 0)
            print(f"Num Significant Units (p < {p_threshold}): {mask.sum()}")
        elif MODE == "cluster":
            PERCENTAGE = int(PERCENTAGE) if PERCENTAGE == 1 else PERCENTAGE
            mask_path = f"{SAVE_DIR}/{LOCALIZER}/cluster_selectivity_{LOCALIZER}_t_values_top{PERCENTAGE}_island=largest_mask.pkl"
            mask = read_pickle(mask_path)

        t_mask[~mask.astype(bool)] = np.nan
        num_units_masked = int(mask.sum())

        # Count the units FIRST (above), in sheet space where the anatomical bands still
        # mean what they say, then convert to per-unit vectors for the hooks.
        if _coords is None:
            raise SystemExit("--coords is required: without it the sheet mask cannot be "
                             "mapped to hidden units and the ablation hits ~0.02% of the "
                             "units it reports.")
        mask = sheet_to_unit_vector(mask, _coords)
        t_mask = sheet_to_unit_vector(t_mask, _coords)
        print(f"> Per-unit mask: {int(np.nan_to_num(mask).sum())} of {mask.size} units")
        print(f"> Num Units Masked: {num_units_masked}/{total_num_units} ({(num_units_masked/total_num_units)*100:.2f}%)")

    if RANDOM_UNITS and ABLATE:

        model_stats_path = f"{SAVE_DIR}/{LOCALIZER}/{LOCALIZER}_selectivity_stats.pkl"
        model_stats = read_pickle(model_stats_path)

        t_mask = model_stats["t"].reshape(SHEET.n_col, SHEET.n_row)
        t_mask = np.rot90(t_mask, k=1)
        t_mask[np.isnan(t_mask)] = 0

        vision_mask = np.zeros_like(t_mask)
        vision_mask[SHEET.thinker_rows:, :SHEET.n_col // 2] = 1  # vision cortical units

        # ablate random units from vision units
        total_num_units = SHEET.encoder_rows * (SHEET.n_col // 2)
        num_units_to_mask = int(total_num_units * (percentage / 100.0))

        vision_mask = vision_mask * (1 - mask) # remove significant units from vision mask
        print(f"Other Vision Units: {vision_mask.sum()}")

        vision_unit_indices = np.where(vision_mask.flatten() == 1)[0]
        random_indices = np.random.choice(vision_unit_indices, size=num_units_to_mask, replace=False)
        random_mask = np.zeros_like(vision_mask).flatten()
        random_mask[random_indices] = 1
        random_mask = random_mask.reshape(SHEET.n_row, SHEET.n_col)
        mask = random_mask.copy()

        print(f"Total Num Units: {total_num_units} | Num Random Units to Mask: {num_units_to_mask}")

        sns.heatmap(random_mask, cmap="viridis")
        plt.title(f"Random {percentage:.2f}% Units Masked Out")
        plt.xlabel("Cortical Units (Cols)")
        plt.ylabel("Cortical Units (Rows)")
        plt.savefig(f"{SAVE_DIR}/{LOCALIZER}/{LOCALIZER}_random_seed={seed}_{percentage:.2f}perc_units_masked.png")    

    save_path = get_save_path(ABLATE, STIMULATE, STIMULI, LOCALIZER, MODE, percentage, num_units_masked, p_threshold)
    if os.path.exists(save_path):
        print(f"Save path {save_path} already exists.")
        exit(0)

    model, processor, _ = load_topo_omni(model=args.checkpoint, device=DEVICE)
    print(f"> Model: {args.checkpoint or os.getenv('TOPO_OMNI_MODEL', '<default>')}")
    
    # options = ["Yes", "No"]
    # options = ["An indoor or outdoor location", "A face", "A toy or object", "A body part"]
    # prompt = "What do you see?\nOptions:\n{options}\nAnswer:"

    # if STIMULI == "faces":
    #     prompt = "Is there a face in the image?\nOptions:\n{options}\nAnswer:"
    # elif STIMULI == "scenes":
    #     prompt = "Does the image show an indoor or outdoor location?\nOptions:\n{options}\nAnswer:"
    # elif STIMULI == "bodies":
    #     prompt = "Does the image contain a body part that is not a face?\nOptions:\n{options}\nAnswer:"
    # elif STIMULI == "objects":
    #     prompt = "Is there a toy or object in the image?\nOptions:\n{options}\nAnswer:"
    # else:
    #     raise ValueError(f"Unknown STIMULI: {STIMULI}")

    prompt = "What do you see in the image?"
    # prompt = "Is there a face in the image?\nOptions:\n- Yes\n- No\nAnswer:"
    stimuli_dir = f"{STIMULI_DIR}/marvi_{STIMULI}/ON"
    stimuli_paths = glob(f"{stimuli_dir}/*.jpg")
    stimuli_paths = [p for p in stimuli_paths if int(os.path.basename(p).split("_")[0][-1]) % 2 == 1] 
  
    results = {
        "stimuli": STIMULI,
        "stimuli_path": stimuli_dir,
        "ablate": ABLATE,
        "stimulate": STIMULATE,
        "mean_ablate": MEAN_ABLATE if ABLATE else False,
        "localizer": LOCALIZER,
        "cluster_id": CLUSTER_ID if MODE == "cluster" else None,
        "percentage_units_masked": percentage if ABLATE else 0.0,
        "num_units_masked": num_units_masked if ABLATE else 0,
        "prompt": prompt,
        "responses": [],
    }

    mean_activations = None    
    save_dir = f"{SAVE_DIR}/mean_activations"

    if not os.path.exists(os.path.join(save_dir, "faces_mean_activations.pt")):
        cache_mean_activations(model, processor, prompt, save_dir)
    

    if ABLATE or STIMULATE:
        other_categories = ["faces", "objects", "scenes", "bodies"]
        other_categories.remove(LOCALIZER)
        baseline_activations = load_mean_activations(other_categories, save_dir)
        mean_activations = load_mean_activations([LOCALIZER], save_dir)
        
        hooks = register_hooks(
            model, 
            mask, 
            stimulate=STIMULATE, 
            mean_activations=mean_activations, 
            baseline_activations=baseline_activations,
            t_mask=t_mask
        )

    for i in tqdm(range(len(stimuli_paths))):
        stimuli_path = stimuli_paths[i]

        # shuffled_options = options.copy()
        # np.random.shuffle(shuffled_options)
        # shuffled_options = shuffled_options + ["Something else", "Nothing"]  # add decoy options
        # formatted_options = "\n".join([f"- {opt}" for opt in shuffled_options])
        # formatted_prompt = prompt.format(options=formatted_options)
        output = generate(model, processor, prompt, stimuli_path)
        print(f"Stimulus: {os.path.basename(stimuli_path)} | Output: {output}")

        results["responses"].append({
            "stimulus": os.path.basename(stimuli_path),
            "output": output,
            # "options": shuffled_options,
        })

    if ABLATE or STIMULATE:
        clean_hooks(hooks)

    write_json(save_path, results)
