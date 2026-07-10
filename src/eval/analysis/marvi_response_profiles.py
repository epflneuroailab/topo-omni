import os
import yaml
import torch
import numpy as np
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.ndimage import binary_opening
from src.core.model_loading import load_topo_omni, unified_grid_coords
from src.eval.run.run_selectivity import extract_features_for_video, extract_features_for_audio, place_on_grid, NeuronSmoothingConv
from src.visualize.selectivity import remove_small_components

from dotenv import load_dotenv
load_dotenv()

CKPT_DIR = os.getenv("CKPT_DIR")
SAVE_DIR = os.getenv("SAVE_DIR")
STIMULI_DIR = os.getenv("STIMULI_DIR")

def read_pickle(filepath):
    with open(filepath, "rb") as f:
        data = pkl.load(f)
    return data

def dump_pickle(data, filepath):
    with open(filepath, "wb") as f:
        pkl.dump(data, f)

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_selectivity_mask(model_name, 
    category, 
    modality, 
    anatomical_constraint=False, 
    top_k_pct=1, 
    p_value_threshold=0.001, 
    odd_or_even="even", 
    fwhm_mm=4.0
):
    dirpath = f"{SAVE_DIR}/{model_name}"
    if odd_or_even is not None:
        filepath = f"{dirpath}/{category}/{category}_{odd_or_even}_fwhm{fwhm_mm}_selectivity_stats.pkl"
    else:
        filepath = f"{dirpath}/{category}/{category}_fwhm{fwhm_mm}_selectivity_stats.pkl"
    stats = read_pickle(filepath)

    t_values = stats['t']  # shape: (num_units, )
    t_values = t_values.reshape(W, H)
    t_values = np.rot90(t_values, k=1)

    if anatomical_constraint:
        if modality in ["language", "cognitive"]:
            t_values[144:, :] = -np.inf
            total_num_units = 144 * 512
        elif modality == "audio":
            t_values[:144, :] = -np.inf
            t_values[144:, :256] = -np.inf
            total_num_units = 160 * 256
        elif modality == "vision":
            t_values[:144, :] = -np.inf
            t_values[144:, 256:] = -np.inf
            total_num_units = 160 * 256
    else:
        total_num_units = H * W

    t_values = t_values.flatten()
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
    
    t_values = t_values.reshape(H, W)

    # selectivity_mask = binary_opening(selectivity_mask, structure=np.ones((3, 3)))
    # selectivity_mask = remove_small_components(selectivity_mask, min_size=20)
    print(f"> Total units: {total_num_units} | Active units: {selectivity_mask.sum()} ({100 * selectivity_mask.sum() / total_num_units:.2f}%)")
    return selectivity_mask, t_values

def collect_files(dirpath, odd_or_even="odd"):
    files = []
    for filename in os.listdir(dirpath):
        if not filename.endswith(".mp4"):
            continue
        file_path = os.path.join(dirpath, filename)
        basename = os.path.basename(file_path)
        run_num = int(''.join(filter(str.isdigit, basename.split("_")[0])))
        if odd_or_even == "odd" and run_num % 2 == 1:
            files.append(file_path)
        elif odd_or_even == "even" and run_num % 2 == 0:
            files.append(file_path)
    if not files:
        raise FileNotFoundError(f"No video found in {dirpath}")
    return files


def collect_audio_files(dirpath):
    files = []
    for filename in os.listdir(dirpath):
        if not filename.endswith(".wav"):
            continue
        file_path = os.path.join(dirpath, filename)
        files.append(file_path)
    if not files:
        raise FileNotFoundError(f"No audio found in {dirpath}")
    return files

def plot_cortical_sheet(sheet, selectivity_mask, title):
    plt.figure(figsize=(12, 6))
    sns.heatmap(sheet, mask=~selectivity_mask, cmap='viridis')
    plt.title(title)
    plt.axis('off')
    plt.savefig(f"{title}_cortical_sheet.png")

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Evaluate response profiles for different conditions.")
    parser.add_argument("--model_name", type=str, default="topo-omni", help="Model name to evaluate")
    parser.add_argument("--localizer", type=str, default="faces", help="Localizer condition to evaluate (e.g., faces, bodies, scenes)")
    parser.add_argument("--top_k_pct", type=int, default=1, help="Top k percent of units to keep")
    parser.add_argument("--odd_or_even", type=str, default=None, help="Whether to use odd or even runs for evaluation")
    parser.add_argument("--fwhm_mm", type=float, default=4.0, help="FWHM in mm for smoothing the cortical sheet")
    parser.add_argument("--anatomical_constraint", type=str, default="true", choices=["true", "false"], help="Whether to apply anatomical constraints for selectivity mask")

    args = parser.parse_args()
    localizer = args.localizer

    p_value_threshold = 0.001
    top_k_pct = args.top_k_pct  # top k percent of units to keep

    fwhm_mm = args.fwhm_mm
    anatomical_constraint = args.anatomical_constraint == "true"
    trial_type = args.odd_or_even  # "odd" or "even"

    H, W = 304, 512  # unified sheet size
    resolution_mm = 1.0
    smoother = NeuronSmoothingConv(fwhm_mm=fwhm_mm, resolution_mm=resolution_mm)

    model_name = args.model_name
    coords = unified_grid_coords()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    conditions = ["faces", "bodies", "objects", "scenes", "words_scr_objects", "false_belief", "false_photo", "math", "nonwords", "quilted_speech"]
    if localizer == "pernet_fold_A":
        conditions = ["pernet_fold_B/ON", "pernet_fold_B/OFF"]
    elif localizer == "pernet_fold_B":
        conditions = ["pernet_fold_A/ON", "pernet_fold_A/OFF"]

    localizer_to_modality = {
        "faces": "vision",
        "bodies": "vision",
        "objects": "vision",
        "scenes": "vision",
        "vwfa": "vision",
        "tom": "cognitive",
        "theory_of_mind_text": "cognitive",
        "language": "cognitive",
        "fedorenko_words_nonwords": "cognitive",
        "multi_demand": "cognitive",
        "multiple_demand_text": "cognitive",
        "speech": "audio",
        "pernet_fold_A": "audio",
        "pernet_fold_B": "audio",
    }

    modality = localizer_to_modality[localizer]
    selectivity_mask, t_values = get_selectivity_mask(
        model_name,
        localizer, 
        modality, 
        anatomical_constraint, 
        top_k_pct, 
        p_value_threshold,
        odd_or_even=trial_type,
        fwhm_mm=fwhm_mm,
    )

    model, processor, _ = load_topo_omni(device=device)

    save_dir = f"{SAVE_DIR}/{model_name}/response_profiles"
    save_path = f"{save_dir}/{localizer}_response_profiles_top{top_k_pct}_{trial_type}_fwhm_mm={fwhm_mm}_anat={anatomical_constraint}.pkl"
    os.makedirs(save_dir, exist_ok=True)

    print(f"> Localizer: {localizer}")
    prompt = "Transcribe the audio of the following video."
    results = []
    for condition in conditions:

        if "pernet" in localizer:
            stimuli_path = f"{STIMULI_DIR}/{condition}"
            files = collect_audio_files(stimuli_path)
            print(f"> Extracting features for condition: {condition} with {len(files)} videos")
            cortical_sheets = extract_features_for_audio(
                model=model,
                processor=processor,
                audio_paths=files,
                batch_size=1,
            )
        else:
            stimuli_path = f"{STIMULI_DIR}/marvi_videos_condition_{condition}"
            files = collect_files(stimuli_path, odd_or_even="odd" if trial_type == "even" else "even")
            print(f"> Extracting features for condition: {condition} with {len(files)} videos")
            cortical_sheets = extract_features_for_video(
                model=model,
                processor=processor,
                video_paths=files,
                batch_size=1,
                prompt=prompt,
            )

        cortical_sheets = smoother(coords, cortical_sheets)
        
        responses = []
        for stimuli_idx, sheet in enumerate(cortical_sheets):
            sheet = sheet.reshape(smoother.height, smoother.width)
            sheet = np.rot90(sheet, k=1)  # rotate back to (H, W)
            magnitude = np.nanmean(sheet[selectivity_mask])
            responses.append(magnitude)
            for unit_idx, unit_response in enumerate(sheet[selectivity_mask].flatten()):
                results.append({
                    "unit_idx": unit_idx,
                    "unit_response": unit_response,
                    "localizer": localizer,
                    "condition": condition,
                    "mean": magnitude,
                    "stimuli_idx": stimuli_idx,
                    "top_k_pct": top_k_pct,
                    "p_value_threshold": p_value_threshold,
                })

        print(f"{condition}: {np.mean(responses):.4f}")

        dump_pickle(results, save_path)
