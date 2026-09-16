import os
import yaml
import torch
import numpy as np
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.ndimage import binary_opening
from src.core.model_loading import load_topo_omni, unified_grid_coords
from src.experiments.selectivity.run import extract_features_for_video, extract_features_for_audio, place_on_grid, NeuronSmoothingConv
from src.experiments.selectivity.plot import remove_small_components
from src.models.spatial_utils import DEFAULT_SHEET_LAYOUT as SHEET

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
    fwhm_mm=4.0,
    significant_q=None,
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

    # Bands from the layout. The literals 144 / 256 described `strip` only; the
    # total_num_units lines right beside them already used SHEET, so the two halves of
    # this block disagreed on every other layout. On tiled32 (thinker_rows=288,
    # n_col=576) the 144 cut the thinker band in half and the 256 split fell 32 columns
    # short of the real 288 midline, leaking audio units into the vision band.
    THK = SHEET.thinker_rows
    MID = SHEET.n_col // 2
    if anatomical_constraint:
        if modality in ["language", "cognitive"]:
            t_values[THK:, :] = -np.inf
            total_num_units = THK * SHEET.n_col
        elif modality == "audio":
            t_values[:THK, :] = -np.inf
            t_values[THK:, :MID] = -np.inf
            total_num_units = SHEET.encoder_rows * MID
        elif modality == "vision":
            t_values[:THK, :] = -np.inf
            t_values[THK:, MID:] = -np.inf
            total_num_units = SHEET.encoder_rows * MID
        else:
            raise SystemExit(f"no anatomical band for modality {modality!r}")
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

    # Optional significance gate: keep only the top-k units that are ALSO FDR-significant
    # with a positive effect. `significant_q` is compared against the same q the selectivity
    # sweep produced, so q=0.05 reproduces the criterion the sweep was run at (summary.json
    # alpha_counts key "0.05", and the alpha5.0e-02 selectivity maps).
    #
    # Orientation matters here: the top-k branch builds its mask in the flattened (H, W)
    # frame that t_values was rotated into, while stats["q"] is still in the raw per-unit
    # order -- so the gate needs the same reshape(W, H) + rot90 the significance branch
    # above uses before the two can be intersected.
    if significant_q is not None:
        _sig = (stats["q"] < significant_q) & (stats["t"] > 0)
        _sig = np.rot90(_sig.reshape(W, H), k=1)
        _before = int(selectivity_mask.sum())
        selectivity_mask = selectivity_mask & _sig
        print(f"> Significance gate q<{significant_q}: {_before} -> "
              f"{int(selectivity_mask.sum())} units")
        if not selectivity_mask.any():
            raise SystemExit(
                f"empty mask: no unit in the top {top_k_pct}% of {category!r} "
                f"({odd_or_even}) is significant at q<{significant_q}")

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
    parser.add_argument("--significant_q", type=float, default=None,
                        help="if set, keep only top-k units that are ALSO FDR-significant "
                             "at q < this value with t > 0. 0.05 matches the alpha the "
                             "selectivity sweep was run at. Omit for a pure top-k mask.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="checkpoint dir to evaluate. WITHOUT this, load_topo_omni "
                             "falls back to $TOPO_OMNI_MODEL -- a DIFFERENT run -- while "
                             "--model_name only picks the stats and the output path, so "
                             "the results land under the right name describing the wrong "
                             "model. Same trap scripts/run_selectivity.py documents.")
    parser.add_argument("--coords", type=str, default=None,
                        help="coords.npy the run was TRAINED with. Required in practice: "
                             "build_sheet_coordinates shuffles every unit within its "
                             "layer block, so the identity grid from unified_grid_coords() "
                             "puts only 94 of 460,800 tiled32 units (0.02%%) back where "
                             "they belong. The selectivity mask is built from stats that "
                             "WERE projected with the training coords, so mixing the two "
                             "samples the wrong units entirely.")

    args = parser.parse_args()
    localizer = args.localizer

    p_value_threshold = 0.001
    top_k_pct = args.top_k_pct  # top k percent of units to keep

    fwhm_mm = args.fwhm_mm
    anatomical_constraint = args.anatomical_constraint == "true"
    trial_type = args.odd_or_even  # "odd" or "even"

    H, W = SHEET.n_row, SHEET.n_col  # unified sheet size
    resolution_mm = 1.0
    smoother = NeuronSmoothingConv(fwhm_mm=fwhm_mm, resolution_mm=resolution_mm)

    model_name = args.model_name
    if args.coords:
        coords = np.load(args.coords)
        print(f"> Unit coordinates: {args.coords} {coords.shape}")
    else:
        coords = unified_grid_coords()
        print("> WARNING: no --coords given; using the identity grid from "
              "unified_grid_coords(). This is only correct for a run whose units were "
              "never shuffled by build_sheet_coordinates.")

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
        significant_q=args.significant_q,
    )

    # Stable unit identity: the flat position of each masked unit in the (H, W) sheet.
    # enumerate() below used to give a per-half POSITIONAL counter, so "unit 7" named a
    # different unit in the even and odd halves and the paired-across-units test compared
    # unrelated units. Boolean indexing returns elements in C order, which is exactly the
    # order flatnonzero lists them in, so ids and responses stay aligned.
    mask_unit_ids = np.flatnonzero(selectivity_mask)


    # Lazy, so a fully cached run never loads the model at all.
    _loaded = {}
    def get_model():
        if not _loaded:
            m, pr, _ = load_topo_omni(model=args.checkpoint, device=device)
            _loaded["m"], _loaded["pr"] = m, pr
            print(f"> Model: {args.checkpoint or os.getenv('TOPO_OMNI_MODEL', '<default>')}")
        return _loaded["m"], _loaded["pr"]

    save_dir = f"{SAVE_DIR}/{model_name}/response_profiles"
    # The significance gate changes which units are measured, so it must name the output
    # too -- otherwise a gated run silently overwrites the ungated one at the same top-k.
    sig_tag = f"_sig={args.significant_q}" if args.significant_q else ""
    save_path = f"{save_dir}/{localizer}_response_profiles_top{top_k_pct}_{trial_type}_fwhm_mm={fwhm_mm}_anat={anatomical_constraint}{sig_tag}.pkl"
    os.makedirs(save_dir, exist_ok=True)
    # Feature cache. What gets extracted depends ONLY on (condition, which video split),
    # never on the localizer or the FWHM -- the localizer enters through the mask and the
    # FWHM through the smoother, both applied AFTER extraction. Without this, a sweep of
    # 5 localizers x 2 FWHM re-runs the identical clips 10 times per split. Raw
    # (unsmoothed) sheets are cached, so one cache serves every FWHM.
    feat_cache_dir = os.path.join(save_dir, "cache")
    os.makedirs(feat_cache_dir, exist_ok=True)

    print(f"> Localizer: {localizer}")
    prompt = "Transcribe the audio of the following video."
    results = []
    for condition in conditions:

        stim_split = "all" if "pernet" in localizer else ("odd" if trial_type == "even" else "even")
        cache_path = os.path.join(
            feat_cache_dir,
            f"{condition.replace(os.sep, chr(95))}_stim={stim_split}_features.pkl")

        if os.path.exists(cache_path):
            cortical_sheets = read_pickle(cache_path)
            print(f"> Cached features for condition: {condition} "
                  f"({len(cortical_sheets)} stimuli, {stim_split} runs)")
        else:
            model, processor = get_model()
            if "pernet" in localizer:
                stimuli_path = f"{STIMULI_DIR}/{condition}"
                files = collect_audio_files(stimuli_path)
                print(f"> Extracting features for condition: {condition} with {len(files)} files")
                cortical_sheets = extract_features_for_audio(
                    model=model,
                    processor=processor,
                    audio_paths=files,
                    batch_size=1,
                )
            else:
                stimuli_path = f"{STIMULI_DIR}/marvi_videos_condition_{condition}"
                files = collect_files(stimuli_path, odd_or_even=stim_split)
                print(f"> Extracting features for condition: {condition} with {len(files)} "
                      f"videos ({stim_split} runs; localizer used the {trial_type} runs)")
                cortical_sheets = extract_features_for_video(
                    model=model,
                    processor=processor,
                    video_paths=files,
                    batch_size=1,
                    prompt=prompt,
                )
            dump_pickle(cortical_sheets, cache_path)

        cortical_sheets = smoother(coords, cortical_sheets)
        
        responses = []
        for stimuli_idx, sheet in enumerate(cortical_sheets):
            sheet = sheet.reshape(smoother.height, smoother.width)
            sheet = np.rot90(sheet, k=1)  # rotate back to (H, W)
            magnitude = np.nanmean(sheet[selectivity_mask])
            responses.append(magnitude)
            for unit_idx, unit_response in zip(mask_unit_ids, sheet[selectivity_mask].flatten()):
                results.append({
                    "unit_idx": int(unit_idx),
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
