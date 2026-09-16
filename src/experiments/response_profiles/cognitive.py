import os
import yaml
import torch
import numpy as np
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.ndimage import binary_opening

from src.core.model_loading import load_topo_omni, unified_grid_coords, MODEL_TITLE
from src.experiments.selectivity.run import extract_features_for_texts, NeuronSmoothingConv
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

def get_selectivity_mask(model_name, category, modality, anatomical_constraint=False, top_k_pct=1, p_value_threshold=0.001, odd_or_even="even", fwhm_mm=None, significant_q=None):
    dirpath = f"{SAVE_DIR}/{model_name}"
    # Mirrors response_profiles/marvi.py. The placement below is reshape(W, H) + rot90,
    # which assumes the stats vector is ALREADY in grid raster order -- true only for a
    # SMOOTHED sweep, because run_selectivity projects through the coords before the
    # t-test. Naming the FWHM makes that the file we read; an unsmoothed sweep is in
    # per-unit order and would be scrambled by this reshape. fwhm_mm=None keeps the
    # original (unsmoothed) filename for backwards compatibility.
    half = f"{category}_{odd_or_even.upper()}"
    suffix = f"_fwhm{fwhm_mm}" if fwhm_mm is not None else ""
    filepath = f"{dirpath}/{half}/{half}_{odd_or_even}{suffix}_selectivity_stats.pkl"
    stats = read_pickle(filepath)

    t_values = stats['t']  # shape: (num_units, )
    t_values = t_values.reshape(W, H)
    t_values = np.rot90(t_values, k=1)

    # Bands from the layout, not literals. 144 / 256 are the `strip` constants
    # (thinker_rows=144, n_col//2=256); on tiled32 (thinker_rows=288, n_col=576) the 144
    # cut the thinker band in half and the 256 fell 32 columns short of the real 288
    # midline, leaking audio units into the vision band -- silently, since the array is
    # large enough that every one of these slices still succeeds. Same fix as
    # response_profiles/marvi.py.
    THK = SHEET.thinker_rows
    MID = SHEET.n_col // 2
    if anatomical_constraint:
        if modality in ["language", "cognitive"]:
            t_values[THK:, :] = -np.inf
            total_num_units = SHEET.thinker_rows * SHEET.n_col
        elif modality == "audio":
            t_values[:THK, :] = -np.inf
            t_values[THK:, :MID] = -np.inf
            total_num_units = SHEET.encoder_rows * MID
        elif modality == "vision":
            t_values[:THK, :] = -np.inf
            t_values[THK:, MID:] = -np.inf
            total_num_units = SHEET.encoder_rows * MID
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

    selectivity_mask = binary_opening(selectivity_mask, structure=np.ones((3, 3)))
    selectivity_mask = remove_small_components(selectivity_mask, min_size=20)
    return selectivity_mask, t_values

def collect_data(dirpath):
    data = []
    for filename in sorted(os.listdir(dirpath)):
        if filename.endswith(".txt"):
            filepath = os.path.join(dirpath, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                data = f.read().split("\n\n")

    data = [text.strip() for text in data if text.strip()]
    return data

def plot_cortical_sheet(sheet, selectivity_mask, title):
    plt.figure(figsize=(12, 6))
    sns.heatmap(sheet, mask=~selectivity_mask, cmap='viridis')
    plt.title(title)
    plt.axis('off')
    plt.savefig(f"{title}_cortical_sheet.png")

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Evaluate response profiles for different conditions.")
    parser.add_argument("--localizer", type=str, default="fedorenko_words_nonwords", help="Localizer condition to evaluate")
    parser.add_argument("--top_k_pct", type=int, default=10, help="Top k percent of units to keep")
    parser.add_argument("--odd_or_even", type=str, default="odd", help="Whether to use odd or even runs for evaluation")
    parser.add_argument("--fwhm_mm", type=float, default=8.0, help="FWHM in mm for smoothing the cortical sheet")
    parser.add_argument("--model_name", type=str, default=MODEL_TITLE,
                        help="run title under $SAVE_DIR to read stats from and write profiles to. "
                             "Was hardcoded to the strip model, so every other run's profiles "
                             "landed in (and were read from) topo-omni/.")
    parser.add_argument("--stats_fwhm_mm", type=float, default=None,
                        help="FWHM naming the selectivity-stats file to read, e.g. 3.0 for "
                             "'<cat>_<half>_fwhm3.0_selectivity_stats.pkl'. Required for a "
                             "smoothed sweep; omit for the legacy unsmoothed filename.")
    parser.add_argument("--significant_q", type=float, default=None,
                        help="if set, keep only top-k units that are ALSO FDR-significant "
                             "at q < this value with t > 0. 0.05 matches the alpha the "
                             "selectivity sweep was run at. Omit for a pure top-k mask.")
    parser.add_argument("--coords", type=str, default=None,
                        help="coords.npy the run was TRAINED with, used to smooth the response "
                             "sheets. Without it the identity grid is assumed, which is only "
                             "correct for a run whose units were never shuffled.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="checkpoint dir or HF id (default: $TOPO_OMNI_MODEL)")

    args = parser.parse_args()
    localizer = args.localizer

    p_value_threshold = 0.001
    top_k_pct = args.top_k_pct  # top k percent of units to keep

    anatomical_constraint = True
    H, W = SHEET.n_row, SHEET.n_col  # unified sheet size

    trial_type = args.odd_or_even  # "odd" or "even"

    fwhm_mm = args.fwhm_mm
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

    localizer_to_modality = {
        "tom": "cognitive",
        "theory_of_mind_text": "cognitive",
        "language_text": "cognitive",
        "fedorenko_words_nonwords": "cognitive",
        "multi_demand": "cognitive",
        "multiple_demand_text": "cognitive",
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
        fwhm_mm=args.stats_fwhm_mm,
        significant_q=args.significant_q,
    )

    # Stable unit identity: the flat position of each masked unit in the (H, W) sheet.
    # enumerate() below used to give a per-half POSITIONAL counter, so "unit 7" named a
    # different unit in the even and odd halves and the paired-across-units test compared
    # unrelated units. Boolean indexing returns elements in C order, which is exactly the
    # order flatnonzero lists them in, so ids and responses stay aligned.
    mask_unit_ids = np.flatnonzero(selectivity_mask)


    model, processor, _ = load_topo_omni(model=args.checkpoint, device=device)
    print(f"> Model: {args.checkpoint or os.getenv('TOPO_OMNI_MODEL', '<default>')}")

    save_dir = f"{SAVE_DIR}/{model_name}/response_profiles"
    # See marvi.py: the gate changes which units are measured, so it must name the output.
    sig_tag = f"_sig={args.significant_q}" if args.significant_q else ""
    save_path = f"{save_dir}/{localizer}_response_profiles_top{top_k_pct}_{trial_type}{sig_tag}.pkl"
    os.makedirs(save_dir, exist_ok=True)

    print(f"> Localizer: {localizer}")
    prompt = "Transcribe the audio of the following video."
    results = []
    for condition in ["ON", "OFF"]:
        other_trial_type = "even" if trial_type == "odd" else "odd"
        dirname = f"{localizer}_{other_trial_type.upper()}/{condition}"
        stimuli_path = f"{STIMULI_DIR}/{dirname}"
        data = collect_data(stimuli_path)
        print(f"> Extracting features for condition: {condition} with {len(data)} texts")

        cortical_sheets = extract_features_for_texts(
            model=model,
            processor=processor,
            texts=data,
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
