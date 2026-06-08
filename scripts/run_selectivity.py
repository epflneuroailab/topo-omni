import os
import yaml
import argparse
import subprocess

from dotenv import load_dotenv
load_dotenv()

STIMULI_DIR = os.getenv("STIMULI_DIR", "./stimuli")
CKPT_DIR = os.getenv("CKPT_DIR", "./ckpts")

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def write_config(config, config_path):
    """Write configuration to a YAML file."""
    with open(config_path, 'w') as file:
        yaml.dump(config, file)

def eval_condition(config):
    condition_config_path = f"src/configs/eval_condition_tmp.yml"
    write_config(config, condition_config_path)
    subprocess.run(["python", "-m", "src.eval.run.run_selectivity", "--config", condition_config_path], check=True)
    os.remove(condition_config_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="qwen2_5_3b_spatial_task_final_7")
    parser.add_argument("--config_path", default='src/configs/eval_marvi.yml')
    parser.add_argument("--odd_or_even", default=None, choices=["odd", "even", None])
    parser.add_argument("--no-smooth", action="store_true", help="smooth")
    parser.add_argument("--overwrite", action="store_true", help="Whether to overwrite existing results")
    args = parser.parse_args()

    model_name = args.model_name
    config_path = args.config_path
    no_smooth = args.no_smooth
    odd_or_even = args.odd_or_even

    config = load_config(config_path)

    run_config = load_config(os.path.join(CKPT_DIR, model_name, "config.yml"))
    
    marvi_conditions = [
        # "faces",
        # "scenes",
        # "speech",
        # "bodies",
        # "vwfa",
        # "objects",
        # "theory_of_mind",
        # "multi_demand",
        # "language",
    ]

    text_conditions =[
        # "language_text_ALL",
        # "multiple_demand_text_ALL",
        # "theory_of_mind_text_ALL",
        # "fedorenko_words_nonwords",

        # "language_text_EVEN",
        # "language_text_ODD",
        # "multiple_demand_text_EVEN",
        # "multiple_demand_text_ODD",
        # "theory_of_mind_text_EVEN",
        # "theory_of_mind_text_ODD",
    ]

    pernet_conditions = [
        # "vocals",
        "pernet_fold_A",
        "pernet_fold_B",
    ]
        
    condition_config = config.copy()
    condition_config["data"]["odd_or_even"] = odd_or_even
    condition_config["stats"]["alpha"] = 0.05
    condition_config["stats"]["topk_pct"] = 0
    condition_config["stats"]["fwhm_mm"] = 4.0 if not no_smooth else 0.0
    condition_config["stats"]["smooth"] = not no_smooth
    condition_config["model"]["run_dir"] = f"{CKPT_DIR}/{model_name}"
    condition_config["model"]["neighborhood_dir"] = run_config["topo-params"]["position-dir"]
    condition_config["model"]["run_title"] = model_name
    condition_config["run"]["run_title"] = model_name
    condition_config["run"]["output_root"] = "results"
    condition_config["run"]["do_pretrained"] = False
    condition_config["run"]["overwrite"] = args.overwrite

    conditions = marvi_conditions + text_conditions + pernet_conditions

    for condition in conditions:
        print(f"> Running MARVi evaluation for condition: {condition}")

        condition_config['run']['category_name'] = condition

        if condition in marvi_conditions:
            condition_config['data']['stimuli_root'] = f"{STIMULI_DIR}/marvi_videos_{condition}"
            condition_config["data"]["mode"] = "video"
        elif condition in text_conditions:
            condition_config["data"]["odd_or_even"] = "even" if "EVEN" in condition else ("odd" if "ODD" in condition else None)
            condition_config['data']['stimuli_root'] = f"{STIMULI_DIR}/{condition}"
            condition_config["data"]["mode"] = "text"
        elif condition in pernet_conditions:
            condition_config['data']['stimuli_root'] = f"{STIMULI_DIR}/pernet_vocal_nonvocal"
            condition_config["data"]["mode"] = "audio"
     
        eval_condition(condition_config)



