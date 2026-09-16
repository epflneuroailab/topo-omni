import os
import uuid
import yaml
import argparse
import subprocess
from glob import glob

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
    # Unique per WRITE, not per (condition, pid). Several of these run at once against the
    # same NFS checkout (one runai job per condition/half), so a shared filename lets one
    # job's config be evaluated under another job's name and makes os.remove crash
    # the rest.
    #
    # os.getpid() is NOT enough: each job is its own container, so the pid is frequently
    # IDENTICAL across jobs (42 in this image). Two jobs differing only by --odd_or_even
    # share a category_name, so `<category>_<pid>` collided and both halves evaluated
    # whichever config was written last -- silently, since the run is perfectly valid, just
    # not the one that was asked for. uuid4 is per-call and cannot collide.
    tag = f"{config['run']['category_name']}_{uuid.uuid4().hex[:12]}"
    condition_config_path = f"src/configs/eval/condition_tmp_{tag}.yml"
    write_config(config, condition_config_path)
    try:
        subprocess.run(["python", "-m", "src.experiments.selectivity.run",
                        "--config", condition_config_path], check=True)
    finally:
        if os.path.exists(condition_config_path):
            os.remove(condition_config_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="topo-omni-frontend-invr")
    parser.add_argument("--config_path", default='src/configs/eval/marvi.yml')
    parser.add_argument("--odd_or_even", default=None, choices=["odd", "even", None])
    parser.add_argument("--no-smooth", action="store_true", help="smooth")
    parser.add_argument("--fwhm", type=float, default=3.0,
                        help="smoothing FWHM in mm. Sets stats.fwhm_mm and, through it, the "
                             "'_fwhm<f>' suffix run_selectivity appends to category_name -- "
                             "so different values write side by side and never collide. The "
                             "feature cache is keyed WITHOUT that suffix, so re-running at a "
                             "new FWHM reuses it and never touches the GPU, as long as "
                             "--overwrite is not passed.")
    parser.add_argument("--overwrite", action="store_true", help="Whether to overwrite existing results")
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint dir, or a name like 'checkpoint-146' under the "
                             "run dir. Default: the highest-numbered checkpoint-* there.")
    parser.add_argument("--conditions", default=None,
                        help="comma-separated subset of conditions, e.g. 'faces'. "
                             "Default: every MARVi / text / pernet condition.")
    args = parser.parse_args()

    model_name = args.model_name
    config_path = args.config_path
    no_smooth = args.no_smooth
    odd_or_even = args.odd_or_even

    config = load_config(config_path)

    run_config = load_config(os.path.join(CKPT_DIR, model_name, "config.yml"))
    
    marvi_conditions = [
        "faces",
        "scenes",
        "speech",
        "bodies",
        "vwfa",
        "objects",
        # "theory_of_mind",
        # "multi_demand",
        # "language",
    ]

    text_conditions =[
        "language_text_ALL",
        "multiple_demand_text_ALL",
        "theory_of_mind_text_ALL",
        # "fedorenko_words_nonwords",

        # Split halves, for the cross-validated response profiles: select units on one
        # half, measure responses on the other (response_profiles/cognitive.py).
        "language_text_EVEN",
        "language_text_ODD",
        "multiple_demand_text_EVEN",
        "multiple_demand_text_ODD",
        "theory_of_mind_text_EVEN",
        "theory_of_mind_text_ODD",
    ]

    pernet_conditions = [
        "vocals",
        # The two CV halves of the vocal/non-vocal set: localize on one fold, measure
        # responses on the other (response_profiles/marvi.py picks the opposite fold).
        # plot_pernet.py reads both back together.
        "pernet_fold_A",
        "pernet_fold_B",
    ]
        
    condition_config = config.copy()
    condition_config["data"]["odd_or_even"] = odd_or_even
    condition_config["stats"]["alpha"] = 0.05
    condition_config["stats"]["topk_pct"] = 0
    condition_config["stats"]["fwhm_mm"] = args.fwhm if not no_smooth else 0.0
    condition_config["stats"]["smooth"] = not no_smooth
    # run_selectivity.py resolves the weights from `cfg.model.model`, falling back to
    # $TOPO_OMNI_MODEL -- it never reads `run_dir`. Setting run_dir alone therefore
    # evaluated whatever $TOPO_OMNI_MODEL pointed at, i.e. a DIFFERENT run, and silently:
    # the only thing keyed to `model_name` was the OUTPUT directory, so the results landed
    # under the right name while describing the wrong model. Name the checkpoint here.
    run_dir = f"{CKPT_DIR}/{model_name}"
    if args.checkpoint:
        ckpt = (args.checkpoint if os.path.isabs(args.checkpoint)
                or os.path.exists(args.checkpoint)
                else f"{run_dir}/{args.checkpoint}")
    else:
        found = sorted(glob(f"{run_dir}/checkpoint-*"),
                       key=lambda p: int(p.rsplit("-", 1)[1]))
        # no checkpoint-* means the weights sit at the top of the run dir
        ckpt = found[-1] if found else run_dir
    if not os.path.exists(ckpt):
        raise SystemExit(f"no such checkpoint: {ckpt}")

    condition_config["model"]["run_dir"] = run_dir
    condition_config["model"]["model"] = ckpt
    print(f"> Model: {ckpt}")

    # The sheet layout is resolved at import time in spatial_utils and fixes the adaptor
    # width, the unit count and whether a front-end slot exists. Evaluating a tiled32 run
    # under the default strip layout mismatches the coords and scrambles every map, so
    # take it from the config the run was trained with.
    layout = run_config.get("sheet-layout")
    if layout:
        os.environ["TOPO_SHEET_LAYOUT"] = layout
        print(f"> TOPO_SHEET_LAYOUT={layout} (from {model_name}/config.yml)")
    condition_config["model"]["neighborhood_dir"] = run_config["topo-params"]["position-dir"]
    condition_config["model"]["run_title"] = model_name
    condition_config["run"]["run_title"] = model_name
    condition_config["run"]["output_root"] = "results"
    condition_config["run"]["do_pretrained"] = False
    condition_config["run"]["overwrite"] = args.overwrite

    conditions = marvi_conditions + text_conditions + pernet_conditions
    if args.conditions:
        want = [c.strip() for c in args.conditions.split(",") if c.strip()]
        unknown = [c for c in want if c not in conditions]
        if unknown:
            raise SystemExit(f"unknown condition(s) {unknown}; known: {conditions}")
        conditions = want

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
            # Each fold must read ITS OWN half. pernet_vocal_nonvocal is the UNION of both
            # folds (20 ON / 20 OFF vs 10 / 10 each), so routing every pernet condition
            # there would hand fold_A and fold_B identical stimuli -- identical stats, and
            # the cross-validation collapses into measuring on the same clips it selected
            # from. Only the pooled "vocals" condition wants the union.
            sub = condition if condition.startswith("pernet_fold_") else "pernet_vocal_nonvocal"
            condition_config['data']['stimuli_root'] = f"{STIMULI_DIR}/{sub}"
            condition_config["data"]["mode"] = "audio"
     
        eval_condition(condition_config)



