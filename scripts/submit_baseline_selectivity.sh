#!/bin/bash
# Selectivity maps for the UNTRAINED Qwen2.5-Omni-3B baseline on the tiled32 sheet.
#
# This is the "Qwen2.5-3B" column of Table 1: the pretrained backbone with IDENTITY
# (pass-through) cortical adaptors, i.e. no topographic reorganisation at all. It is a
# different control from topo-omni-tiled32-4608-taskonly, which is trained on the task loss
# and therefore has adaptors that moved.
#
# WHY IT IS NEEDED. The NSD feature extraction masks units by each model's OWN selectivity
# stats, so the baseline column of the brain-alignment table cannot be computed until this
# sweep exists. faces/scenes/bodies/vwfa are the four the Brain-Score benchmark scores; the
# remaining six are included so the baseline can also serve as a control elsewhere.
#
# The baseline has no run directory, so --config_from supplies the sheet layout and
# position-dir; it must name the run this is a control FOR, or the geometry will not match.
# --model_name only names the output directory. There is no --checkpoint: load_topo_omni's
# baseline flag resolves to Qwen/Qwen2.5-Omni-3B and ignores any model override.
#
# NOTE the stock Qwen config carries no `cortical_dim`, so the adaptors would default to
# square (1280/2048) and `_band_from_layers` would abort with "sheet layout tiled gives 4608
# units per encoder slot, but the vision adaptor is 1280 wide". load_topo_omni now fills the
# width in from the active sheet layout when the config lacks one; TOPO_SHEET_LAYOUT below is
# therefore load-bearing, not just a formality.
#
# Usage:   ./submit_baseline_selectivity.sh [cond1,cond2,...]
# Monitor: runai list | grep topoomni-base-sel-
# Logs:    runai logs topoomni-base-sel-<cond> -f

arg_conditions=${1:-all}
arg_node_pool=${NODE_POOL:-default}
MY_IMAGE=bkhmsi/epfl:cortexlm

CONDA_ENV=topoomni
REPO_DIR=topo-omni-epfl

MODEL_NAME=${MODEL_NAME:-qwen-omni-base-tiled32}
CONFIG_FROM=${CONFIG_FROM:-topo-omni-tiled32-4608}
FWHM=${FWHM:-3.0}
SHEET_LAYOUT=${SHEET_LAYOUT:-tiled32}

ALL_CONDITIONS=(faces scenes bodies vwfa objects speech vocals \
                language_text_ALL multiple_demand_text_ALL theory_of_mind_text_ALL)

if [ "$arg_conditions" = "all" ]; then
    CONDITIONS=("${ALL_CONDITIONS[@]}")
else
    IFS=',' read -r -a CONDITIONS <<< "$arg_conditions"
fi

echo "== submitting baseline (untrained Qwen) selectivity jobs =="
echo "   output as : $MODEL_NAME"
echo "   geometry  : $CONFIG_FROM (layout $SHEET_LAYOUT)"
echo "   fwhm      : $FWHM"
echo

n=0
for c in "${CONDITIONS[@]}"; do
    job="topoomni-base-sel-$(echo "$c" | tr '_' '-' | tr '[:upper:]' '[:lower:]')"
    echo "-- $job ($c)"
    runai submit \
        --name "$job" \
        -i $MY_IMAGE \
        --gpu 1 \
        --cpu 8 \
        --large-shm \
        --node-pools "$arg_node_pool" \
        -e "TOPO_SHEET_LAYOUT=$SHEET_LAYOUT" \
        --pvc nlp-scratch:/mnt/u14157_ic_nlp_001_files_nfs/nlpdata1 \
        -- \
        $REPO_DIR \
        $CONDA_ENV \
        scripts/run_selectivity.py \
        --baseline \
        --model_name "$MODEL_NAME" \
        --config_from "$CONFIG_FROM" \
        --fwhm "$FWHM" \
        --conditions "$c"
    n=$((n+1))
done

echo
echo "== submitted $n job(s). watch with: runai list | grep topoomni-base-sel- =="
