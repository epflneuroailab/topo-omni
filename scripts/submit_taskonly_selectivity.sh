#!/bin/bash
# Selectivity maps for the TASK-ONLY (non-topographic) tiled32 control, one runai job per
# condition. These back the Appendix E topo-vs-non-topo island Moran's I comparison, which
# has no tiled32 counterpart yet -- the published one is the 309x512 layout.
#
# CHECKPOINT 292, NOT 365. checkpoint-365's third safetensors shard is truncated by
# 195,860,016 bytes (an interrupted save), so it cannot be loaded at all. 292 is intact and
# verified. NOTE the control is therefore 4 epochs against the topo run's 10 (checkpoint-730)
# -- state that mismatch wherever the comparison is reported.
#
# Both checkpoints also shipped without the processor/tokenizer configs and without
# model.safetensors.index.json; those were copied/rebuilt in place before this was written.
#
# Usage:   ./submit_taskonly_selectivity.sh [cond1,cond2,...]
# Monitor: runai list | grep topoomni-sel-
# Logs:    runai logs topoomni-sel-<cond> -f

arg_conditions=${1:-all}
arg_node_pool=${NODE_POOL:-default}
MY_IMAGE=bkhmsi/epfl:cortexlm

CONDA_ENV=topoomni
REPO_DIR=topo-omni-epfl

MODEL_NAME=${MODEL_NAME:-topo-omni-tiled32-4608-taskonly}
CHECKPOINT=${CHECKPOINT:-checkpoint-292}
FWHM=${FWHM:-3.0}
SHEET_LAYOUT=${SHEET_LAYOUT:-tiled32}

ALL_CONDITIONS=(faces scenes bodies vwfa objects speech vocals \
                language_text_ALL multiple_demand_text_ALL theory_of_mind_text_ALL)

if [ "$arg_conditions" = "all" ]; then
    CONDITIONS=("${ALL_CONDITIONS[@]}")
else
    IFS=',' read -r -a CONDITIONS <<< "$arg_conditions"
fi

echo "== submitting task-only selectivity jobs =="
echo "   model      : $MODEL_NAME"
echo "   checkpoint : $CHECKPOINT"
echo "   fwhm       : $FWHM | layout $SHEET_LAYOUT"
echo

n=0
for c in "${CONDITIONS[@]}"; do
    # job names cannot hold underscores
    job="topoomni-sel-$(echo "$c" | tr '_' '-' | tr '[:upper:]' '[:lower:]')"
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
        --model_name "$MODEL_NAME" \
        --checkpoint "$CHECKPOINT" \
        --fwhm "$FWHM" \
        --conditions "$c"
    n=$((n+1))
done

echo
echo "== submitted $n job(s). watch with: runai list | grep topoomni-sel- =="
