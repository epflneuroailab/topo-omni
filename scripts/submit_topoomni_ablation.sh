#!/bin/bash
# Run the ablation grid in PARALLEL — one runai job per (stimuli, localizer) cell.
#
# Each cell is an independent forward pass over one stimulus set with one selectivity mask
# applied, and writes its own results/<run>/ablation/*.json, so they fan out cleanly to one
# single-GPU job each. Run sequentially instead and you pay a model load from NFS per cell:
# 20 cells (4x4 matrix + 4 no-ablation baselines) back to back.
#
# PREREQUISITE — seed the mean-activation cache first, with ONE run:
#   python -m src.experiments.ablation.run --stimuli faces \
#       --model_name $MODEL_NAME --stats_fwhm_mm 3.0 --coords <coords> --checkpoint <ckpt>
# cache_mean_activations() is guarded only by an os.path.exists check on
# results/<run>/mean_activations/faces_mean_activations.pt, so N jobs launched against a
# cold cache all miss, all recompute, and all torch.save to the same paths at once.
#
# Usage:
#   ./scripts/submit_topoomni_ablation.sh                    # 4x4 matrix + 4 baselines (20 jobs)
#   ./scripts/submit_topoomni_ablation.sh faces,scenes       # subset of stimuli (still all localizers)
#   PERCENTAGE=20 ./scripts/submit_topoomni_ablation.sh      # a different ablation fraction
#   BASELINES_ONLY=1 ./scripts/submit_topoomni_ablation.sh   # just the no-ablation runs
#
# Monitor:  runai list | grep topoomni-abl-
# Logs:     runai logs topoomni-abl-<stim>-<loc> -f
# Clean up: for j in $(runai list 2>/dev/null | awk '/topoomni-abl-/{print $1}'); do runai delete job $j; done

arg_stimuli=${1:-all}
arg_node_pool=${NODE_POOL:-default}          # default pool == A100
MY_IMAGE=bkhmsi/epfl:cortexlm

CONDA_ENV=topoomni
REPO_DIR=topo-omni-epfl

MODEL_NAME=${MODEL_NAME:-topo-omni-tiled32-4608}
FWHM=${FWHM:-3.0}
MODE=${MODE:-top_p}
PERCENTAGE=${PERCENTAGE:-10}
CHECKPOINT=${CHECKPOINT:-artifacts/ckpts/${MODEL_NAME}/checkpoint-730}
COORDS=${COORDS:-src/neighborhoods/model=qwen2-5omni-3b-tiled32-4608-v7_radius=dynamic_coords=42/coords.npy}

# ablation/run.py reads DEFAULT_SHEET_LAYOUT at IMPORT time, and its load_dotenv() call
# comes AFTER that import — so the repo .env cannot set this in time. It has to be in the
# container environment before python starts, hence -e below. Without it the tiled32 stats
# (460,800 units) are read against the default strip sheet and the reshape throws.
SHEET_LAYOUT=${SHEET_LAYOUT:-tiled32}

ALL_STIMULI=(faces bodies scenes objects)
ALL_LOCALIZERS=(faces bodies scenes objects)

if [ "$arg_stimuli" = "all" ]; then
    STIMULI=("${ALL_STIMULI[@]}")
else
    IFS=',' read -r -a STIMULI <<< "$arg_stimuli"
fi

echo "== submitting ablation jobs =="
echo "   model      : $MODEL_NAME"
echo "   checkpoint : $CHECKPOINT"
echo "   node pool  : $arg_node_pool (1 GPU per job)"
echo "   mode       : $MODE @ ${PERCENTAGE}%  | stats fwhm $FWHM | layout $SHEET_LAYOUT"
echo

submit_one() {
    local job_name=$1; shift
    echo "-- $job_name"
    runai submit \
        --name "$job_name" \
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
        -m src.experiments.ablation.run \
        --model_name "$MODEL_NAME" \
        --stats_fwhm_mm "$FWHM" \
        --coords "$COORDS" \
        --checkpoint "$CHECKPOINT" \
        "$@"
}

n=0
# --- no-ablation baselines: one per stimulus set -------------------------------------
for s in "${STIMULI[@]}"; do
    submit_one "topoomni-abl-${s}-none" --stimuli "$s"
    n=$((n+1))
done

# --- the localizer x stimuli matrix ---------------------------------------------------
if [ -z "$BASELINES_ONLY" ]; then
    for l in "${ALL_LOCALIZERS[@]}"; do
        for s in "${STIMULI[@]}"; do
            submit_one "topoomni-abl-${s}-${l}" \
                --stimuli "$s" --ablate --localizer "$l" \
                --mode "$MODE" --percentage "$PERCENTAGE"
            n=$((n+1))
        done
    done
fi

echo
echo "== submitted $n job(s). watch with: runai list | grep topoomni-abl- =="
