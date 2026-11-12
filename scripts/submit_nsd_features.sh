#!/bin/bash
# Extract NSD cortical-sheet features for the brain-alignment analysis, sharded over
# save-batches so the 73,000 images fan out to several single-GPU jobs.
#
# WHAT THIS FEEDS. The output under results/<model>/nsd_features/ is the input to the
# external Brain-Score run (benchmark `nsd_func1pt8mm_individualROIs`) that produces
# nsd_topo_omni_results_merged.csv, which in turn fills the top half of Table 1. The
# feature keys are "<category>_top<k>", and the existing CSV scores exactly
# faces/bodies/scenes/vwfa x top-1/5/10 against 12 ventral ROIs -- hence the defaults
# below. `objects` and `speech` are NOT scored by that benchmark and are left out; add
# them to CATEGORIES if that changes.
#
# SHARDING. Each save-batch writes one self-contained hdf5 and reads nothing from its
# neighbours, so disjoint ranges run safely in parallel. They must stay DISJOINT: the
# script's skip-if-exists resume check is not a lock, so two jobs handed the same batch
# will both start it and race on the write. A job that dies can simply be resubmitted --
# it skips whatever already landed.
#
# 286 save-batches of 256 images. With NUM_SHARDS=8 that is 36 batches (~9,200 images) each.
#
# Usage:
#   ./submit_nsd_features.sh                       # 8 shards, topo-omni-tiled32-4608
#   NUM_SHARDS=4 ./submit_nsd_features.sh          # fewer, larger shards
#   MODEL_NAME=... CHECKPOINT=... ./submit_nsd_features.sh
#
# Monitor: runai list | grep topoomni-nsd-
# Logs:    runai logs topoomni-nsd-0 -f
# Progress: ls results/<model>/nsd_features/*.hdf5 | wc -l   (expect 286)

arg_node_pool=${NODE_POOL:-default}
MY_IMAGE=bkhmsi/epfl:cortexlm

CONDA_ENV=topoomni
REPO_DIR=topo-omni-epfl

# BASELINE=1 extracts with the UNTRAINED Qwen2.5-Omni-3B (identity cortical adaptors)
# instead of a checkpoint -- the "Qwen2.5-3B" column of Table 1. The unit masks still come
# from MODEL_NAME's own selectivity stats, so point MODEL_NAME at the baseline sweep
# (qwen-omni-base-tiled32), not at the topo run.
BASELINE=${BASELINE:-}
if [ -n "$BASELINE" ]; then
    MODEL_NAME=${MODEL_NAME:-qwen-omni-base-tiled32}
    CHECKPOINT=""
    JOB_PREFIX=${JOB_PREFIX:-topoomni-nsd-base}
else
    MODEL_NAME=${MODEL_NAME:-topo-omni-tiled32-4608}
    CHECKPOINT=${CHECKPOINT:-artifacts/ckpts/topo-omni-tiled32-4608}
    JOB_PREFIX=${JOB_PREFIX:-topoomni-nsd}
fi
STATS_FWHM=${STATS_FWHM:-3.0}
CATEGORIES=${CATEGORIES:-faces,scenes,bodies,vwfa}
TOP_K_PCTS=${TOP_K_PCTS:-1,5,10}
SAVE_BATCH=${SAVE_BATCH:-256}
NUM_SHARDS=${NUM_SHARDS:-8}

# extract_features.py reads DEFAULT_SHEET_LAYOUT at import time, before load_dotenv(), so
# the repo .env cannot set this in time. It fixes the adaptor width and the unit count --
# get it wrong and the tiled32 selectivity stats (460,800 units) are read against the
# default strip sheet and every mask is scrambled.
SHEET_LAYOUT=${SHEET_LAYOUT:-tiled32}

# 73,000 images / SAVE_BATCH, rounded up
NUM_IMAGES=${NUM_IMAGES:-73000}
NUM_BATCHES=$(( (NUM_IMAGES + SAVE_BATCH - 1) / SAVE_BATCH ))
PER_SHARD=$(( (NUM_BATCHES + NUM_SHARDS - 1) / NUM_SHARDS ))

echo "== submitting NSD feature extraction =="
echo "   model      : $MODEL_NAME"
echo "   weights    : ${CHECKPOINT:-Qwen/Qwen2.5-Omni-3B (baseline, identity adaptors)}"
echo "   layout     : $SHEET_LAYOUT | stats fwhm $STATS_FWHM"
echo "   categories : $CATEGORIES  x  top-{$TOP_K_PCTS}%"
echo "   batches    : $NUM_BATCHES of $SAVE_BATCH images, $NUM_SHARDS shards x $PER_SHARD"
echo

n=0
for (( shard=0; shard<NUM_SHARDS; shard++ )); do
    lo=$(( shard * PER_SHARD ))
    hi=$(( lo + PER_SHARD ))
    if [ "$lo" -ge "$NUM_BATCHES" ]; then break; fi
    if [ "$hi" -gt "$NUM_BATCHES" ]; then hi=$NUM_BATCHES; fi

    job="${JOB_PREFIX}-${shard}"
    echo "-- $job  (save-batches ${lo}..$(( hi - 1 )))"
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
        -m src.experiments.selectivity.extract_features \
        --model_name "$MODEL_NAME" \
        ${CHECKPOINT:+--checkpoint "$CHECKPOINT"} \
        ${BASELINE:+--baseline} \
        --stats_fwhm_mm "$STATS_FWHM" \
        --categories "$CATEGORIES" \
        --top_k_pcts "$TOP_K_PCTS" \
        --save_batch_size "$SAVE_BATCH" \
        --start_batch "$lo" \
        --end_batch "$hi"
    n=$(( n + 1 ))
done

echo
echo "== submitted $n shard(s). watch with: runai list | grep ${JOB_PREFIX}- =="
