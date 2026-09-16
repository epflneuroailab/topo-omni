#!/bin/bash
# SpaceTop topographic discovery — find category-selective regions without a localizer.
#
# Two phases, because they are bound by different things:
#
#   EXTRACT (GPU jobs, sharded). One forward pass per 2-second clip to cache its mean
#     unified cortical sheet. The GPU sits near idle — the cost is decoding video+audio
#     — so this shards across N workers that share the sheet cache and skip what is
#     already there. Resumable: re-run and it picks up where it stopped.
#
#   CLUSTER (one job). Walk the Ward tree over the clip embeddings top-down, splitting
#     a node only while both children are MORE selective than the parent (median Welch t
#     of the cluster's sheets vs every other clip's). Then dendrogram, per-cluster
#     selectivity, and the collages you label the clusters from. Pure numpy over the
#     cached sheets — no GPU needed, but it wants the sheets complete.
#
# The clip embeddings (NVIDIA Omni-Embed-Nemotron) that seed the tree are stimulus-side
# and checkpoint-independent, so they are NOT re-extracted here. Seed a new run's
# directory with an existing pair:
#     cp <somewhere>/{video_clip_vectors.npy,clips_manifest.json} \
#        $SAVE_DIR/$MODEL_NAME/spacetop_discovery/
# or regenerate them with extract_video_embeddings + compile_embeddings.
#
# TOPO_SHEET_LAYOUT is read at IMPORT time and must match the checkpoint (tiled32 for
# the 800x576 runs); .env is loaded too late, hence -e below. Get it wrong and a random
# front-end adaptor is applied silently.
#
# Run this from the WORKSTATION, not inside a pod -- the pods have no runai CLI, and
# `runai submit` there fails silently into /dev/null.
#
# Usage:
#   ./scripts/submit_topoomni_spacetop_discovery.sh extract    # N sharded GPU jobs
#   ./scripts/submit_topoomni_spacetop_discovery.sh cluster    # the walk + figures
#   ./scripts/submit_topoomni_spacetop_discovery.sh all        # extract, then cluster
#
# Monitor:  runai list | grep topoomni-spacetop-

arg_node_pool=${NODE_POOL:-default}
MY_IMAGE=bkhmsi/epfl:cortexlm
CONDA_ENV=topoomni
REPO_DIR=topo-omni-epfl

MODEL_NAME=${MODEL_NAME:-topo-omni-tiled32-4608}
CHECKPOINT=${CHECKPOINT:-artifacts/ckpts/${MODEL_NAME}}
SHEET_LAYOUT=${SHEET_LAYOUT:-tiled32}
NUM_SHARDS=${NUM_SHARDS:-8}

MODE=${1:-all}

submit() {
  # submit <job-suffix> <gpus> <module> [args...]
  local suffix=$1; shift
  local gpus=$1; shift
  local module=$1; shift
  local job="topoomni-spacetop-${suffix}"
  echo "-- $job"
  runai submit \
      --name "$job" \
      -i $MY_IMAGE \
      --gpu "$gpus" \
      --cpu 8 \
      --large-shm \
      --node-pools "$arg_node_pool" \
      -e "TOPO_SHEET_LAYOUT=$SHEET_LAYOUT" \
      -e "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
      --pvc nlp-scratch:/mnt/u14157_ic_nlp_001_files_nfs/nlpdata1 \
      -- \
      $REPO_DIR \
      $CONDA_ENV \
      -m "$module" \
      --model_name "$MODEL_NAME" \
      "$@" >/dev/null 2>&1
}

echo "== spacetop discovery =="
echo "   model  : $MODEL_NAME ($CHECKPOINT)"
echo "   layout : $SHEET_LAYOUT"
echo "   mode   : $MODE"
echo

M=src.experiments.spacetop_discovery

if [ "$MODE" = "extract" ] || [ "$MODE" = "all" ]; then
  echo "-- extraction: $NUM_SHARDS shards"
  for i in $(seq 0 $((NUM_SHARDS - 1))); do
    submit "extract-$i" 1 "$M.agglomerative_early_stopping" \
        --checkpoint "$CHECKPOINT" --extract_only \
        --shard "$i" --num_shards "$NUM_SHARDS"
  done
  echo
  echo "   wait for all $NUM_SHARDS to finish, then re-run with 'cluster'."
fi

if [ "$MODE" = "cluster" ] || [ "$MODE" = "all" ]; then
  # One GPU: any clip a shard missed is extracted here before the walk starts.
  echo "-- clustering + figures"
  submit "cluster" 1 "$M.agglomerative_early_stopping" --checkpoint "$CHECKPOINT"
  submit "dendrogram" 0 "$M.plot_dendrogram"
  submit "selectivity" 0 "$M.calculate_selectivity"
  submit "plot" 0 "$M.plot_model_selectivity"
  submit "collages" 0 "$M.make_cluster_collages"
  echo
  echo "   these four run in dependency order -- submit them one at a time, or run"
  echo "   them locally in the pod, since runai has no job dependencies here."
fi

echo
echo "== done submitting. watch with: runai list | grep topoomni-spacetop- =="
