#!/bin/bash
# Train the TASK-ONLY (non-topographic) control for the tiled32 sheet.
#
# Same architecture, same data, same optimisation as topo-omni-tiled32-4608 -- the config
# differs from tiled32_4608.yml by exactly one line, `apply-spatial-loss: false`. This is
# the control the Appendix E island-Moran's-I comparison needs, and the reason it matters
# that it be matched: the existing partial run reached only 4 epochs against the topo
# run's 10, which by itself can explain the reversal seen for the language and
# multiple-demand localizers (a less-trained adaptor sits closer to the tiled-identity
# initialisation, which is itself spatially autocorrelated).
#
# RESUMES from checkpoint-292 (step 292/730, epoch 4/10), which carries full DeepSpeed
# optimizer state, scheduler.pt, rng_state_*.pth and trainer_state.json. train.py globs
# `ckpts/<run-title>/checkpoint-*` and passes resume_from_checkpoint=True, so the resume is
# automatic -- there is no flag. It picks the HIGHEST step present, which is why the
# truncated checkpoint-365 had to be moved out of the run directory first; it is now under
# artifacts/ckpts/_quarantine/. Do not move it back.
#
# HARDWARE: 4 GPUs of >=140 GiB. Neither number is negotiable. The effective batch is
# batch-size 3 x grad-accum 5 x 4 processes = 60, which only matches the topo run at
# exactly 4 GPUs, and configs/accelerate.yml hardcodes num_processes: 4. The topo run
# measured ~139/140 GiB per device and the spatial loss it drops is only ~5 GiB of that,
# so an 80 GiB A100 (the `default` and `h100` pools) will OOM regardless. Shrinking the
# micro-batch to fit is not a way out either: batch 1 x accum 15 is the only 4-process
# split that keeps the effective batch, and it gives 72 optimizer steps per epoch against
# the checkpoint's 73, so max_steps and the cosine LR horizon stop matching the run this
# is a control for.
#
# H200 ONLY. DO NOT add b300 to the pool list, however tempting its 288 GiB looks.
# Tried on 2026-09-23: the job queued 7h, landed on gpu405 (b300) and every rank died in
# `_pool_cortical_states` with
#     nvrtc: error: invalid value for --gpu-architecture (-arch)
# The container's nvrtc (torch 2.9.1+cu128) cannot target Blackwell B300, and
# configs/accelerate.yml sets dynamo_backend: INDUCTOR, so the first step JIT-compiles a
# kernel and there is no way around it short of disabling Inductor -- which would stop this
# being a matched control, since the topo run trained with Inductor on H200. Naming b300
# is worse than useless: it is FIRST in scheduling priority, so the job waits for a b300
# slot and then fails on it.
#
# WATCH OUT: that failure was reported by runai as **Succeeded**. entrypoint.sh runs
#     python "${@:2}" 2>&1 | tee -a runai_output.txt || { ...; exit 1; }
# and a pipeline's exit status is `tee`'s, so a crashed trainer still exits 0. Check the
# run directory for new checkpoints, not the job status.
#
# Usage:
#   ./submit_taskonly_training.sh                    # resume to step 730
#   NODE_POOL=h200 ./submit_taskonly_training.sh     # explicit (same as default)
#
# Monitor: runai list | grep topoomni-taskonly-train
# Logs:    runai logs topoomni-taskonly-train -f
#          (also appended to <repo>/runai_output.txt by the container entrypoint)
# W&B:     project "topo-omni", run name "topo-omni-tiled32-4608-taskonly"

arg_node_pool=${NODE_POOL:-h200}
MY_IMAGE=bkhmsi/epfl:cortexlm

CONDA_ENV=topoomni
REPO_DIR=topo-omni-epfl

JOB_NAME=${JOB_NAME:-topoomni-taskonly-train}
CONFIG=${CONFIG:-experiments/train/tiled32_4608_taskonly.yml}
NUM_GPUS=${NUM_GPUS:-4}

# train.sh lifts `sheet-layout:` out of the config into TOPO_SHEET_LAYOUT itself, but the
# variable is also read at import time by anything the launcher touches first, so set it
# explicitly as well -- a mismatch here builds the adaptors at the wrong width and produces
# a checkpoint that is not comparable to the topo run at all.
SHEET_LAYOUT=${SHEET_LAYOUT:-tiled32}

echo "== submitting task-only training =="
echo "   job        : $JOB_NAME"
echo "   config     : $CONFIG"
echo "   gpus       : $NUM_GPUS on node pool $arg_node_pool"
echo "   layout     : $SHEET_LAYOUT"
echo "   resumes    : artifacts/ckpts/topo-omni-tiled32-4608-taskonly/checkpoint-292 -> step 730"
echo

runai submit \
    --name "$JOB_NAME" \
    -i $MY_IMAGE \
    --gpu "$NUM_GPUS" \
    --cpu 32 \
    --large-shm \
    --node-pools "$arg_node_pool" \
    -e "TOPO_SHEET_LAYOUT=$SHEET_LAYOUT" \
    -e "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
    --pvc nlp-scratch:/mnt/u14157_ic_nlp_001_files_nfs/nlpdata1 \
    -- \
    $REPO_DIR \
    $CONDA_ENV \
    scripts/submit_train.py \
    -c "$CONFIG" \
    --wandb

echo
echo "== submitted. watch with: runai list | grep $JOB_NAME =="
