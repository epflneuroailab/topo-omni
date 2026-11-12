#!/usr/bin/env bash
#SBATCH --job-name=jung_clusters
#SBATCH --account=upschrimpf1
#SBATCH --partition=bigmem
#SBATCH --qos=bigmem
#SBATCH --cpus-per-task=4
#SBATCH --mem=52G
#SBATCH --time=12:00:00
#SBATCH --array=0-26
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.out
#
# Cluster-vs-others fMRI GLM for EVERY cluster of one family: array task i = cluster ID i.
# Each task re-fits the per-subject GLM from the fsaverage6 BOLD (83 canonical subjects ->
# n=78 after the published confound drop) and runs the group test + FDR, writing
#   <RESULTS_ROOT>/cluster_contrasts_<FAMILY>clusters_exclbrokenruns/{subject_level,group_level}/
# By default it runs the analysis decided on 2026-09-23: the 4 runs whose events file only
# repeats another run's video rows are dropped (--exclude-broken-runs; README "Runs with a
# copied events file"). EXCLUDE_BROKEN_RUNS=0 runs the published analysis instead, into
# cluster_contrasts_<FAMILY>clusters/ (never mixed with the default's outputs).
# Measured cost: ~2 h and <52 GB per cluster on bigmem (July 2026 calibration; the Sep 2026
# smoke run took 80 s and 20 GB peak per subject).
# The array range above covers the new27 family (IDs 0-26); override on the command line for
# other families, e.g. --array=0-53 FAMILY=new54, or --array=0-15 FAMILY=new16.
#
# Submit FROM fMRI/Jung_2025 (paths and the log dir are relative to it):
#   cd fMRI/Jung_2025 && mkdir -p slurm/logs
#   sbatch --export=ALL,DERIV=<fsaverage6 derivatives>,BIDS=<ds005256 root>,PY=<python> \
#       slurm/cluster_contrasts_array.sbatch
# Smoke test first (one cluster, 3 subjects, ~10 min):
#   sbatch --array=0 --time=01:00:00 --export=ALL,DERIV=...,BIDS=...,PY=...,\
#RESULTS_ROOT=<scratch dir>,SUBJECTS="sub-0001 sub-0002 sub-0003" slurm/cluster_contrasts_array.sbatch
#
# Env: DERIV, BIDS (required) · PY (default python3; use the pinned Jung env,
# environment/analysis_env_jung.yml) · FAMILY (default new27) · RESULTS_ROOT (default DERIV)
# · SUBJECTS (space-separated; default all 83 canonical) · EXCLUDE_BROKEN_RUNS (default 1)
# · DRY_RUN=1 prints the command only.
set -euo pipefail

: "${DERIV:?set DERIV=<fsaverage6 derivatives root>}"
: "${BIDS:?set BIDS=<ds005256 BIDS root (task-alignvideo events)>}"
: "${SLURM_ARRAY_TASK_ID:?run as a SLURM array job (or set SLURM_ARRAY_TASK_ID)}"
PY="${PY:-python3}"
FAMILY="${FAMILY:-new27}"
RESULTS_ROOT="${RESULTS_ROOT:-$DERIV}"
JUNG_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

if [ ! -f "$JUNG_DIR/analysis/cluster_contrasts.py" ]; then
    echo "ERROR: submit from fMRI/Jung_2025 (no analysis/cluster_contrasts.py under $JUNG_DIR)" >&2
    exit 2
fi

SUFFIX="_exclbrokenruns"   # = config.EXCLUDED_RUNS_SUFFIX
if [ "${EXCLUDE_BROKEN_RUNS:-1}" = "1" ]; then
    EXCL_ARGS=(--exclude-broken-runs)
else
    EXCL_ARGS=()
    SUFFIX=""
fi

CMD=("$PY" -u "$JUNG_DIR/analysis/cluster_contrasts.py"
     --family "$FAMILY"
     --cluster-id "$SLURM_ARRAY_TASK_ID"
     --cluster-file "$JUNG_DIR/data/cluster_assignments/cluster_assignments_${FAMILY}clusters.csv"
     --derivatives-root "$DERIV"
     --bids-root "$BIDS"
     --output-dir "$RESULTS_ROOT/cluster_contrasts_${FAMILY}clusters${SUFFIX}"
     "${EXCL_ARGS[@]}")
if [ -n "${SUBJECTS:-}" ]; then
    read -r -a SUBJ_ARR <<< "$SUBJECTS"
    CMD+=(--subjects "${SUBJ_ARR[@]}")
fi

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "${CMD[*]}"
    exit 0
fi

echo "=== $(hostname) | family $FAMILY | cluster $SLURM_ARRAY_TASK_ID | $(date) ==="
echo "python: $PY ($("$PY" -c 'import numpy, scipy, nibabel; print("numpy", numpy.__version__, "scipy", scipy.__version__, "nibabel", nibabel.__version__)'))"
"${CMD[@]}"
echo "=== cluster $SLURM_ARRAY_TASK_ID done $(date) ==="
