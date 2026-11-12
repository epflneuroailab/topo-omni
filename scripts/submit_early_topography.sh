#!/bin/bash
# Early-topography evals (retinotopy + orientation) on the TILED-32 run at checkpoint-730.
#
# WHY. Both analyses in the manuscript were measured on the wrong model. Fig. 2e-f
# (polar angle / eccentricity) comes from the old 309x512 strip run, and the orientation /
# pinwheel numbers in the new SI section come from step 438 of this run rather than 730.
# Every other number in the paper is ckpt730, and Reviewer #1's point 3 is specifically
# about these panels, so they need to be on the same model as the claims made around them.
#
# THE COORDS TRAP. run_angle.py and run.py had the OLD run's coords path hardcoded
# (158,208 units) and run_ecc.py used unified_grid_coords(), the identity grid. Neither is
# right for tiled32: build_sheet_coordinates shuffles units within their layer block, so
# the identity grid restores only 94 of 460,800 units to their trained position, and a
# wrong-sized coords file is a different sheet entirely. All three now read
# stats.coords_path from the config and abort if the unit count disagrees with
# TOPO_SHEET_LAYOUT. fwhm_mm is likewise config-driven (it was 2.0 / 4.0 / 1.4 hardcoded
# per script, so the angle and eccentricity panels were smoothed differently and could not
# be read against one another).
#
# Three jobs, all independent single-GPU forward passes over image banks:
#   angle        polar-angle wedges      -> results/<run>/retinotopy_angle/
#   ecc          eccentricity rings      -> results/<run>/retinotopy_eccentricity/
#   orientation  TDANN unbiased gratings -> results/<run>-nosmooth/orientation_unbiased/
#
# The orientation job fetches the grating bank from OSF on first run if missing.
#
# Usage:   ./submit_early_topography.sh [angle,ecc,orientation]
# Monitor: runai list | grep topoomni-early-
# Logs:    runai logs topoomni-early-angle -f

arg_which=${1:-all}
arg_node_pool=${NODE_POOL:-default}
MY_IMAGE=bkhmsi/epfl:cortexlm

CONDA_ENV=topoomni
REPO_DIR=topo-omni-epfl
SHEET_LAYOUT=${SHEET_LAYOUT:-tiled32}

RETINO_CFG=${RETINO_CFG:-src/configs/experiments/eval/retinotopy_tiled32_ckpt730.yml}
ORI_CFG=${ORI_CFG:-src/configs/experiments/eval/orientation_tiled32_ckpt730.yml}

ALL=(angle ecc orientation)
if [ "$arg_which" = "all" ]; then
    WHICH=("${ALL[@]}")
else
    IFS=',' read -r -a WHICH <<< "$arg_which"
fi

submit_one() {
    local job=$1; shift
    echo "-- $job"
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
        "$@"
}

echo "== submitting early-topography evals on checkpoint-730 =="
echo "   layout    : $SHEET_LAYOUT"
echo "   retinotopy: $RETINO_CFG"
echo "   orientation: $ORI_CFG"
echo

n=0
for w in "${WHICH[@]}"; do
    case "$w" in
        angle)
            submit_one topoomni-early-angle \
                -m src.experiments.retinotopy.run_angle --config "$RETINO_CFG" ;;
        ecc)
            submit_one topoomni-early-ecc \
                -m src.experiments.retinotopy.run_ecc --config "$RETINO_CFG" ;;
        orientation)
            submit_one topoomni-early-orientation \
                -m src.experiments.orientation.run --config "$ORI_CFG" ;;
        *) echo "unknown target '$w' (expected: ${ALL[*]})"; exit 1 ;;
    esac
    n=$((n+1))
done

echo
echo "== submitted $n job(s). watch with: runai list | grep topoomni-early- =="
