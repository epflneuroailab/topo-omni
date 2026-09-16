#!/bin/bash
# Stimulation sweep for figure 6a — one runai job per (localizer, stimuli, coverage, strength).
#
# Asks whether DRIVING a region's units makes the model report that region's category on a
# stimulus that does not contain it. So each localizer is paired only with the stimulus sets
# it does NOT match: stimulating the face region while showing a scene is an induced face
# percept, whereas showing a face would confound induction with what is already there.
#
# Grid: 4 localizers x 3 non-matching stimuli x 6 coverages => 72 jobs.
#
# There is deliberately no strength axis. Stimulation applies a FIXED steering vector (the
# mean-activation difference on the selected units) with no scalar gain -- the old
# --stimulation_strength flag never reached the forward pass and has been removed. A sweep
# over it produced byte-identical outputs across all 72 configs.
#
# driving_perception.py plots coverage on x and one line per localizer, reading the
# `prediction` field, so the judge must run in choice mode:
#   python -m src.experiments.ablation.similarity --model_name <run> --stimulate \
#       --judge_mode choice --percentages 5,10,15,20,25,30
#
# PREREQUISITE: the mean-activation cache, same as the ablation grid — see
# submit_topoomni_ablation.sh. It is already seeded if that grid has run.
#
# Usage:
#   ./scripts/submit_topoomni_stimulation.sh                  # full sweep (72 jobs)
#   PERCENTAGES=10 ./scripts/submit_topoomni_stimulation.sh   # a 12-job probe
#
# Monitor:  runai list | grep topoomni-stim-
# Clean up: for j in $(runai list 2>/dev/null | awk '/topoomni-stim-/{print $1}'); do runai delete job $j; done

arg_node_pool=${NODE_POOL:-default}
MY_IMAGE=bkhmsi/epfl:cortexlm
CONDA_ENV=topoomni
REPO_DIR=topo-omni-epfl

MODEL_NAME=${MODEL_NAME:-topo-omni-tiled32-4608}
FWHM=${FWHM:-3.0}
MODE=${MODE:-top_p}
CHECKPOINT=${CHECKPOINT:-artifacts/ckpts/${MODEL_NAME}/checkpoint-730}
COORDS=${COORDS:-src/neighborhoods/model=qwen2-5omni-3b-tiled32-4608-v7_radius=dynamic_coords=42/coords.npy}
SHEET_LAYOUT=${SHEET_LAYOUT:-tiled32}

# driving_perception.py's x axis. Keep the two in step or the line plot has gaps.
PERCENTAGES=${PERCENTAGES:-5,10,15,20,25,30}

CATS=(faces bodies scenes objects)
IFS=',' read -r -a PCTS <<< "$PERCENTAGES"

echo "== submitting stimulation jobs =="
echo "   model       : $MODEL_NAME ($CHECKPOINT)"
echo "   coverages   : ${PCTS[*]}"
echo "   pairs       : each localizer x the 3 stimuli it does NOT match"
echo

n=0
for l in "${CATS[@]}"; do
  for s in "${CATS[@]}"; do
    [ "$s" = "$l" ] && continue          # matched pair: percept already present
    for p in "${PCTS[@]}"; do
      {
        job="topoomni-stim-${s}-${l}-p${p}"
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
            -m src.experiments.ablation.run \
            --model_name "$MODEL_NAME" \
            --stats_fwhm_mm "$FWHM" \
            --coords "$COORDS" \
            --checkpoint "$CHECKPOINT" \
            --stimuli "$s" \
            --stimulate \
            --localizer "$l" \
            --mode "$MODE" \
            --percentage "$p" >/dev/null 2>&1
        n=$((n+1))
      }
    done
  done
done

echo
echo "== submitted $n job(s). watch with: runai list | grep topoomni-stim- =="
