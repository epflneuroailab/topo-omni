#!/bin/bash
#SBATCH --job-name=marvi_fmriprep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Stage 0 (step 2) — fMRIPrep 24.0.1 + FreeSurfer 7.3.2 for one subject.
# Faithful port of dev src/02_fmriprep_single_subject.sh (Marvi 2025 EMFL), parameterized
# (no baked-in dataset paths / subject). Produces the fMRIPrep derivatives + FreeSurfer
# recon-all that (together with the CVS warps from cvs_register_single_subject.sbatch) form
# the precomputed cut. Output spaces: T1w, MNI152NLin2009cAsym:res-2, fsnative, fsaverage5,
# fsaverage6, fsLR:den-91k (+ CIFTI).
#
# NOT golden-mastered (containerized, not bitwise-reproducible) — faithful port +
# provenance spot-check (docs/DESIGN.md §2.5/§6). The --input-source precomputed path skips this.
#
# Parameterized via environment variables (pass with sbatch --export=...). Example:
#   sbatch --export=SUBJECT=sub-kaneff01,RAW_ROOT=/path/ds006179,\
#FMRIPREP_IMAGE=/path/fmriprep-24.0.1.simg,FS_LICENSE=/path/license.txt,\
#TEMPLATEFLOW_DIR=/path/templateflow,WORK_ROOT=/path/work \
#     fmriprep_single_subject.sbatch
set -e

# --- Required parameters (no baked-in dataset paths) ---
: "${SUBJECT:?set SUBJECT=sub-XXXX (full BIDS label, e.g. sub-kaneff01)}"
: "${RAW_ROOT:?set RAW_ROOT=<ds006179 BIDS root>}"
: "${FMRIPREP_IMAGE:?set FMRIPREP_IMAGE=<fmriprep-24.0.1.simg>}"
: "${FS_LICENSE:?set FS_LICENSE=<FreeSurfer license.txt>}"
SUBJECT_ID=${SUBJECT#sub-}

# --- Optional parameters (sensible defaults) ---
OUTPUT_DIR="${OUTPUT_DIR:-${RAW_ROOT}/derivatives}"
WORK_DIR="${WORK_ROOT:-${RAW_ROOT}/work}"
TEMPLATEFLOW_DIR="${TEMPLATEFLOW_DIR:-${RAW_ROOT}/templateflow}"

echo "=== fMRIPrep 24.0.1 | ${SUBJECT} | $(date) | $(hostname) ==="
echo "  RAW:    ${RAW_ROOT}"
echo "  OUTPUT: ${OUTPUT_DIR}"
echo "  WORK:   ${WORK_DIR}"
echo "  OUTPUT SPACES: T1w MNI152NLin2009cAsym:res-2 fsnative fsaverage5 fsaverage6 fsLR:den-91k (+CIFTI)"

mkdir -p "${OUTPUT_DIR}" "${WORK_DIR}" "${TEMPLATEFLOW_DIR}"
export FS_LICENSE=${FS_LICENSE}
export TEMPLATEFLOW_HOME=${TEMPLATEFLOW_DIR}
export APPTAINERENV_TEMPLATEFLOW_HOME=${TEMPLATEFLOW_DIR}
export SINGULARITYENV_TEMPLATEFLOW_HOME=${TEMPLATEFLOW_DIR}  # legacy compatibility

apptainer run --cleanenv \
    --bind ${RAW_ROOT}:${RAW_ROOT}:ro \
    --bind ${OUTPUT_DIR}:${OUTPUT_DIR} \
    --bind ${WORK_DIR}:${WORK_DIR} \
    --bind ${TEMPLATEFLOW_DIR}:${TEMPLATEFLOW_DIR} \
    --bind ${FS_LICENSE}:/opt/freesurfer/license.txt:ro \
    ${FMRIPREP_IMAGE} \
    ${RAW_ROOT} \
    ${OUTPUT_DIR} \
    participant \
    --participant-label ${SUBJECT_ID} \
    --output-spaces T1w MNI152NLin2009cAsym:res-2 fsnative fsaverage5 fsaverage6 fsLR:den-91k \
    --cifti-output \
    --nthreads 16 \
    --omp-nthreads 8 \
    --mem_mb 64000 \
    --fs-license-file /opt/freesurfer/license.txt \
    --skip_bids_validation \
    --write-graph \
    --stop-on-first-crash \
    -w ${WORK_DIR}

# set -e above means a nonzero fMRIPrep exit already aborted the job; reaching here = success.
echo "=== fMRIPrep complete | ${SUBJECT} | $(date) ==="
echo "  derivatives: ${OUTPUT_DIR}/sub-${SUBJECT_ID}"
echo "  FreeSurfer:  ${OUTPUT_DIR}/sourcedata/freesurfer/sub-${SUBJECT_ID}  (needed by CVS registration)"
echo "  report:      ${OUTPUT_DIR}/sub-${SUBJECT_ID}.html"
