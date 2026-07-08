#!/bin/bash
#SBATCH --job-name=jung_fmriprep
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=55
#SBATCH --mem=360G
#SBATCH --time=72:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Stage 0 — fMRIPrep 24.0.1 + FreeSurfer 7.x, surface-only output (fsaverage6 primary).
# Faithful port of dev fmriprep_jobs/00_fmriprep_single_subject_optimized.sh
# (20251211_fMRI_movie_watching_spacetop @ 4066746). Produces the precomputed cut:
#   *_hemi-{L,R}_space-fsaverage6_bold.func.gii  +  *_desc-confounds_timeseries.tsv
# MNI152 + fsLR-CIFTI are intentionally skipped (save ~54 GB/subject; not used downstream).
#
# NOT golden-mastered (containerized, not bitwise-reproducible) — faithful port +
# provenance spot-check (docs/DESIGN.md §2.5/§6). The --input-source precomputed path skips this.
#
# Parameterized via environment variables (pass with sbatch --export=...). Example:
#   sbatch --export=SUBJECT=sub-0001,RAW_ROOT=/path/ds005256,\
#SINGULARITY_IMAGE=/path/fmriprep-24.0.1.simg,FS_LICENSE=/path/license.txt,\
#TEMPLATEFLOW_DIR=/path/templateflow,WORK_ROOT=/path/fmriprep_work \
#     fmriprep_single_subject.sbatch
set -e

# --- Required parameters (no baked-in dataset paths) ---
: "${SUBJECT:?set SUBJECT=sub-XXXX}"
: "${RAW_ROOT:?set RAW_ROOT=<ds005256 BIDS root>}"
: "${SINGULARITY_IMAGE:?set SINGULARITY_IMAGE=<fmriprep-24.0.1.simg>}"
: "${FS_LICENSE:?set FS_LICENSE=<FreeSurfer license.txt>}"
SUBJECT_ID=${SUBJECT#sub-}

# --- Optional parameters (sensible defaults) ---
OUTPUT_DIR="${OUTPUT_DIR:-${RAW_ROOT}/derivatives}"
WORK_DIR="${WORK_ROOT:-${RAW_ROOT}/fmriprep_work}/${SUBJECT}"
TEMPLATEFLOW_DIR="${TEMPLATEFLOW_DIR:-${RAW_ROOT}/templateflow}"
BIDS_FILTER="${BIDS_FILTER:-$(dirname "$0")/bids_filter_alignvideo.json}"

echo "=== fMRIPrep (surface-only) | ${SUBJECT} | $(date) | $(hostname) ==="
echo "  RAW:    ${RAW_ROOT}"
echo "  OUTPUT: ${OUTPUT_DIR}"
echo "  WORK:   ${WORK_DIR}"
echo "  OUTPUT SPACES: fsaverage6 (primary) fsaverage5 (sanity) anat  [MNI/fsLR skipped]"

mkdir -p "${WORK_DIR}" "${OUTPUT_DIR}" "${TEMPLATEFLOW_DIR}"
export TEMPLATEFLOW_HOME=${TEMPLATEFLOW_DIR}
export APPTAINERENV_TEMPLATEFLOW_HOME=${TEMPLATEFLOW_DIR}

singularity run --cleanenv \
    -B ${RAW_ROOT}:/data:ro \
    -B ${OUTPUT_DIR}:/out \
    -B ${WORK_DIR}:/work \
    -B ${TEMPLATEFLOW_DIR}:/templateflow \
    -B ${FS_LICENSE}:/opt/freesurfer/license.txt:ro \
    -B ${BIDS_FILTER}:/bids_filter.json:ro \
    ${SINGULARITY_IMAGE} \
    /data /out participant \
    --participant-label ${SUBJECT_ID} \
    --task-id alignvideo \
    --bids-filter-file /bids_filter.json \
    --output-spaces fsaverage5 fsaverage6 anat \
    --use-syn-sdc \
    --skull-strip-template OASIS30ANTs \
    --work-dir /work \
    --n-cpus 55 \
    --mem-mb 350000 \
    --omp-nthreads 40 \
    --skip_bids_validation \
    --stop-on-first-crash \
    --notrack

echo "=== fMRIPrep complete: ${SUBJECT} | $(date) ==="
OUT_SUB="${OUTPUT_DIR}/${SUBJECT}"
if [ -d "${OUT_SUB}" ]; then
    echo "fsaverage6 GIFTIs: $(find ${OUT_SUB} -name '*space-fsaverage6*bold.func.gii' | wc -l)"
    echo "confounds TSVs:    $(find ${OUT_SUB} -name '*confounds_timeseries.tsv' | wc -l)"
    du -sh "${OUT_SUB}"
else
    echo "ERROR: output dir not found: ${OUT_SUB}"; exit 1
fi

# Reclaim the (large) working directory
[ -d "${WORK_DIR}" ] && rm -rf "${WORK_DIR}" && echo "cleaned work dir"
