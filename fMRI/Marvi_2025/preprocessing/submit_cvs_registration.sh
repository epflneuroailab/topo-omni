#!/bin/bash
# Stage 0 (step 17) — batch CVS registration submission over many subjects.
#
# Faithful port of dev src/17_batch_cvs_registration.sh (Marvi 2025 EMFL), parameterized
# (no baked-in subject list / paths). Submits one cvs_register_single_subject.sbatch per subject.
#
# Usage:
#   FREESURFER_IMAGE=/path/freesurfer-7.3.2.sif FS_LICENSE=/path/license.txt \
#   DERIVATIVES=/path/ds006179/derivatives ./submit_cvs_registration.sh sub-kaneff01 sub-kaneff06 ...
#   (or SUBJECTS_FILE=subjects.txt instead of positional args)
#
# Test one subject first:
#   FREESURFER_IMAGE=... FS_LICENSE=... DERIVATIVES=... \
#     bash cvs_register_single_subject.sbatch sub-kaneff01
set -e

: "${FREESURFER_IMAGE:?set FREESURFER_IMAGE=<freesurfer-7.3.2 .sif>}"
: "${FS_LICENSE:?set FS_LICENSE=<FreeSurfer license.txt>}"
: "${DERIVATIVES:?set DERIVATIVES=<fMRIPrep derivatives root>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CVS_SBATCH="${SCRIPT_DIR}/cvs_register_single_subject.sbatch"

SUBJECTS=("$@")
if [ ${#SUBJECTS[@]} -eq 0 ] && [ -n "${SUBJECTS_FILE:-}" ]; then
    while IFS= read -r s || [ -n "$s" ]; do [ -n "$s" ] && SUBJECTS+=("$s"); done < "$SUBJECTS_FILE"
fi
[ ${#SUBJECTS[@]} -eq 0 ] && { echo "ERROR: no subjects (args or SUBJECTS_FILE=)"; exit 1; }

echo "=== submitting CVS registration for ${#SUBJECTS[@]} subjects ==="
for subject in "${SUBJECTS[@]}"; do
    JOB_ID=$(sbatch --job-name="marvi_cvs_${subject}" \
        --export=SUBJECT="$subject",FREESURFER_IMAGE="$FREESURFER_IMAGE",FS_LICENSE="$FS_LICENSE",DERIVATIVES="$DERIVATIVES",FS_SUBJECTS_DIR="${FS_SUBJECTS_DIR:-}",OUTPUT_BASE="${OUTPUT_BASE:-}" \
        "$CVS_SBATCH" | awk '{print $4}')
    echo "  submitted ${subject}: job ${JOB_ID}"
done
echo "=== all submitted. monitor: squeue -u \$USER ==="
