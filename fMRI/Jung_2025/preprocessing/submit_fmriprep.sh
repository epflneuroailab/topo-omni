#!/bin/bash
# Stage 0 — staggered fMRIPrep submission over many subjects.
#
# Faithful port of dev fmriprep_jobs/05_submit_first_20.sh (staggered sbatch), parameterized
# (no baked-in subject list). Submits one fmriprep_single_subject.sbatch per subject with a
# delay between submissions so the scheduler / IO isn't hammered at once.
#
# Usage:
#   RAW_ROOT=/path/ds005256 SINGULARITY_IMAGE=/path/fmriprep-24.0.1.simg \
#   FS_LICENSE=/path/license.txt ./submit_fmriprep.sh sub-0001 sub-0002 ...
#   (or SUBJECTS_FILE=subjects.txt instead of positional args; DELAY=90 by default)
set -e

: "${RAW_ROOT:?set RAW_ROOT}"
: "${SINGULARITY_IMAGE:?set SINGULARITY_IMAGE}"
: "${FS_LICENSE:?set FS_LICENSE}"
DELAY=${DELAY:-90}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FMRIPREP_SBATCH="${SCRIPT_DIR}/fmriprep_single_subject.sbatch"

SUBJECTS=("$@")
if [ ${#SUBJECTS[@]} -eq 0 ] && [ -n "${SUBJECTS_FILE:-}" ]; then
    while IFS= read -r s || [ -n "$s" ]; do [ -n "$s" ] && SUBJECTS+=("$s"); done < "$SUBJECTS_FILE"
fi
[ ${#SUBJECTS[@]} -eq 0 ] && { echo "ERROR: no subjects (args or SUBJECTS_FILE=)"; exit 1; }

echo "=== submitting fMRIPrep for ${#SUBJECTS[@]} subjects (delay ${DELAY}s) ==="
i=0
for subject in "${SUBJECTS[@]}"; do
    i=$((i + 1))
    echo "[$i/${#SUBJECTS[@]}] $subject"
    sbatch --export=SUBJECT="$subject",RAW_ROOT="$RAW_ROOT",SINGULARITY_IMAGE="$SINGULARITY_IMAGE",FS_LICENSE="$FS_LICENSE",OUTPUT_DIR="${OUTPUT_DIR:-}",WORK_ROOT="${WORK_ROOT:-}",TEMPLATEFLOW_DIR="${TEMPLATEFLOW_DIR:-}" \
        "$FMRIPREP_SBATCH"
    [ $i -lt ${#SUBJECTS[@]} ] && [ "$DELAY" -gt 0 ] && sleep "$DELAY"
done
echo "=== all submitted. monitor: squeue -u \$USER ==="
