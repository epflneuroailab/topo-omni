#!/bin/bash
# Stage 0 — download the task-alignvideo subset of OpenNeuro ds005256 via DataLad.
#
# Faithful port of dev download_raw_data/{download_alignvideo_batch,
# remove_all_non_alignvideo_symlinks}.sh (20251211_fMRI_movie_watching_spacetop @ 4066746),
# parameterized (no baked-in paths / batch files, no interactive sleep). Downloads AND
# unlocks (git-annex symlinks → real files) only what fMRIPrep needs: alignvideo BOLD +
# sbref, T1w anat, fieldmaps, metadata. Optionally prunes non-alignvideo tasks afterwards.
#
# Prereqs: the ds005256 DataLad dataset already cloned at RAW_ROOT
#   (datalad clone https://github.com/OpenNeuroDatasets/ds005256.git RAW_ROOT).
#
# Usage:
#   RAW_ROOT=/path/ds005256 ./download_raw_data.sh sub-0001 sub-0002 ...
#   RAW_ROOT=/path/ds005256 SUBJECTS_FILE=subjects.txt ./download_raw_data.sh
#   RAW_ROOT=/path/ds005256 PRUNE=1 ./download_raw_data.sh sub-0001   # also drop other tasks
set -e

: "${RAW_ROOT:?set RAW_ROOT=<ds005256 BIDS root>}"

# Collect subjects from args or SUBJECTS_FILE
SUBJECTS=("$@")
if [ ${#SUBJECTS[@]} -eq 0 ] && [ -n "${SUBJECTS_FILE:-}" ]; then
    while IFS= read -r s || [ -n "$s" ]; do [ -n "$s" ] && SUBJECTS+=("$s"); done < "$SUBJECTS_FILE"
fi
[ ${#SUBJECTS[@]} -eq 0 ] && { echo "ERROR: no subjects (pass as args or SUBJECTS_FILE=)"; exit 1; }

cd "$RAW_ROOT"
echo "=== DataLad get (task-alignvideo) for ${#SUBJECTS[@]} subjects into $RAW_ROOT ==="

for subject in "${SUBJECTS[@]}"; do
    echo "--- $subject ---"
    if [ -z "$(find "${subject}" -name '*task-alignvideo*.nii.gz' 2>/dev/null)" ]; then
        echo "  ⚠ no alignvideo files present, skipping"; continue
    fi
    datalad get -J 4 ${subject}/ses-01/anat/*T1w*                                  2>&1 | tail -1
    datalad get -J 4 ${subject}/ses-*/func/*task-alignvideo*bold.nii.gz            2>&1 | tail -1
    datalad get -J 4 ${subject}/ses-*/func/*task-alignvideo*sbref.nii.gz           2>&1 | tail -1
    datalad get     ${subject}/ses-*/func/*task-alignvideo*.json                   2>&1 | tail -1
    datalad get     ${subject}/ses-*/func/*task-alignvideo*.tsv                    2>&1 | tail -1
    datalad get     ${subject}/ses-01/anat/*.json                                  2>&1 | tail -1
    datalad get -J 4 ${subject}/ses-*/fmap/*.nii.gz                                2>&1 | tail -1
    datalad get     ${subject}/ses-*/fmap/*.json                                   2>&1 | tail -1
    datalad unlock  ${subject}                                                     2>&1 | tail -1

    if [ "${PRUNE:-0}" = "1" ]; then
        non_align=$(find "${subject}" -name '*task-*' ! -name '*task-alignvideo*' 2>/dev/null || true)
        if [ -n "$non_align" ]; then
            echo "  pruning $(echo "$non_align" | wc -l) non-alignvideo files"
            echo "$non_align" | while read -r f; do rm -f "$f"; done
        fi
    fi
    echo "  ✓ $subject"
done

echo "=== done. fMRIPrep next: sbatch --export=SUBJECT=<sub>,RAW_ROOT=$RAW_ROOT,... fmriprep_single_subject.sbatch ==="
