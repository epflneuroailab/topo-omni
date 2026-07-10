#!/bin/bash
#SBATCH --job-name=marvi_cvs_reg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Stage 0 (step 16) — CVS registration for one subject (FreeSurfer 7.3.2 `mri_cvs_register`).
# Faithful port of dev src/16_run_cvs_registration.sh (Marvi 2025 EMFL), parameterized
# (no baked-in paths / subject).
#
# CVS = Combined Volume and Surface registration: a nonlinear subject->template warp used to
# project the atlas fROI parcels into each subject's native space. Registers the subject to two
# templates, producing the .m3z morph files consumed by analysis/project_parcels_to_surface.py
# (via `mri_vol2vol --m3z ... --inv-morph`, README §9):
#   tocvs_avg35/final_CVSmorph_tocvs_avg35.m3z            <- Julian parcels (FFA/OFA/PPA/...)
#   tocvs_avg35_inMNI152/final_CVSmorph_tocvs_avg35_inMNI152.m3z   <- VWFA / MD parcels
#
# These .m3z warps are shipped inside the precomputed cut (config.py `cvs_m3z_warps`), so the
# --input-source precomputed path never runs this. NOT golden-mastered (containerized, not
# bitwise-reproducible) — faithful port + provenance spot-check (docs/DESIGN.md §2.5/§6).
#
# Needs the FreeSurfer recon-all output from fMRIPrep (this container has FreeSurfer at
# /usr/local/freesurfer, so the license binds there — not /opt/freesurfer as in the fMRIPrep image).
#
# Parameterized via environment variables (pass with sbatch --export=... or as a positional arg):
#   sbatch --export=SUBJECT=sub-kaneff01,FREESURFER_IMAGE=/path/freesurfer-7.3.2.sif,\
#FS_LICENSE=/path/license.txt,DERIVATIVES=/path/ds006179/derivatives \
#     cvs_register_single_subject.sbatch
#   (or: bash cvs_register_single_subject.sbatch sub-kaneff01   with the same env vars set)
set -e

SUBJECT="${SUBJECT:-${1:?set SUBJECT=sub-XXXX (env var or positional arg)}}"
: "${FREESURFER_IMAGE:?set FREESURFER_IMAGE=<freesurfer-7.3.2 .sif>}"
: "${FS_LICENSE:?set FS_LICENSE=<FreeSurfer license.txt>}"
: "${DERIVATIVES:?set DERIVATIVES=<fMRIPrep derivatives root>}"
FS_SUBJECTS_DIR="${FS_SUBJECTS_DIR:-${DERIVATIVES}/sourcedata/freesurfer}"
OUTPUT_BASE="${OUTPUT_BASE:-${DERIVATIVES}/cvs_transforms}"

echo "=== CVS registration | ${SUBJECT} | $(date) | $(hostname) ==="
echo "  FS subjects: ${FS_SUBJECTS_DIR}"
echo "  output:      ${OUTPUT_BASE}/${SUBJECT}"

if [ ! -d "${FS_SUBJECTS_DIR}/${SUBJECT}" ]; then
    echo "ERROR: FreeSurfer subject not found: ${FS_SUBJECTS_DIR}/${SUBJECT}"; exit 1
fi

# Run one subject->template CVS registration inside the FreeSurfer container.
# Binds the subject's recon-all dir into the container's own subjects dir so mri_cvs_register
# can also find the cvs_avg35* templates that ship inside the image.
run_cvs () {
    local template="$1" out_dir="$2" expected="$3"
    mkdir -p "${out_dir}"
    if [ -f "${out_dir}/${expected}" ]; then
        echo "  SKIP: ${expected} already exists"; return 0
    fi
    echo "  ${SUBJECT} -> ${template}  (out: ${out_dir})"
    apptainer exec --cleanenv \
        --bind ${FS_LICENSE}:/usr/local/freesurfer/license.txt:ro \
        --bind ${FS_SUBJECTS_DIR}/${SUBJECT}:/usr/local/freesurfer/subjects/${SUBJECT} \
        --bind ${out_dir}:${out_dir} \
        --env FS_LICENSE=/usr/local/freesurfer/license.txt \
        ${FREESURFER_IMAGE} \
        bash -c "
            export SUBJECTS_DIR=/usr/local/freesurfer/subjects
            mri_cvs_register \
                --mov ${SUBJECT} \
                --template ${template} \
                --outdir ${out_dir} \
                --nocleanup
        "
    if [ -f "${out_dir}/${expected}" ]; then
        echo "  SUCCESS: ${expected}"
    else
        echo "  ERROR: ${template} registration failed (${expected} missing)"; exit 1
    fi
}

# Registration 1: subject -> cvs_avg35            (Julian parcels)
run_cvs cvs_avg35 \
    "${OUTPUT_BASE}/${SUBJECT}/tocvs_avg35" \
    final_CVSmorph_tocvs_avg35.m3z

# Registration 2: subject -> cvs_avg35_inMNI152   (VWFA / MD parcels)
run_cvs cvs_avg35_inMNI152 \
    "${OUTPUT_BASE}/${SUBJECT}/tocvs_avg35_inMNI152" \
    final_CVSmorph_tocvs_avg35_inMNI152.m3z

echo "=== CVS registration complete: ${SUBJECT} | $(date) ==="
