#!/usr/bin/env bash
# =============================================================================
# reproduce_precomputed_figures.sh
#
# Reviewer convenience wrapper for AlKhamissi & Mehrer et al., 2026 — reproduces
# the brain-side fMRI figures FROM THE HOSTED PRECOMPUTED CUT (the recommended,
# CI-covered path; no raw data, no fMRIPrep/FSL).
#
# It chains the three tools the release already ships:
#   1. download_precomputed.py  — pull each dataset's precomputed cut from OSF
#                                  (public, DOI 10.17605/OSF.IO/EHRT6) + sha256-verify
#   2. make_all_figures.py      — render every paper figure from that cut
#   3. download_figures.py      — pull the authors' PUBLISHED renders, so you can
#                                  put "mine" next to "theirs" and compare
#
# Outputs:
#   - your renders          -> fMRI/<Dataset>/plots/
#   - authors' renders      -> $REF_DIR/<Dataset>_fMRI_figures/
#   - downloaded input cut  -> $DATA_DIR/<Dataset>/   (bulky; .gitignore'd)
#
# Usage:
#   ./reproduce_precomputed_figures.sh                       # all 3 datasets
#   ./reproduce_precomputed_figures.sh Jung_2025             # one dataset (fast, ~143 MB, login-node OK)
#   DATA_DIR=/scratch/cut REF_DIR=/scratch/ref ./reproduce_precomputed_figures.sh
#   PYTHON=/path/to/python ./reproduce_precomputed_figures.sh
#
# NOTE ON PERNET MEMORY: the precomputed path for Pernet's Fig-3b RE-FITS a
# 218-subject group GLM (nilearn SecondLevelModel) and OOMs on a memory-capped
# login node — run Pernet on a compute/bigmem node (several GB RAM).
# See fMRI/README.md and Pernet_2015/README.md §7.
#
# NOTE ON RUNTIME: Jung (minutes) and Pernet (minutes, on a bigmem node) are fast.
# MARVI IS BY FAR THE SLOWEST — allow MULTIPLE HOURS: its Branch-A Fig-A2 lineage
# re-derives + cross-validates ~108 fROIs for each of the 6 subjects (thousands of
# resample_to_img calls), with no intermediate output, so a quiet terminal is normal,
# not a hang. Running the three datasets in parallel (one shell each) is fine.
#
# RESOURCES: single-threaded (nilearn n_jobs=1) — RAM binds, not cores. RAM peaks on
# Pernet (~9.5 GB). Rough guide: Jung ~8 GB / Pernet >=16 GB / Marvi ~8-16 GB; all three
# at once >=32 GB and 4-8 cores. Validated on one node with 80 GB / 20 cores (generous).
# Full table: fMRI/README.md "Compute resources".
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# --- configuration (override via env) ----------------------------------------
PYTHON="${PYTHON:-python3}"                              # interpreter with the render stack; override: PYTHON=/path/to/python (see requirements.txt)
DATA_DIR="${DATA_DIR:-$HERE/_precomputed_cut}"           # where the OSF cut is downloaded/extracted
REF_DIR="${REF_DIR:-$HERE/_reference_figures}"           # where the authors' published renders land
DATASETS=("$@")                                          # positional args = dataset subset (default: all)

echo "== Reproduce precomputed fMRI figures =="
echo "   python   : $PYTHON"
echo "   cut dir  : $DATA_DIR"
echo "   ref  dir : $REF_DIR"
echo "   datasets : ${DATASETS[*]:-all three}"
echo

# --- 0. dependency check ------------------------------------------------------
# The render stack + osfclient. Missing ones -> install requirements.txt.
echo "-- 0/3  checking dependencies --"
if ! "$PYTHON" - <<'PY'
import importlib.util as u, sys
# statsmodels is in requirements.txt but NOT imported by any figure lineage, so it is
# not required here (Moran's I stats use scipy/esda/libpysal).
need = ("nilearn","nibabel","numpy","pandas","scipy","sklearn",
        "matplotlib","seaborn","geopandas","osfclient","libpysal","esda")
missing = [m for m in need if u.find_spec(m) is None]
if missing:
    print("MISSING:", ", ".join(missing)); sys.exit(1)
print("all render + download deps present")
PY
then
    echo "   -> install the missing ones, e.g.:  $PYTHON -m pip install -r $HERE/requirements.txt"
    echo "   (or, for byte-faithful goldens, use the per-dataset conda envs in environment/)"
    exit 1
fi
echo

# build the optional --datasets flag shared by the download + render steps
DS_FLAG=()
if [ "${#DATASETS[@]}" -gt 0 ]; then
    DS_FLAG=(--datasets "${DATASETS[@]}")
fi

# --- 1. download the precomputed cut from OSF (idempotent + sha256-verified) --
echo "-- 1/3  download precomputed cut from OSF -> $DATA_DIR --"
"$PYTHON" download_precomputed.py --dest "$DATA_DIR" "${DS_FLAG[@]}"
echo

# --- 2. render every figure from the cut --------------------------------------
# make_all_figures maps --derivatives-root <BASE> to each dataset's cut flag
# (Pernet:--results-root, Marvi/Jung:--derivatives-root) and dispatches each
# make_figures.py as a subprocess. Figures land in each fMRI/<Dataset>/plots/.
echo "-- 2/3  render figures (make_all_figures.py) --"
"$PYTHON" make_all_figures.py \
    --input-source precomputed \
    --derivatives-root "$DATA_DIR" \
    --python "Pernet_2015=$PYTHON" \
    --python "Marvi_2025=$PYTHON" \
    --python "Jung_2025=$PYTHON" \
    "${DS_FLAG[@]}"
echo

# --- 3. download the authors' published renders for side-by-side comparison ---
echo "-- 3/3  download authors' published renders -> $REF_DIR --"
"$PYTHON" download_figures.py --dest "$REF_DIR" "${DS_FLAG[@]}" || \
    echo "   (reference-figure download failed or partial — your renders in <Dataset>/plots/ are still valid)"
echo

echo "== DONE =="
echo "  your renders   : $HERE/<Dataset>/plots/"
echo "  paper renders  : $REF_DIR/<Dataset>_fMRI_figures/"
echo "  compare the matching filenames between those two trees."
