#!/usr/bin/env bash
# =============================================================================
# reproduce_precomputed_figures.sh
#
# One-command reproduction of the MODEL-side paper figures (AlKhamissi & Mehrer
# et al., 2026) FROM THE HOSTED PRECOMPUTED CUT — no GPU, no stimuli, no model.
#
# It chains the two release tools:
#   1. download_precomputed.py  — pull the precomputed cut from OSF + sha256-verify
#   2. make_all_figures.py      — render the figures from that cut
#
# Outputs:
#   your renders          -> $OUT_DIR/figure_<id>/
#   downloaded input cut  -> $CUT_DIR/   (bulky; .gitignore'd)
#
# Usage:
#   ./reproduce_precomputed_figures.sh                 # all model-side figures
#   ./reproduce_precomputed_figures.sh 3 4             # a subset
#   CUT_DIR=/scratch/cut OUT_DIR=/scratch/figs ./reproduce_precomputed_figures.sh
#   PYTHON=/path/to/python ./reproduce_precomputed_figures.sh
#
# The brain-side figures are separate — see fMRI/reproduce_precomputed_figures.sh.
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PYTHON="${PYTHON:-python3}"
CUT_DIR="${CUT_DIR:-$HERE/_precomputed_cut}"
OUT_DIR="${OUT_DIR:-$HERE/figures_out}"
FIGURES=("$@")   # positional args = figure subset (default: all)

export MPLBACKEND="${MPLBACKEND:-Agg}"

echo "== Reproduce precomputed model figures =="
echo "   python   : $PYTHON"
echo "   cut dir  : $CUT_DIR"
echo "   out dir  : $OUT_DIR"
echo "   figures  : ${FIGURES[*]:-all}"
echo

# --- 0. dependency check -----------------------------------------------------
echo "-- 0/2  checking dependencies --"
if ! "$PYTHON" - <<'PY'
import importlib.util as u, sys
need = ("numpy", "pandas", "scipy", "matplotlib", "seaborn", "skimage", "osfclient")
missing = [m for m in need if u.find_spec(m) is None]
if missing:
    print("MISSING:", ", ".join(missing)); sys.exit(1)
print("all render + download deps present")
PY
then
    echo "   -> install them:  $PYTHON -m pip install -r $HERE/requirements.txt"
    exit 1
fi
echo

FIG_FLAG=()
if [ "${#FIGURES[@]}" -gt 0 ]; then
    FIG_FLAG=(--figures "$(IFS=,; echo "${FIGURES[*]}")")
fi

# --- 1. download the precomputed cut from OSF (idempotent + sha256-verified) --
echo "-- 1/2  download precomputed cut from OSF -> $CUT_DIR --"
"$PYTHON" download_precomputed.py --dest "$CUT_DIR"
echo

CUT_DIR=${CUT_DIR}/topo-omni-cut  # the cut is inside a subdir of the zip

# --- 2. render the figures from the cut --------------------------------------
echo "-- 2/2  render figures (make_all_figures.py) --"
"$PYTHON" make_all_figures.py \
    --input-source precomputed \
    --derivatives-root "$CUT_DIR" \
    --out "$OUT_DIR" \
    "${FIG_FLAG[@]}"
echo

echo "== DONE =="
echo "  panels: $OUT_DIR/figure_<id>/"
