"""Pernet 2015 — dataset config (paths, subjects, contrasts, precomputed cut).

Voice localizer (Pernet et al., 2015). Precomputed cut = **contrast maps**
(`results/00_volumetric_GLM/`): Pernet's FSL preprocessing runs in a temp dir and the
smoothed BOLD is handed to the Nilearn GLM IN MEMORY — no preprocessed BOLD is ever
saved (docs/DESIGN.md §1). So there is NO `--derivatives-root`; the precomputed tier is
contrast-level and resolves via `--results-root`.

Port source: 20241003_pernet_2015/src/config.py (commit f842b1a).
STATUS: scaffold — values below are placeholders to be filled during the port.
"""
from __future__ import annotations

DATASET = "Pernet_2015"

# --- Raw source (external, not hosted) — docs/DESIGN.md §5 ---
RAW_SOURCE = {
    "name": "Edinburgh DataShare 10283/818",
    "url": "https://datashare.ed.ac.uk/handle/10283/818",
    "n_subjects": 218,
    "license": "CC-BY 4.0 (attribution: cite Pernet et al. 2015)",
}

# --- Precomputed cut (what we ship on OSF) — docs/DESIGN.md §5.1 ---
# Contrast maps + half-split CV GLMs; NOT subject BOLD (lowest-risk tier).
PRECOMPUTED = {
    "kind": "contrast_maps",
    "results_subdirs": ["00_volumetric_GLM", "04_cross_validation/per_subject"],
    "space": "MNI152 2mm",
    "osf_guid": "6uwzr",       # OSF component (umbrella ehrt6); https://osf.io/6uwzr/
    "manifest": "data/precomputed_manifest.json",  # in-repo; path+sha256 per file (download_precomputed.py)
}

# --- Analysis parameters (lifted from the dev pipeline @ f842b1a) ---
# Subjects are sub001_Ed .. sub218_Ed; each contributes one vocal>non-vocal contrast map.
N_SUBJECTS = 218
SUBJECT_TEMPLATE = "sub{:03d}_Ed"
SUBJECTS: list[str] = [SUBJECT_TEMPLATE.format(i) for i in range(1, N_SUBJECTS + 1)]
CONTRAST = "vocal_vs_nonvocal"     # the single localizer contrast (vocal > non-vocal)
SURFACE_TARGET = "fsaverage6"

# Group-analysis (step 01) constants — Pernet 2015 GRF-FWE.
FWE_THRESHOLD = 4.79               # t(1,217), p<0.05 FWE, n=218
CLUSTER_THRESHOLD = 10             # min cluster size (voxels)
PEAK_MIN_DISTANCE = 8.0            # mm between reported peaks

# Cross-validated fROI profile (Fig. 3b 2-bar; steps cv_04 -> cv_05).
# Block-level half-split (seed=42) fit per fold; fROI from one fold, responses from the
# other (no double-dipping). The precomputed CV cut = 04_cross_validation/per_subject/.
CV_SUBDIR = "04_cross_validation"
CV_FOLD_SPLIT_SEED = 42            # cv_01 block half-split (fold_split.json)
CV_FWE_THRESHOLD = FWE_THRESHOLD   # same GRF-FWE t>4.79 for the per-fold fROI masks

# --- Stage 0 (raw -> precomputed cut) layout — preprocessing/run_stage0.py ---
# `--input-source raw` regenerates the cut from the raw BIDS dataset (FSL + Nilearn GLM,
# pinned nilearn 0.10.4). These mirror the constants in preprocessing/run_stage0.py.
GLM_SUBDIR = "00_volumetric_GLM"                     # per-subject contrast maps (Fig. 3b map + B3b)
RAW_TVA_LOC_RELPATH = "voice_localizer/TVA_loc.txt"  # localizer stimulus order (under --raw-root)
RAW_SUBS_RELPATH = "subs"                            # subjects dir (anat + motion .mat, under --raw-root)

# Vendored model input for Fig. B3b (drawn here — docs/DESIGN.md §7, §10). See data/PROVENANCE.md.
FIG_B3B_MODEL_VALUES = "data/fig_b3b_model_island_morans_i.json"  # TODO(vendor): from topo-omni
