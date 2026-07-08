"""Jung 2025 (Spacetop movie) — dataset config.

Naturalistic movie / cluster-discovery dataset (Jung et al., 2025). Precomputed cut =
**fMRIPrep fsaverage6 GIFTIs (L/R) + `desc-confounds` TSVs** (docs/DESIGN.md §1). Resolves
via `--derivatives-root`. GLM engine = the unified fsaverage6 engine (stays local:
`analysis/glm_engine.py` + `analysis/regressors.py`, pure numpy/scipy/nibabel — NO
nilearn at analysis time).

Port source: 20251211_fMRI_movie_watching_spacetop (commit 4066746). No in-repo
lockfile — used the shared neuromod_friends conda env (versions frozen in
environment/analysis_env_jung.yml).
"""
from __future__ import annotations

DATASET = "Jung_2025"

# --- Raw source (external, not hosted) — docs/DESIGN.md §5 ---
RAW_SOURCE = {
    "name": "OpenNeuro ds005256 v1.1.0",
    "url": "https://openneuro.org/datasets/ds005256/versions/1.1.0",
    "n_subjects_available": 83,
    "n_subjects_used": 78,     # see N78 note below — this IS the published analysis
    "license": "CC0",
}

# --- Reproduction pin: n=78 / df=77 (docs/DESIGN.md §7, §10 — DECIDED) ---
# `load_confounds` hard-requires all 24 named columns and drops 5 subjects whose runs
# legitimately have fewer cosine/tCompCor columns. ALL SIX PUBLISHED MAPS ARE n=78 —
# this drop is what produced the paper. We KEEP it (pinned by a characterization test)
# and correct only the paper text df=82 -> 77. Padding to n=83 is a *different* analysis
# and is NOT built (no --confounds-tolerant deviation).
N_SUBJECTS_PUBLISHED = 78
DF_PUBLISHED = 77
CONFOUND_DROPPED_SUBJECTS = ("0035", "0044", "0061", "0084", "0131")

# --- Acquisition / surface constants (dev src/glm_unified.py, create_regressors_unified.py) ---
TR = 0.46                       # seconds; multiband-8, 3T Prisma
SURFACE = "fsaverage6"
N_VERTICES_PER_HEMI = 40962     # fsaverage6

# --- Precomputed cut (what we ship on OSF) — docs/DESIGN.md §5.1 ---
PRECOMPUTED = {
    "kind": "fsaverage6_giftis_plus_confounds",
    "surface": "fsaverage6",
    "bold_glob": "{subject}/ses-*/func/*_hemi-{hemi}_space-fsaverage6_bold.func.gii",
    "confounds_glob": "{subject}/ses-*/func/*_desc-confounds_timeseries.tsv",
    "osf_guid": "dpeys",       # OSF component (umbrella ehrt6); https://osf.io/dpeys/
    "manifest": "data/precomputed_manifest.json",  # in-repo; path+sha256 per file (download_precomputed.py)
    # NOTE: the SHIPPED Tier-1 cut is the per-subject cluster-contrast t-maps
    # (cluster_contrasts_new54clusters/subject_level/, 6 published clusters) rendered via
    # make_figures --from-subject-maps — NOT the 1.5 TB fsaverage6 BOLD (docs/DESIGN.md §5.1-C). The
    # bold_glob/confounds_glob above describe the raw-path (--input-source raw) inputs only.
    "minimal_subset": None,    # Tier 1 IS the whole (small) cut; no further subset needed
}

# --- Canonical subjects (dev src/canonical_subjects.py) — the fixed 83 ---
# The published analysis iterates these 83; 5 fail in load_confounds (see N78 above) and
# are dropped, yielding the n=78 group maps. Keep this list verbatim.
CANONICAL_SUBJECTS: list[str] = [
    "sub-0001", "sub-0002", "sub-0003", "sub-0004", "sub-0005", "sub-0006", "sub-0008", "sub-0010",
    "sub-0013", "sub-0014", "sub-0016", "sub-0018", "sub-0019", "sub-0020", "sub-0021", "sub-0025",
    "sub-0026", "sub-0029", "sub-0031", "sub-0032", "sub-0033", "sub-0034", "sub-0035", "sub-0037",
    "sub-0038", "sub-0040", "sub-0043", "sub-0044", "sub-0046", "sub-0050", "sub-0051", "sub-0052",
    "sub-0053", "sub-0055", "sub-0058", "sub-0059", "sub-0060", "sub-0061", "sub-0062", "sub-0064",
    "sub-0065", "sub-0066", "sub-0069", "sub-0070", "sub-0075", "sub-0076", "sub-0077", "sub-0078",
    "sub-0079", "sub-0080", "sub-0083", "sub-0084", "sub-0086", "sub-0087", "sub-0088", "sub-0089",
    "sub-0090", "sub-0092", "sub-0093", "sub-0094", "sub-0095", "sub-0098", "sub-0099", "sub-0100",
    "sub-0101", "sub-0102", "sub-0104", "sub-0105", "sub-0106", "sub-0107", "sub-0109", "sub-0111",
    "sub-0112", "sub-0115", "sub-0116", "sub-0122", "sub-0126", "sub-0127", "sub-0129", "sub-0130",
    "sub-0131", "sub-0132", "sub-0133",
]
assert len(CANONICAL_SUBJECTS) == 83, f"expected 83 canonical subjects, got {len(CANONICAL_SUBJECTS)}"

# --- The 24 standard confounds (dev src/confounds_standard.py) ---
# Friston-24 motion (12) + aCompCor (5) + tCompCor (3) + discrete-cosine drift (4).
# load_confounds hard-requires ALL 24 in EVERY run — the source of the n=78 drop.
STANDARD_CONFOUNDS: list[str] = [
    "trans_x", "trans_y", "trans_z",
    "rot_x", "rot_y", "rot_z",
    "trans_x_derivative1", "trans_y_derivative1", "trans_z_derivative1",
    "rot_x_derivative1", "rot_y_derivative1", "rot_z_derivative1",
    "a_comp_cor_00", "a_comp_cor_01", "a_comp_cor_02", "a_comp_cor_03", "a_comp_cor_04",
    "t_comp_cor_00", "t_comp_cor_01", "t_comp_cor_02",
    "cosine00", "cosine01", "cosine02", "cosine03",
]
N_CONFOUNDS = 24
assert len(STANDARD_CONFOUNDS) == N_CONFOUNDS

# --- Cluster set (allow-list; the single published family) ---
# ALL six brain-validation maps are SINGLE discovered clusters from the 54-cluster
# (new54) partition. Appendix D's "14 clusters" is a typo for 54; the 14-/21-/22-cluster
# branches are NOT in the paper (authors Mehrer + AlKhamissi, Slack 2026-07-07 —
# README §4). The contrast is target-cluster-vs-all-other-clusters (rating TRs
# unmodeled). Ported drivers live in analysis/; the vendored CSV + model JSON are under
# data/ (see data/PROVENANCE.md).
CLUSTER_ASSIGNMENTS_DIR = "data/cluster_assignments"   # vendored (from Badr / topo-omni)
CLUSTERS = {
    "new54": {
        "lineage": ["44", "45", "47"],
        "model_json": f"{CLUSTER_ASSIGNMENTS_DIR}/54_cluster.json",
        "assignments_csv": f"{CLUSTER_ASSIGNMENTS_DIR}/cluster_assignments_new54clusters.csv",
        "derivatives_subdir": "cluster_contrasts_new54clusters",
    },
}

# Published figures -> new54 cluster IDs (single discovered clusters). Both figures draw
# from the one new54 family above; the semantic labels are the authors' (Slack 2026-07-07),
# not auto-derived. Cluster 49 (normativeprosocial videos) is the faces network — strong
# right IT near FFA — NOT the null talking-head clusters (angrygrandpa/harrymetsally).
#
# ⚠ IDs are 0-BASED (0..53): the same integer indexes the model JSON `cluster_id`, the CSV
# `cluster_id`, the on-disk `group_cluster-NN` filenames, and the paper's cluster numbers
# (no off-by-one). Full ID -> content -> figure map: data/cluster_assignments/CLUSTER_INDEX.md.
FIGURES = {
    "fig6_d4": {
        "figure": "Fig. 6 / Fig. D4",
        "ids": [5, 32, 49],
        "labels": {5: "animals", 32: "natural landscapes", 49: "faces"},
    },
    "figD5": {
        "figure": "Fig. D5",
        "ids": [6, 30, 31],
        "labels": {6: "planetearth", 30: "mountainbike", 31: "mountainbike"},
    },
}

# --- Visualization (dev 17_visualize_cluster_contrasts_v2.py) ---
FDR_Q = 0.05                    # one-tailed BH FDR
VIZ_TOP_PERCENTILE = 10.0       # top 10% of FDR survivors by t → the six published maps
