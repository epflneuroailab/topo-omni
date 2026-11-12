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

# --- Runs with a copied events file (ds005256 v1.1.0 data defect; found 2026-09-23) ---
# For each of these four KEPT subjects, the listed run's task-alignvideo events.tsv holds
# ONLY video rows (no rating rows), and those rows (onset, duration, stim_file) are identical
# to another run's (the value = that source run, whose own file is complete). The BOLD is
# the listed run (length = the canonical run's for sub-0021 / sub-0065; shorter for
# sub-0094 / sub-0105), so that run's true video timing is UNKNOWN. The published maps modeled these
# runs with the wrong videos (their six paper clusters kept n=78; 16 non-paper new54
# clusters lost one of these subjects). Remedy decided by the authors 2026-09-23: drop
# exactly these runs (opt-in `--exclude-broken-runs`; outputs under *_exclbrokenruns).
# A subject whose target videos fall ONLY in its excluded run still drops from that cluster
# (no data): for new27 that is cluster 8 (sub-0105) and cluster 14 (sub-0094).
# Pinned by tests/test_exclude_broken_runs.py (byte-copy check on the raw events).
BROKEN_EVENT_RUNS = {
    ("sub-0021", "ses-01", "run-03"): "run-04",
    ("sub-0065", "ses-04", "run-02"): "run-01",
    ("sub-0094", "ses-04", "run-01"): "run-02",
    ("sub-0105", "ses-02", "run-02"): "run-01",
}
EXCLUDED_RUNS_SUFFIX = "_exclbrokenruns"   # derivatives subdir suffix for the remedied analysis

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
    # (cluster_contrasts_new16clusters_exclbrokenruns/subject_level/, the 2 Fig. 6 clusters) rendered via
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

# --- Cluster families ---
# new54 = the arXiv v1/v2 family: its six brain-validation maps were SINGLE discovered
# clusters from the 54-cluster (new54) partition. Appendix D's "14 clusters" is a typo for 54; the 14-/21-/22-cluster
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
    # Rebuttal rerun (Sep 2026): the SAME cluster-vs-others fMRI validation on the partition
    # from the new Topo-Omni checkpoint (Badr's spacetop_discovery step-4 output,
    # `clusters_tvals_final.json` = raw dendrogram leaves, no hand merge). NOT a paper figure
    # (the paper uses its coarser cut new16); all 27 clusters are run (slurm/cluster_contrasts_array.sbatch) and tabulated
    # (analysis/summarize_family.py). Provenance: data/PROVENANCE.md.
    "new27": {
        "lineage": ["44", "45"],
        "model_json": f"{CLUSTER_ASSIGNMENTS_DIR}/27_cluster.json",
        "model_json_sha256": "5f2cd1b5959ee27c3f0bfb2b4ce2fa30d18affa6471d8115f9a49a1577437ead",
        "assignments_csv": f"{CLUSTER_ASSIGNMENTS_DIR}/cluster_assignments_new27clusters.csv",
        "derivatives_subdir": "cluster_contrasts_new27clusters",
        "n_clusters": 27,
    },
    # Rebuttal rerun, coarser cut (Sep 2026): same checkpoint and disjoint clips as new27, cut
    # higher in the dendrogram with min cluster size 60 (`clusters_tvals_disjoint_min60.json`).
    # Every new16 cluster is a union of whole new27 clusters (NEW16_FROM_NEW27). THE PAPER
    # FAMILY of the revised manuscript (Fig. 6: clusters 7 and 10, see FIGURES); compared with
    # new27 / new54 by analysis/compare_families.py.
    "new16": {
        "lineage": ["44", "45"],
        "model_json": f"{CLUSTER_ASSIGNMENTS_DIR}/16_cluster.json",
        "model_json_sha256": "d1b5bd5489f2396469d4c9020ff9fce3c3db5564a0e58eef066137e4f8427078",
        "assignments_csv": f"{CLUSTER_ASSIGNMENTS_DIR}/cluster_assignments_new16clusters.csv",
        "derivatives_subdir": "cluster_contrasts_new16clusters",
        "n_clusters": 16,
    },
}

# new16 cluster ID -> the new27 cluster IDs it is the union of (whole clusters; verified from
# the two JSONs, pinned by tests/test_new16_family.py). A single ID = the identical clip set,
# hence the identical GLM design (the design depends only on the target clips), so those maps
# must reproduce the new27 ones.
NEW16_FROM_NEW27 = {
    0: [0, 1], 1: [2, 3], 2: [4], 3: [5], 4: [6], 5: [7], 6: [8, 9], 7: [10, 11, 12],
    8: [13], 9: [14, 15], 10: [16, 17], 11: [18], 12: [19, 20], 13: [21], 14: [22],
    15: [23, 24, 25, 26],
}

# Published figure -> cluster family + IDs. The revised paper (Nature Communications, Nov 2026)
# shows the 16-cluster (new16) partition of the revised checkpoint, broken runs excluded:
# Fig. 6a = cluster 7 (television news interviews, 120 segments), Fig. 6b = cluster 10
# (natural landscapes, 155 segments); both n=78 / df=77. Labels are the paper's.
# The arXiv v1/v2 figures (new54 IDs 5/32/49 and 6/30/31, published runs kept) stay
# reproducible via analysis/cluster_contrasts.py --family new54 (README §"arXiv v1 maps").
#
# IDs are 0-BASED: the same integer indexes the model JSON `cluster_id`, the CSV
# `cluster_id`, the on-disk `group_cluster-NN` filenames and the paper's cluster numbers.
FIGURES = {
    "fig6": {
        "figure": "Fig. 6",
        "family": "new16",
        "exclude_broken_runs": True,
        "ids": [7, 10],
        "labels": {7: "television news interviews", 10: "natural landscapes"},
    },
}

# --- Visualization (dev 17_visualize_cluster_contrasts_v2.py) ---
FDR_Q = 0.05                    # one-tailed BH FDR
VIZ_TOP_PERCENTILE = 10.0       # top 10% of FDR survivors by t → the six published maps
