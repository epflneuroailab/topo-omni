# Jung_2025/preprocessing — Stage 0 (raw BIDS → fMRIPrep fsaverage6 derivatives)

Faithful port of dev `download_raw_data/` + `fmriprep_jobs/`
(`20251211_fMRI_movie_watching_spacetop` @ `4066746`). Container-based, run once,
produces the **precomputed cut** the Stage-1 pipeline consumes:

- `*_hemi-{L,R}_space-fsaverage6_bold.func.gii` — the analysis input (40,962 vtx/hemi)
- `*_desc-confounds_timeseries.tsv` — the 24-confound source

**Not golden-mastered.** fMRIPrep + FreeSurfer are containerized and not bitwise
reproducible across runs/machines (docs/DESIGN.md §2.5/§6). Stage 0 gets a faithful port +
provenance spot-check (right container version, expected output spaces/files present).
The default `--input-source precomputed` path **never runs this** — it uses the shipped
derivatives (OSF) directly.

## Toolchain (pin at release)
- **fMRIPrep 24.0.1** (Singularity/Apptainer image, e.g. `fmriprep-24.0.1.simg`)
- **FreeSurfer 7.x** recon-all (bundled in the fMRIPrep container; needs a FS `license.txt`)
- Output spaces: **fsaverage6** (primary, analysed) + fsaverage5 (sanity) + anat.
  MNI152 volumetric and fsLR-91k CIFTI are **intentionally skipped** (~54 GB/subject saved;
  not used by any paper figure).
- Raw dataset: OpenNeuro **ds005256 v1.1.0** (CC0), task-alignvideo only.

## Files
| File | Purpose | Dev source |
|---|---|---|
| `download_raw_data.sh` | DataLad-get + unlock the alignvideo subset (BOLD/sbref/anat/fmap/meta); optional `PRUNE=1` drops other tasks | `download_alignvideo_batch.sh` + `remove_all_non_alignvideo_symlinks.sh` |
| `bids_filter_alignvideo.json` | restrict fMRIPrep to task-alignvideo | verbatim copy |
| `fmriprep_single_subject.sbatch` | one-subject fMRIPrep (surface-only output) | `00_fmriprep_single_subject_optimized.sh` |
| `submit_fmriprep.sh` | staggered submission over many subjects | `05_submit_first_20.sh` |

**Parameterized** (no baked-in dataset paths / subject batch files): pass `RAW_ROOT`,
`SINGULARITY_IMAGE`, `FS_LICENSE` (+ optional `OUTPUT_DIR`, `WORK_ROOT`,
`TEMPLATEFLOW_DIR`, `DELAY`) via environment. The dev "first_20 / batch1_30 / batch2_33"
convenience launchers are dropped.

## Typical flow
```bash
# 0. clone the raw DataLad dataset (once)
datalad clone https://github.com/OpenNeuroDatasets/ds005256.git  $RAW_ROOT

# 1. fetch + unlock the alignvideo subset (add PRUNE=1 to drop other tasks)
RAW_ROOT=$RAW_ROOT ./download_raw_data.sh sub-0001 sub-0002 ...

# 2. run fMRIPrep (surface-only) — staggered over subjects
RAW_ROOT=$RAW_ROOT SINGULARITY_IMAGE=/path/fmriprep-24.0.1.simg \
  FS_LICENSE=/path/license.txt ./submit_fmriprep.sh sub-0001 sub-0002 ...

# 3. Stage 1 then consumes $RAW_ROOT/derivatives:
#    python ../make_figures.py --input-source raw \
#      --derivatives-root $RAW_ROOT/derivatives --raw-root $RAW_ROOT
```

> The 83 canonical subjects are listed in `../config.py` (`CANONICAL_SUBJECTS`); the
> published analysis retains **78** after the confound-loader drop (docs/DESIGN.md §7).
