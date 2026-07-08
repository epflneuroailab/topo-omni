# Marvi_2025/preprocessing — Stage 0 (fMRIPrep + CVS)

Stage-0 scripts that turn the raw BIDS dataset (OpenNeuro ds006179) into the fMRIPrep
derivatives + FreeSurfer recon-all + CVS `.m3z` warps that make up the precomputed cut.
**The `--input-source precomputed` path (the default) never runs any of this** — the cut is
downloaded from OSF. These scripts are only for `--input-source raw`.

Not golden-mastered: Stage 0 is containerized fMRIPrep/FreeSurfer, not bitwise reproducible
across runs/machines, so it is validated by provenance (right container version, expected
output spaces/files present) rather than a frozen golden (docs/DESIGN.md §2.5 / §6).

## Scripts (run in order)

| Step | Script | What |
|---|---|---|
| 1 | [`validate_bids_structure.py`](validate_bids_structure.py) | Scan the BIDS root; report per-subject tasks/runs and which subjects have complete EMFL (`effloc`, 5 runs) data. |
| 2 | [`fmriprep_single_subject.sbatch`](fmriprep_single_subject.sbatch) | fMRIPrep 24.0.1 + FreeSurfer 7.3.2 for one subject (6 output spaces + CIFTI). |
| 3 | [`submit_fmriprep.sh`](submit_fmriprep.sh) | Batch-submit step 2 over many subjects (staggered). |
| 16 | [`cvs_register_single_subject.sbatch`](cvs_register_single_subject.sbatch) | CVS registration (`mri_cvs_register`) for one subject → the `.m3z` warps. |
| 17 | [`submit_cvs_registration.sh`](submit_cvs_registration.sh) | Batch-submit step 16 over many subjects. |

(Step numbers match the dataset's dev pipeline; steps 4–15 are the Stage-1 GLM/ROI analysis
under [`../analysis/`](../analysis/), not preprocessing.)

## What CVS is, and why it's here

**CVS** (Combined Volume and Surface registration, FreeSurfer's `mri_cvs_register`) computes a
nonlinear subject→template warp, stored as a `.m3z` morph file. The atlas fROI parcels live in
template space; the pipeline needs them in each subject's native space, so it *inverts* these
warps (`mri_vol2vol --m3z … --inv-morph`) in [`../analysis/project_parcels_to_surface.py`](../analysis/project_parcels_to_surface.py).
Two registrations are produced per subject:

- `tocvs_avg35/final_CVSmorph_tocvs_avg35.m3z` — for the Julian parcels (FFA/OFA/fSTS/PPA/…)
- `tocvs_avg35_inMNI152/final_CVSmorph_tocvs_avg35_inMNI152.m3z` — for the VWFA / MD parcels

These `.m3z` warps ship inside the precomputed cut (`config.py` → `cvs_m3z_warps`), so a
reviewer on the default path gets them without running CVS at all.

## Parameters

No dataset paths are baked in — everything is passed via environment variables (and, for the
Python validator, `--bids-dir`). The fMRIPrep step needs `RAW_ROOT`, `FMRIPREP_IMAGE`,
`FS_LICENSE`; CVS needs `FREESURFER_IMAGE`, `FS_LICENSE`, `DERIVATIVES`. See each script's
header for the full list and an example invocation.

> **NOTE — where the heavy Stage-1 producers live.** The `--input-source raw` GLM producers a
> reviewer regenerates from raw BIDS (`first_level_glm` / `glm_splits`) live under
> [`../analysis/`](../analysis/), not here — Stage 0 is only the fMRIPrep/FreeSurfer/CVS
> preprocessing upstream of them. Both raw GLM paths need SLURM bigmem.
