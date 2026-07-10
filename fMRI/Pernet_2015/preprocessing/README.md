# Pernet_2015/preprocessing — Stage 0 (FSL + Nilearn first-level GLM)

Faithful port of the dev-repo Stage-0 pipeline (`20241003_pernet_2015/src` @ `f842b1a`):
FSL preprocessing + Nilearn first-level GLM. FSL preproc runs in a **temp dir** and the
smoothed BOLD is handed to the Nilearn GLM **in memory** — no preprocessed BOLD is written
to disk (docs/DESIGN.md §1). The only on-disk derivative is the contrast map. The `precomputed`
path never runs this; `--input-source raw` does (via `make_figures.py`).

**Not golden-mastered** (env-pinned to nilearn **0.10.4** + FSL-dependent; not bitwise
reproducible) → faithful port + provenance spot-check + raw-dispatch smoke test only
(docs/DESIGN.md §2.5 / §6). See `tests/test_stage0_smoke.py` + the `raw` dispatch tests in
`tests/test_scaffold.py`.

## Modules (dev source → release file)

| Release file | Dev source | Role |
|---|---|---|
| `data_loader.py` | `data/load_data.py` | `Pernet2015DataLoader` (func/anat/motion `.mat`). **Path-agnostic:** `base_path` now required (raw BIDS root) — no baked-in dataset path. |
| `timing.py` | `data/timing.py` | stimulus order / events from `TVA_loc.txt` (verbatim). |
| `motion_correction.py` | `processing/motion_correction.py` | 6 motion params + derivs + Carling-2000 outlier regressors (verbatim). |
| `volumetric_glm.py` | `processing/volumetric_glm.py` | `VolumetricGLMAnalyzer`: slicetimer → mcflirt → flirt coreg → fnirt(MNI 2mm) → fslmaths(6 mm FWHM) → Nilearn `FirstLevelModel`. **Path params** thread through (no hard-coded dataset/TVA paths); numerics verbatim. |
| `cv_split.py` | `cv_01_define_fold_split.py` + `cv_02_split_glm_single_subject.py` | block half-split (seed=42) + per-subject half GLMs. RNG/design/contrasts verbatim. |
| `run_stage0.py` | `00_volumetric_glm_parallel.py` + `run_single_subject_glm.py` + `cv_03_submit_split_glm_slurm.py` | orchestrator `build_precomputed_cut()` (the `raw` back-end) + CLI (`glm`/`fold-split`/`cv-split`/`all`) + SLURM-array emitter. |

## What it produces (the precomputed cut)

```
<raw-root>/{subs, voice_localizer/TVA_loc.txt}
  └─ run_stage0.py glm         (00)     → <results-root>/00_volumetric_GLM/sub*/sub*_contrast_estimates.nii.gz
  └─ run_stage0.py fold-split  (cv_01)  → <results-root>/04_cross_validation/fold_split.json
  └─ run_stage0.py cv-split    (cv_02)  → <results-root>/04_cross_validation/per_subject/sub*/half-*.nii.gz
```

`00_volumetric_GLM/` feeds Fig. 3b map + Fig. B3b; `04_cross_validation/per_subject/`
feeds the Fig. 3b 2-bar profile. Both are what we ship on OSF for `--input-source precomputed`.

## Running

Regenerate the cut and continue to figures in one call:

```bash
python make_figures.py --input-source raw --raw-root <BIDS> --results-root <OUT>
```

Or run a Stage-0 step directly (single subject or SLURM array):

```bash
python preprocessing/run_stage0.py glm      --raw-root <BIDS> --results-root <OUT> --subject-id sub001_Ed
python preprocessing/run_stage0.py cv-split --raw-root <BIDS> --results-root <OUT> --slurm --dry-run
```

**Preconditions (site-specific, not shipped):** the pinned analysis env
(`environment/analysis_env_pernet.yml`, nilearn 0.10.4) must be active and **FSL must be
on `PATH`** (`FSLDIR` set). The dev repo sourced a `setup_fsl.sh`; that glue is
site-specific, so the SLURM emitter leaves env/FSL activation as marked placeholders.
Cost: ~30–60 min/subject (218 subjects → use the SLURM array).

STATUS: **ported** — faithful Stage-0 port, path-agnostic, raw dispatch wired + smoke-tested.
