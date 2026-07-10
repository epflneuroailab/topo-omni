# Marvi 2025 — efficient multifunction localizer (EMFL)

Reanalysis code for the Marvi EMFL dataset, producing the paper's Marvi-side figures
(AlKhamissi & Mehrer et al., 2026).

> **Pinned source index:** `20251030_Marvi_2025_efficient_fMRI_localizer/docs/OMNI_PAPER_MARVI_CODE_INDEX.md`
> @ commit **`ef1da34`** (2026-07-06). Allow/deny list tracks that frozen index
> (docs/DESIGN.md §10).

> **STATUS: complete.** Branch A + Branch B are ported end-to-end and wired into
> `make_figures.py`; the golden-master suite passes (`84 passed` under the pinned
> analysis env). The precomputed data tier is hosted on OSF (public, DOI
> `10.17605/OSF.IO/EHRT6`), so an outside reviewer can run the pipeline from a clean
> clone via `download_precomputed.py` — no token required (docs/DESIGN.md §5.1).
> Detailed port history: the **Implementation notes** section below.

## Figures & lineages (allow-list — docs/DESIGN.md §2.4)

| Paper figure | Lineage (dev-repo scripts) | Golden master (docs/DESIGN.md §6) |
|---|---|---|
| Fig. A2 fROI profiles | Branch A: `06` → splits → frois → CV → extract → plot | `condition_responses_details_*.csv` means ± ε; fROI masks (Dice) |
| Figs. 2 & 3 surface maps | Branch B: `08` → `09` → `11` → `12/18` → `10/19` | Figs 2 & 3 maps: spatial-r vs reference |

Individual-subject **fsnative** surface (paper). Branch B uses the **concatenated
GLM (`08`)**, not per-run (`07`) — docs/DESIGN.md §7.

> **Reference renders** are not shipped in this code-only release. Fig. A2 is regenerated
> by `make_figures.py` from the pinned canonical CSV; the curated Figs 2 & 3 native-surface
> panels (interactive HTML, ~85 MB) are hosted on OSF alongside the precomputed cut. Minor
> cosmetic differences from the published panels are expected (Inkscape post-processing);
> the data should match.

> **Fig. 1b** is a graphical abstract — its bars restate Figs 2 (visual) & 4 (== Fig. A2
> language/MD). There is no separate Fig. 1b lineage; the panel ingredients come from the
> Branch A + Branch B renders and are assembled downstream.

## Precomputed cut

**fMRIPrep derivatives**: BOLD in 3 spaces (MNI 2mm / native T1w / fsnative) +
FreeSurfer recon-all + CVS `.m3z` warps. Resolves via `--derivatives-root`.

```bash
# Branch A (Fig. A2) + Branch B (Figs 2 & 3) from the precomputed derivatives:
python make_figures.py --input-source precomputed --derivatives-root <DIR>
```

**Anatomical parcels are vendored (standalone).** Branch A (fROI definition) needs the
Julian-lab anatomical parcels; the 6 used categories (~6.8 MB) are committed under
[`data/PARCELS/`](data/PARCELS/) (see `data/PROVENANCE.md`), so a fresh clone reproduces
Fig. A2 with no external atlas. `emfl.config.get_parcels_dir()` resolves `$MARVI_PARCELS_DIR`
→ `data/PARCELS/` and **hard-fails** if neither is present — it no longer silently falls back
to a dev-machine absolute path (which had masked a broken standalone build).

**Branch B default = published panels, not the full grid.** The native-surface render can emit
the full 6 subjects × 9 contrasts × L/R = **108-panel grid** — the matrix the paper's panels
were *curated from*. That is now **opt-in** (`--exhaustive`); by default it renders only the
panels published in Figs 2 & 3 (`visualize_native_surface.PAPER_PANELS`). Narrow further with
`--subjects`.

```bash
python make_figures.py --input-source precomputed --derivatives-root <DIR>              # paper panels (default)
python make_figures.py --input-source precomputed --derivatives-root <DIR> --exhaustive # full 108-panel grid
```

**Runtime — expect multiple hours.** Marvi is by far the slowest of the three datasets (Jung
and Pernet finish in minutes). Branch A (Fig. A2) re-derives and cross-validates ~108 fROIs for
each of the 6 subjects — thousands of `resample_to_img` calls — and `cross_validation` writes no
intermediate files, so a long-silent terminal is normal, not a hang. To spot-check faster,
restrict to one subject (`--subjects sub-kaneff01`) or run only the surface maps
(`--figures fig2_surface fig3_surface`, which skips the heavy Branch-A cross-validation).

**Raw path (two-part).** `--input-source raw` starts from the **fMRIPrep derivatives** and
regenerates the *GLM-level* cut from them + raw event TSVs (`--raw-root`) — the heavy nilearn
GLM (Branch A split GLM; Branch B concat GLM `08`), run via SLURM bigmem, not golden-mastered,
best-effort. It does **not** produce the fMRIPrep derivatives themselves: to build `<DIR>` from
raw scans first, run the containerized fMRIPrep + CVS jobs in
[`preprocessing/`](preprocessing/) (see [`preprocessing/README.md`](preprocessing/README.md)).

```bash
python make_figures.py --input-source raw --derivatives-root <DIR> --raw-root <BIDS>
```

## Reproducibility: what is guarded, and how tightly

Two kinds of validation, with different scientific weight (docs/DESIGN.md §6). **Frozen
goldens** run on every `pytest` (tolerance calibrated-then-frozen — a failure means "a
number moved"). **Spot-checks** are one-off validations of the heavy / non-deterministic
steps that cannot live in CI (they need SLURM bigmem or a FreeSurfer container) — a
snapshot, not a standing guarantee.

**Frozen CI goldens** (deterministic, light enough to run without a cluster):

| Step | What is pinned | Frozen tolerance (measured) |
|---|---|---|
| define_frois (A) | fROI masks vs published `frois/` | **Dice = 1.0** + exact voxel counts (22 masks) |
| cross_validation (A) | Dice, spatial-r + p, mean responses | `atol = 1e-6` (measured ≤ 8e-17); integer counts exact |
| extract_condition_responses (A) | per-condition β — **the Fig-A2 master** | `atol = 1e-9` (measured ≤ 2e-16; 300 β cells) |
| project_to_native_surface / 09 (B) | native-surface projections | `atol = 0.0` — **bitwise** (24 maps) |
| convert_inflated_surfaces / 11 (B) | inflated GIFTIs | **bitwise** (coords & faces max\|Δ\| = 0) |
| project_parcels_to_surface / 12 (B) | contour tracer + parcel-routing tables | 6 deterministic unit tests (numpy-only, no FreeSurfer) |

**One-off spot-checks** (heavy / non-deterministic — NOT in CI):

| Step | Checked once (subset) | Result |
|---|---|---|
| first_level_glm / 06 (A) | 1 subj/run, `faces_vs_objects` zmap | spatial-r = 1.0, max\|Δ\| = 0 (SLURM bigmem, nilearn 0.12.1) |
| concatenated_glm / 08 (B) | 1 subj visual, 5 contrasts | spatial-r = 1.0, max\|Δ\| ~1e-12 (SLURM bigmem) |
| project_parcels_to_surface / 12 (B) | 3 contours (CVS + affine paths) | Dice = 1.0, exact vertex sets (fMRIPrep-24.0.1 container) |

Render steps (`plot_figure_a2`; `visualize_native_surface` / 10) are **not**
golden-mastered — the numeric golden sits upstream and only the render is exercised
(e2e smoke). The goldens are meaningful only under the pinned env
(`environment/analysis_env_marvi.yml`: nilearn 0.12.1 / numpy 1.26.4).

> Golden tests are **data-gated**: they recompute from the on-disk derivatives and skip
> when those are absent. On a fresh checkout without the precomputed cut, most of the
> suite skips — this is expected, not a missing-tests signal (docs/DESIGN.md §6).

## Deliberate deviations from the dev pipeline (`ef1da34`)

The vendored `emfl/` package is otherwise a byte-faithful copy of the dev repo. Three
places break that on purpose (each carries an inline `RELEASE PORT NOTE`) — flagged here
so a reviewer diffing against the original knows these are intended, not silent edits.
**None change the published numbers.**

1. **Explicit events root** (`emfl/glm/first_level.py`, `__init__`). Dev located the raw
   event TSVs by `str(derivatives_dir).replace('derivatives','orig_data')` — a brittle
   sibling-path hack. The port adds an explicit `orig_data_dir` param (fed by
   `--raw-root`); `None` falls back to the dev behavior. **Behavior-preserving.**
2. **zmap restored to the volumetric GLM path** (`emfl/glm/first_level.py`,
   `_compute_zscore_contrasts` / `_volumetric_contrast_formulas`) — *the substantive one.*
   The dev volumetric engine saved `beta/tmap/pval` + per-condition `effect` but **dropped
   the z-score contrast**, yet every Branch-A reader (fROI definition / cross-validation /
   extraction) consumes `..._res-2_zmap.nii.gz` and the published split cut contains exactly
   `{zmap, effect}`. So the dev raw path could **not regenerate its own inputs** — it was
   structurally broken end-to-end. The port re-adds the z-score contrast + zmap save
   (mirroring the surface path, which always saved it). Verified **bitwise** vs the published
   `faces_vs_objects` zmap (spatial-r = 1.0, max\|Δ\| = 0) under nilearn 0.12.1. A fix that
   makes the pipeline runnable, not a change to results.
3. **Release default `--metric signed_log_p`** for the surface render (step 10). Dev
   defaulted to `t_fdr` (exploratory); the paper uses signed-log-p at ±3. Only the default
   changes — both metrics are preserved (reproduce-not-re-analyze, docs/DESIGN.md §2.7).

## Known fixes folded in during the port (docs/DESIGN.md §7)

- **Ship concat GLM (`08`), not per-run (`07`)** — matches the paper text; the
  `07 + aggregate_individual_subject_stats` route is not ported.
- **Individual-subject surface only** — no group-level EMFL surface map is in the
  paper; group scripts `14`/`15` + random-effects are not ported.
- `spatial_stats.py` (historically hosted here) moves to `core/` — single source of
  truth, also consumed by Pernet.

## Not ported (deny-list — index §7 / docs/DESIGN.md §8)

Exploratory fsaverage5/6 whole-brain surface pipeline, group fixed/random-effects,
per-run T1w route (`07`), single-video modelling, `archive/`.

## Data license

OpenNeuro `ds006179` — **CC0** (redistribution permitted). Be conservative on the
derivative tier: surface-space + defaced anat only (docs/DESIGN.md §5(b) / §10).

## Implementation notes

Port-time implementation detail, retained for provenance. This records the durable
rationale from the Marvi port: why values are pinned, the golden-master reproductions,
the environment pins, and the Fig. A2 / Branch B lineages. The section numbers below
match the citations in the code (e.g. a comment "README §6b" resolves to §6b here).

### §1 Key facts (design, parameters, environment)

**Subjects / design**
- 6 subjects: `sub-kaneff01, 06, 07, 08, 09, 21` (public subset of Marvi's 20).
- 5 EMFL runs each (`001..005`); task label in filenames = `task-effloc`. Derivatives
  also contain other tasks (`eploc, foss, lang, speech, spwm`) that are NOT EMFL — filter
  to `task-effloc`.
- Splits: `all=[1..5]`, `even=[2,4]`, `odd=[1,3,5]`.
- 9 contrasts (5 visual + 4 auditory) = the paper's Marvi Table 3:
  visual: faces/scenes/bodies/words `_vs_objects` + `objects_vs_words`;
  auditory: `false_belief_vs_false_photo, nonwords_vs_quilted, math_vs_theory_of_mind,
  english_vs_nonwords`. (The config comment says "6 visual" but only 5 are listed —
  `ALL_CONTRASTS` = 9; trust the list, not the comment.)
- GLM params: TR=2.0, smoothing FWHM=3mm, high-pass 0.01 Hz, drift = polynomial order 1,
  noise = AR(1), 6 motion confounds (trans/rot x/y/z), canonical (SPM) HRF.
- fROI: top **10%** t-voxels within each anatomical parcel; `MIN_FROI_VOXELS=10`.
- `PARCEL_CONTRAST_MAP` + `PARCEL_CATEGORIES` categories: julian / language / tom / md /
  speech / vwfa. vwfa = LH only.

**Golden-master reference**
- Canonical Fig. A2 CSV: `condition_responses_details_20260310_203831.csv` (+ matching
  summary). This timestamp is pinned as THE Fig. A2 golden master.
- Per-subject precomputed intermediates that serve as golden-master references:
  `first_level_glm/` (per-modality, per-split), `frois/` (per-parcel),
  `roi_cross_validation/*.csv`.

**Environment (pinned)** — nilearn 0.12.1, nibabel 5.3.2, numpy 1.26.4, pandas 2.3. The
goldens are valid only under this stack (see `environment/analysis_env_marvi.yml`).

**Compute caution** — the login-node per-user cgroup memory cap is 8 GB; heavy nilearn
GLM fits OOM there (exit 137). Run heavy fits via SLURM bigmem
(`--partition=bigmem --qos=bigmem`, ~32 GB). Marvi is only 6 subjects, so per-subject
first-level GLM is light; the Branch B concatenated GLM (5 runs pooled) is heavier —
watch memory.

### §2 Release file map and pipeline order

Branch A drivers, in pipeline order (the lineage cited from the analysis modules):
`06 first_level_glm` → `glm_splits` → `define_frois` → `cross_validation` →
`extract_condition_responses` → `plot_figure_a2`.
Branch B drivers: `08 concatenated_glm` → `09 project_to_native_surface` →
`11 convert_inflated_surfaces` → `12/18 project_parcels_to_surface` →
`10/19 visualize_native_surface`.

The `emfl/` engine is kept as an importable sub-package (7 interdependent modules:
`config`, `io/events`, `glm/first_level`, `roi/{definition,validation,extraction}`, plus
`utils`) rather than flattened — the modules are tightly coupled and a hand-rewrite risks
golden-master parity drift.

**Core-module decisions** — `core/spatial_stats.py` is shared (also used by Pernet);
Marvi's figures don't re-import the old path. The fROI CV math stays local to
`emfl.roi.{definition,validation,extraction}` (Marvi's top-10%-in-parcel / even-odd /
Dice+pattern-corr scheme diverges from Pernet's half-split scheme). `core/surface.py` is
NOT used for Branch B: step 09 uses scipy trilinear (not nilearn `vol_to_surf`) and step
10 uses `view_surf` with a Marvi-specific look — no clean shared kernel.

### §6b Implementation findings (engine internals)

**Events live in raw BIDS, not derivatives.** The GLM engine reads event TSVs from
`orig_data/sub-*/func/`. Effloc has **two** event files per run:
`task-efflocVisualConditions` and `task-efflocAuditoryConditions` (NOT `task-effloc`);
`filter_fixation()` drops `trial_type=='fixation'`. Consequence: Branch A needs raw event
TSVs even in `precomputed` mode (they are tiny → shipped with the precomputed cut). The
port replaces the dev `str(derivatives_dir).replace('derivatives','orig_data')`
sibling-path hack with an explicit events/raw root (`--raw-root`; `None` falls back to the
dev behavior).

**GLM engine** (`emfl/glm/first_level.py`, `EFMLOCFirstLevelGLM`).
- Volumetric path: nilearn `FirstLevelModel` (t_r from BOLD header, noise='ar1',
  hrf='spm', drift='polynomial' order 1, high_pass=0.01, smoothing_fwhm=3.0,
  minimize_memory=False). Fits per run × modality; computes per-condition effect_size maps
  AND per-contrast beta/tmap/pval. The fROI step reads `zmap`, so the driver must also
  save `z_score` — see the zmap deviation in "Deliberate deviations" above.
- Surface path: manual vertex-wise `OLSModel` per vertex, L/R separately (Branch B /
  fsnative).
- Contrast formulas (verbatim, needed for parity):
  visual: faces-objects, scenes-objects, bodies-objects, words_scr_objects-objects,
  objects-words_scr_objects.
  auditory: false_belief-false_photo, nonwords-quilted_speech,
  math-0.5*false_belief-0.5*false_photo, 0.5*false_belief+0.5*false_photo-nonwords.
  Condition regressor names: faces, scenes, bodies, objects, words_scr_objects /
  false_belief, false_photo, nonwords, quilted_speech, math.

**On-disk map/space layout** (per subject `first_level_glm/`): dirs
`effloc_{visual,auditory}` (all runs) + `..._split-{even,odd}`. Each `run-00N/` mixes MNI
(`_res-2`), T1w, fsaverage5/6 maps; Branch A uses MNI `_res-2_zmap` / `_res-2_effect`
only. Filename:
`{subj}_task-effloc_run-{run}_{modality}_{name}_space-{space}[_res-2]_{maptype}.nii.gz`.

**fROI definition** (`emfl/roi/definition.py` + `batch_define_frois.py`). On-disk layout
(the driver, not the library `save_froi`, is the source of truth):
`frois/{cat}_{parcel}/{subj}_{parcel}[_{hemi}]_space-MNI152NLin2009cAsym_split-{split}_froi.nii.gz`.
Algorithm: load parcel → `resample_to_img`(nearest) to functional grid → average zmaps
across split runs → `np.percentile(z_in_parcel, 100-10)` threshold → mask =
`z >= threshold` within parcel; saved float32. Pure nilearn resample + numpy. Hemisphere
rules: julian/language = lh+rh; tom midline {mmpfc,vmpfc,dmpfc,pc} = None (bilateral);
speech = None; vwfa = lh only. Both splits.

**Cross-validation** (`emfl/roi/validation.py` + `batch_cross_validation.py`). Loads even
& odd fROI masks; extracts responses cross-split (odd-mask→even-runs, even-mask→odd-runs);
computes Dice(even,odd), mean preferred-contrast response per split, and spatial pattern
correlation (pearsonr of parcel z-scores, even-avg vs odd-avg). Writes
`roi_cross_validation/{subj}_{roi_label}_{even_from_odd,odd_from_even}.csv` + a per-subject
summary. roi_label = `{hemi}_{parcel}` or `{parcel}`.

**Extraction** (`emfl/roi/extraction.py`, `ROIResponseExtractor`).
`extract_condition_responses(mask, run_split)`: per modality, glob per-condition effect
maps, `mean(effect_data[mask>0])` per (condition, run). Pure nibabel+numpy →
version-robust. `extract_roi_responses` is the same for contrast zmaps → `response` column.

**Canonical Fig-A2 CSV — schema + CV math** (`batch_extract_condition_responses.py`). For
each parcel/hemi, load even & odd fROI masks, then:
- `even_df = extract_condition_responses(even_mask, run_split='odd')` (even mask, ODD data)
- `odd_df  = extract_condition_responses(odd_mask,  run_split='even')` (odd mask, EVEN data)
- per condition: `mean_beta = (mean(even_df.beta) + mean(odd_df.beta)) / 2`.
CSV columns: `subject, parcel_category, parcel_name, hemisphere, roi_label, condition,
modality, even_beta, odd_beta, mean_beta`. Pure nibabel+numpy → version-robust golden.
This is THE Fig. A2 golden boundary — feeding published fROI masks + published effect maps
reproduces the CSV without re-running the GLM.
The extraction hemisphere rule is category-only (vwfa=lh, speech=bilateral, else lh+rh)
and differs slightly from define_frois: the ToM midline parcels are looked up as lh/rh,
miss on disk, and are silently skipped — so only tpj survives ToM. This reproduces the
published 50-ROI / 500-row-per-subject layout (16 julian + 12 language + 2 tom + 18 md +
1 speech + 1 vwfa).

**Fig-A2 plotting** is self-contained in the dev extraction driver (`create_figure4_plot`
= 3×5 fROI-category × condition grid → `figure4_replication_with_indiv.svg`). Aggregates:
Language = mean(ifg,mfg,anttemp,posttemp,ag); Frontal MD =
mean(supfrontal,midfrontal,medialfrontal,ifgop); Parietal MD =
mean(parietal,precentral). Baseline-shift to min condition; 68% CI ≈ SEM. Fixed condition
order: faces, bodies, scenes, objects, words_scr_objects, false_belief, false_photo,
nonwords, quilted_speech, math. The plot is render-dependent and NOT golden-mastered — the
numeric golden is the details CSV.

### §8 Vendoring and golden-master results

**Vendoring strategy** — the `emfl/` sub-package is a verbatim copy of the dev repo
(pinned @ `ef1da34`), then parameterized. Deny-listed modules were not vendored:
`emfl/glm/group_level.py` (group-level GLM) and the whole `emfl/visualization/`
(group-level `EFMLOCVisualizer`, which reads `group_level_glm/*.nc` and renders
fsaverage5 — the exploratory whole-brain path, not the paper's native-surface lineage).
The `group_level` import was pruned from `emfl/glm/__init__.py`.

**config.py** re-exports the analysis params from the vendored `emfl.config` (single
source of truth) and pins the canonical CSV timestamp `20260310_203831`.

**PARCELS vendored** — the used parcel subdirs are committed under `data/PARCELS/`: julian
(23), language (13), md (20), tom (7), speech (1), vwfa (1) = 65 NIfTIs, all md5-identical
to the dev source. `get_parcels_dir()` resolves the in-repo copy; the `MARVI_PARCELS_DIR`
env override is still honored.

**Frozen golden-master results** (all under the pinned env):
- define_frois: 22 masks reproduce the published `frois/` bitwise — Dice = 1.0 and exact
  voxel counts. `resample_to_img` under nilearn 0.12.1 is bitwise; do NOT add
  `force_resample=True` to the vendored `definition.py` (it changes behavior and would
  break parity; the nilearn-0.13 `force_resample`/`copy_header` FutureWarning is expected
  under 0.12.1). Re-calibrate if the env ever moves to nilearn ≥ 0.13.
- cross_validation: 11 ROIs reproduce the dev reference
  (`cross_validation_details_20251215_135301.csv`, 324 rows = 6 subj × 54 ROIs) to float
  precision — max scalar abs err 8.3e-17, max response abs err 4.4e-16; frozen
  `atol = 1e-6`, integer counts exact.
- extract_condition_responses: 10 ROIs / 300 beta cells reproduce the canonical CSV
  (3000 rows = 6 subj × 50 ROIs × 10 conditions) to max abs err 2.2e-16; frozen
  `atol = 1e-9`. Each beta is a pure nibabel-load + `mean(effect_data[mask>0])` on the
  fROI's own grid (no resampling, no nilearn) → version-robust.

**Faithful reproduction of an upstream quirk** — CV/extract log "fROI masks not found" for
the four ToM midline parcels (mmpfc/vmpfc/dmpfc/pc): define_frois writes their masks
without a hemi prefix, but CV/extract look for `lh_`/`rh_`-prefixed masks and drop them.
The dev published CSV also omits all four, so the reproduction matches exactly (26
parcels). This is reproduced faithfully rather than silently "fixed".

### §9 Branch B — native-surface findings

**Three dev-source corrections** (Branch B does not use the CLI tools its filenames
imply):
1. **09 is not `mri_vol2surf`.** Pure Python: reads pial GIFTI RAS coords → `inv(affine)`
   → voxel idx → `scipy.ndimage.map_coordinates(order=1, mode='constant', cval=0.0)` =
   trilinear at pial vertices (projfrac ≈ 0). No CLI/container/FS env → version-robust
   bitwise golden. (Parcel contours DO use `mri_vol2surf --projfrac 0.5` — a different
   sampler.)
2. **11 is not `mris_convert`.** Pure `nibabel.freesurfer.read_geometry` → GIFTI
   POINTSET(float32) + TRIANGLE(int32) → bitwise golden vs `anat/*_inflated.surf.gii`.
3. **10 default `--metric` is `t_fdr`** (BH FDR q=0.05, positive-t only), NOT the paper's
   signed-log-p. The paper render requires `--metric signed_log_p` (threshold ±3 =
   p<0.001); the figure driver passes it explicitly.

**Per-script**:
- **08 concatenated_glm** — concat all 5 runs → one design matrix, one `FirstLevelModel`
  per subj×modality in T1w volume (ar1 / spm hrf / poly drift order 1 / high_pass 0.01 /
  fwhm 3.0 / minimize_memory=False). Only emfl import = `emfl.io.events.get_effloc_events`.
  Reads T1w BOLD + 6 motion confounds + events from raw (parameterized via `--raw-root`).
  Contrasts inline (note `english_vs_nonwords = 0.5*fb+0.5*fp-nonwords`,
  `math = math-0.5*fb-0.5*fp`). Saves beta/tmap/pval +
  `signed_log_p = -log10(max(p,1e-300))·sign(t)` (inf→±300, nan→0). Heavy GLM → NOT
  golden-mastered; SLURM bigmem spot-check reproduced sub-kaneff01 visual, 5 contrasts ×
  {beta,tmap,signed_log_p} vs published at spatial r = 1.0, max|Δ| ~1e-12.
- **09 project_to_native_surface** — see correction 1. In:
  `concatenated_glm/…{tmap,pval,signed_log_p}` + `anat/{subj}_hemi-{L,R}_pial.surf.gii`.
  Out: `native_surface_projections/…hemi-{L,R}_{maptype}.func.gii` (float32). Bitwise
  golden: max|Δ| = 0 across 24 maps.
- **11 convert_inflated_surfaces** — see correction 2. In:
  `sourcedata/freesurfer/{subj}/surf/{lh,rh}.inflated`. Out:
  `anat/{subj}_hemi-{L,R}_inflated.surf.gii`. Bitwise golden (coords & faces max|Δ| = 0).
- **12/18 project_parcels_to_surface** — FreeSurfer CLI. For each parcel: MNI/CVS→T1w then
  T1w→pial-surface, threshold >0.5, build black boundary contour. Transform routing:
  julian→CVS (`mri_vol2vol --noDefM3zPath --m3z … --inv-morph --nearest`); vwfa,md→
  CVS-MNI152; language,speech,tom→affine (`mri_vol2vol --lta-inv talairach.lta --nearest`);
  overrides lh.vwfa→vwfa, rh.sts→julian. To surface:
  `mri_vol2surf --regheader {subj} --projfrac 0.5 --surf pial --noreshape`. Important:
  MNI→T1w must use `--lta-inv` (resamples), NOT `--regheader` (header-only, misplaces
  ROIs); `--regheader` stays correct for the already-native T1w→surface step. Stage-0-ish
  (needs a FreeSurfer container) → NOT golden-mastered; container spot-check reproduced 3
  contours (CVS + affine paths) bitwise, Dice = 1.0.
- **10/19 visualize_native_surface** — render. nilearn `view_surf` (+ optional
  `plot_surf_stat_map` PNG). Custom `black_viridis` cmap (0=black for contours), bg =
  binarized sulc. signed_log_p path: |data|<thr→NaN, thr=3.0, vmin=0 vmax=10. Views:
  visual→ventral, auditory→lateral, both hemis. NOT golden-mastered (render).

**Published Branch-B reference cut on disk** — `concatenated_glm/` (08),
`native_surface_projections/` (09), `native_surface_parcels/…parcel_contour.func.gii`
(12), `cvs_transforms/…final_CVSmorph_*.m3z` (16, Stage-0),
`{subj}/anat/…{inflated,pial}.surf.gii` + `_sulc.shape.gii` (11 + fMRIPrep),
`sourcedata/freesurfer/{subj}/…`.
