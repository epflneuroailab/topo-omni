# Pernet_2015/analysis — Stage 1 (Nilearn)

The testable boundary: precomputed cut (contrast maps) → paper figures. Ported in
index order, each lineage pinned by a golden master (docs/DESIGN.md §6, §9).

- `01` group → `02` surface .............. Fig. 3b surface map ............. **PORTED**
- `cv_04` group fROI → `cv_05` responses .. Fig. 3b 2-bar profile (CV fROI) . **PORTED**
- `05` island Moran's I → `06` comparison . Fig. B3b (3-bar vs model) ....... **PORTED**

## Fig. 3b lineage (`01` → `02`)

```
<results-root>/00_volumetric_GLM/sub*/sub*_contrast_estimates.nii.gz   (218, MNI 2mm) ── ship on OSF
  └─ 01_group_analysis.py   second-level one-sample t-test + GRF-FWE (t≥4.79)
       → 01_group_analysis/{t_map,z_map,p_map,thresholded_t_map}.nii.gz + cluster_table.csv
  └─ 02_surface_projection.py   vol_to_surf → fsaverage6 (core.surface)
       → 02_surface_projection/surface_data_fsaverage6.npz            (→ Fig. B3b input)
         + fig3b_voice_selective_{left,right}_lateral.svg             (Fig. 3b)
```

Run both via the driver: `python make_figures.py --input-source precomputed --results-root <DIR> --figures fig3b_map`.

**Two different determinism regimes (docs/DESIGN.md §2.2, §6):**
- `02` surface projection is nilearn-version-robust — 0.12.1 reproduces the published
  0.10.4 npz **bitwise**; golden master asserts exact equality (`test_surface_projection_golden.py`).
- `01` is a dataset-specific GLM engine pinned to Pernet's nilearn **0.10.4**. Its golden
  master (`test_group_analysis_golden.py`) is heavy (218-subject fit, opt-in
  `PERNET_RUN_HEAVY=1`) and is **calibrated + frozen** at `atol=1e-13` on the t/z/p maps
  (measured max\|Δ\| = 1.11e-15, bit-identical across nilearn 0.10.4 and 0.12.1 on a
  bigmem node), plus the shape/corr/cluster/peak invariants.

## Fig. 3b 2-bar profile lineage (`cv_04` → `cv_05`)

```
<results-root>/04_cross_validation/per_subject/sub*/half-{A,B}_{contrast,vocal_beta,nonvocal_beta}.nii.gz  ── ship on OSF
  └─ cv_04_group_froi_analysis.py   per-fold second-level t-test + GRF-FWE (t>4.79)
       → 04_cross_validation/group/half-{A,B}_{t_map,fROI_mask}.nii.gz
  └─ cv_05_extract_responses_and_plot.py   mean beta in the *opposite* fold's mask (cross-validated)
       → 04_cross_validation/cv_responses.csv (golden)  +  cv_bar_plot.{svg,png}  (Fig. 3b)
```

Run both via the driver: `python make_figures.py --input-source precomputed --results-root <DIR> --figures fig3b_profile`.

The precomputed CV cut is the half-split per-subject GLMs (`per_subject/`); the upstream
`cv_01` fold-split (seed=42) and `cv_02`/`cv_03` per-subject split GLMs are **Stage 0**
(they produce that cut) — now ported in `../preprocessing/` (`cv_split.py` +
`run_stage0.py`; faithful, not golden-mastered).

**Same two determinism regimes:**
- `cv_05` extraction is pure nibabel + numpy (mean-in-mask, NO nilearn) → bitwise
  version-robust; reproduces the published `cv_responses.csv` to max|Δ| ≈ 8e-17, golden
  at `atol=1e-12` (`test_cv_responses_golden.py`). The golden feeds the *published* fROI
  masks, so it never re-runs the GLM.
- `cv_04` is the same `SecondLevelModel` GLM engine as `01`, pinned to nilearn **0.10.4**;
  its golden master (`test_cv_group_froi_golden.py`) is heavy (`PERNET_RUN_HEAVY=1`) and is
  **calibrated + frozen**: it reproduces the exact fROI voxel counts (A=2185, B=2944 @
  t>4.79) with mask Dice=1.0, bit-identical across nilearn 0.10.4 and 0.12.1.

## Fig. B3b lineage (`05` → `06`)

```
<results-root>/02_surface_projection/surface_data_fsaverage6.npz               (from 02)
  └─ 05_island_morans_i.py   island Moran's I of the voice-selective map (FDR q<0.05,
       min island 8 vtx, 999 perms, seeded PERM_SEED=42)  → core.spatial_stats
       → 03_spatial_analysis/island_morans_i_results.json  (golden: deterministic I fields)
  └─ 06_island_morans_i_comparison.py   3-bar chart: Non-Topo | Topo-Omni | Brain
       model bars = mean±SE over the vendored per-island distributions
       (data/model_island_morans_i/, sha256-pinned); brain bar = 05's point estimate;
       t-test + Wilcoxon (model dist vs brain mean)
       → 03_spatial_analysis/island_morans_i_comparison.{svg,png,pdf}   (Fig. B3b)
```

Run both via the driver: `python make_figures.py --input-source precomputed --results-root <DIR> --figures figB3b_morans_i`.

**Determinism:**
- `05`'s Moran's I values are deterministic and golden-mastered (`atol=1e-9`); the
  permutation p-values are now **seeded** (`PERM_SEED=42`, distinct per hemisphere) so
  they reproduce, though they remain not golden-pinned.
- `06`'s model bars come from the vendored per-island JSONs (Topo-Omni mean I=0.575,
  n=79; Non-Topo I=0.235, n=418) — the exact artifacts the published figure used. The
  chart is a visual artefact; `test_b3b_comparison.py` pins the numeric net (per-bar
  mean/SE + comparison direction). **Provenance fix**: these distributions supersede the
  orphaned point-estimate literals (0.594 / 0.126) in `data/fig_b3b_model_island_morans_i.json`,
  which came from a non-figure dev-`05` computation (see `../data/PROVENANCE.md`).
