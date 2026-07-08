# Pernet 2015 — voice localizer

Reanalysis code for the Pernet voice-localizer dataset, producing the paper's
Pernet-side figures (AlKhamissi & Mehrer et al., 2026).

> **Pinned source index:** `20241003_pernet_2015/docs/OMNI_PAPER_PERNET_CODE_INDEX.md`
> @ commit **`f842b1a`** (2026-07-06). This README's allow/deny list tracks that
> frozen index; if the dev repo drifts, re-pin here (docs/DESIGN.md §10).

## Figures & lineages (allow-list — docs/DESIGN.md §2.4)

| Paper figure | Lineage (dev-repo scripts) | Golden master (docs/DESIGN.md §6) |
|---|---|---|
| Fig. 3b surface map | `00` GLM → `01` group → `02` surface | Fig-3b surface map: spatial-r vs reference |
| Fig. 3b 2-bar profile | `cv_04` group fROI → `cv_05` responses (CV; `cv_01`–`cv_03` = Stage 0) | `cv_responses.csv` scalars, `atol=1e-12` |
| Fig. B3b island Moran's I | `05` → `06` (via `core.spatial_stats`) | `island_morans_i_results.json` scalar ± tol |

## Precomputed cut

**Contrast maps** (`results/00_volumetric_GLM/`) + half-split CV GLMs
(`results/04_cross_validation/per_subject/`), MNI152 2mm. Pernet's FSL preproc runs
in a temp dir and hands smoothed BOLD to the Nilearn GLM **in memory** — no
preprocessed BOLD is ever saved (docs/DESIGN.md §1). So the cut is contrast-level: the CLI
uses `--results-root`, **not** `--derivatives-root`.

```bash
# default: reproduce from the precomputed cut
python make_figures.py --input-source precomputed --results-root <DIR>

# or regenerate the cut from the raw BIDS dataset first (Stage 0; FSL + nilearn 0.10.4, heavy)
python make_figures.py --input-source raw --raw-root <BIDS> --results-root <DIR>
```

## Golden masters — calibrated tolerances (docs/DESIGN.md §6, measure→freeze)

| Lineage | Fixture / reference | Frozen tolerance | Calibration note |
|---|---|---|---|
| Fig. 3b surface projection (`02`) | published `02_surface_projection/surface_data_fsaverage6.npz` (data-gated) | **exact** (atol=0), NaN-aware | `core.surface.vol_to_surf` (radius=0, kind=auto, n_samples=1) reproduces all six arrays (`t_map`/`z_map`/`thresholded` × lh/rh) to **max \|Δ\| = 0.0** under nilearn 0.12.1 vs the 0.10.4 reference. Projection is bitwise version-robust → belongs in `core`. |
| Fig. 3b group analysis (`01`) | published `01_group_analysis/{t,z,p}_map.nii.gz` (data-gated, **heavy** `PERNET_RUN_HEAVY=1`) | **`atol=1e-13`** on t/z/p maps (calibrated 2026-07-06) | `SecondLevelModel` GLM engine pinned to nilearn **0.10.4**. Calibrated on a bigmem node (SLURM 65405112): worst per-voxel max\|Δ\| = 1.11e-15 (machine epsilon), **bit-identical across nilearn 0.10.4 and 0.12.1**; frozen at 1e-13 (~90× headroom). Also asserts shape + corr>0.9999 + 4 FWE clusters + peak t≈15.76. |
| Fig. 3b CV responses (`cv_05`) | published `04_cross_validation/cv_responses.csv` (data-gated) | **`atol=1e-12`** on both bars (218 subj) | Extraction is pure nibabel+numpy (mean-in-mask, no nilearn) → version-robust; reproduces both columns for all 218 subjects to **max \|Δ\| ≈ 9.9e-17** (~1 ULP). Fed the *published* fROI masks, so it never re-runs the GLM. |
| Fig. 3b CV group fROI (`cv_04`) | published `04_cross_validation/group/half-{A,B}_{t_map,fROI_mask}.nii.gz` (data-gated, **heavy** `PERNET_RUN_HEAVY=1`) | **exact** fROI counts + Dice=1.0 (calibrated 2026-07-06) | Same `SecondLevelModel` engine as `01` (pinned nilearn **0.10.4**). Reproduces the published fROI counts EXACTLY (A=2185, B=2944 @ t>4.79, Δ0) with Dice=1.0, bit-identical across nilearn 0.10.4 and 0.12.1. Also asserts both folds, n=218, t-map corr>0.9999. |
| Fig. B3b island Moran's I (`05`) | `tests/fixtures/island_morans_i_results.golden.json` | `atol=1e-9` on I values; **exact** on `n_islands`/`n_vertices`/`df` | First green port reproduced every deterministic field to **max \|Δ\| ≈ 5.3e-15** under libpysal 4.8.1 / esda 2.5.1 — and, notably, under nilearn **0.12.1** reproducing a **0.10.4**-produced reference. Confirms `spatial_stats` is nilearn-version-robust (→ belongs in `core`). Permutation p-values are now **seeded** (`PERM_SEED=42` in `05`, threaded to `core.spatial_stats`) → reproducible, though still not golden-pinned. |
| Fig. B3b comparison chart (`06`) | model bars from `data/model_island_morans_i/*.json` (sha256-pinned) | model mean/SE per bar (`test_b3b_comparison.py`) | 3-bar chart (Non-Topo / Topo-Omni / Brain). Model bars = mean±SE over the vendored per-island distributions (Topo-Omni 0.575±0.042 n=79; Non-Topo 0.235±0.012 n=418) — the exact artifacts the published figure used. Chart is a visual artefact (not pinned); the numeric net is the per-bar mean/SE. |

Port status: **All three Stage-1 figure lineages — DONE**: Fig. 3b map (`01`→`02`),
Fig. 3b 2-bar profile (`cv_04`→`cv_05`), Fig. B3b island Moran's I (`05`→`06` 3-bar
chart). The version-robust golden masters (`02`, `cv_05`, `05`) are verified against
published references; the two GLM-engine masters (`01`, `cv_04`) are now **calibrated
and frozen** (`atol=1e-13` for the group maps; exact fROI counts + Dice=1.0), measured
bit-identical across nilearn 0.10.4 and 0.12.1 on a bigmem node.
**Stage 0 — DONE (faithful port):** the volumetric GLM (`00`) and the CV preproc
(`cv_01`/`cv_02`/`cv_03`, which produce the `per_subject/` cut) are ported to
`preprocessing/` and wired to `--input-source raw` (see `preprocessing/README.md`). Being
FSL-dependent + nilearn-0.10.4-pinned, Stage 0 is not golden-mastered — faithful port +
raw-dispatch smoke test only.

## Known fixes folded in during the port (docs/DESIGN.md §7)

- **Vendor `spatial_stats.py` → `core/`** — removes the absolute cross-repo import in `05`.
- Fix the stale "speech > nonspeech" docstring in `05`.
- **Fig. B3b drawn here** (`06`): vendor the model's *per-island* island-Moran's-I
  **distributions** into `data/model_island_morans_i/` (sha256-pinned; see
  `data/PROVENANCE.md`), compute the brain bar from `05`, and draw the full 3-bar chart
  (mean±SE model bars vs brain point estimate, t-test + Wilcoxon). This corrects a
  provenance error: the dev fixture `data/fig_b3b_model_island_morans_i.json` held
  orphaned point-estimate literals (0.594 / 0.126) from a *non-figure* dev-`05`
  computation that never matched the published chart (docs/DESIGN.md §7/§10).
- **Seed the island permutation p-values** (`05` `PERM_SEED=42`, threaded to
  `core.spatial_stats.compute_island_morans_i(seed=…)`) — previously unseeded/stochastic.

## Not ported (deny-list — index §7 / docs/DESIGN.md §8)

Global Moran's I (`03`), per-subject surface (`04`), the null-distribution / baseline
`06_baseline_*` + `06_individual_*` scripts (distinct from the ported
`06_island_morans_i_comparison` B3b chart), alternative multiple-comparison / mixture
methods, abandoned surface-GLM, backups.

## Data license

Edinburgh DataShare `10283/818` — **CC-BY 4.0** (redistribution permitted *with
attribution*; cite Pernet et al. 2015). We ship only derived contrast maps, not
subject data. One loose end: read the DataShare End-user Licence PDF (docs/DESIGN.md §10).

> **STATUS: Stages 0 + 1 complete** — all three figure lineages (`01`→`02`, `cv_04`→`cv_05`,
> `05`) ported, wired, and golden-mastered; Stage-0 preprocessing (`00`; `cv_01`–`cv_03`)
> ported to `preprocessing/` and wired to `--input-source raw` (faithful port +
> raw-dispatch smoke test — not golden-mastered; docs/DESIGN.md §6). The GLM-engine golden
> masters (`01`, `cv_04`) are calibrated and frozen under the pinned 0.10.4 env (see the
> golden-master table above), and the precomputed cut is hosted on OSF (public, DOI
> `10.17605/OSF.IO/EHRT6`), so an outside reviewer can reproduce every figure from a clean
> clone via `download_precomputed.py` — no token required.

## Implementation notes

Port-time implementation detail, retained for provenance. The numbered subsections
below keep their original `§N` anchors so a citation of the form "README, Implementation
notes §7" resolves.

### §1 Figure claims a reviewer checks

The manuscript makes three checkable Pernet-side claims; each is regenerable from the
shipped cut via `make_figures.py` and inspectable at every intermediate step.

| Paper figure | Claim | Lineage |
|---|---|---|
| Fig. 3b surface map | voice-selective cortex (FWE p<0.05) | `00 GLM → 01 group → 02 surface` |
| Fig. 3b 2-bar profile | cross-validated vocal > non-vocal | `cv_04 group fROI → cv_05 responses` |
| Fig. B3b | brain island Moran's I vs model | `05 island Moran's I → 06 3-bar chart` |

`make_figures.py` dispatches all three via `--figures {fig3b_map,fig3b_profile,figB3b_morans_i}`,
path-agnostic through `--results-root`. Every step exposes a side-effect-free
`compute()` / `extract_responses()` / `compute_bars()` returning the maps/DataFrame/dict,
so intermediates can be imported and inspected without running the disk-writing `main()`.

### §2 Determinism regimes

The code distinguishes numbers that are bitwise-portable across nilearn versions from
those pinned to the dataset's nilearn 0.10.4 GLM engine:

- **Version-robust** (`02` surface projection, `cv_05` CV responses, `05` island
  Moran's I) — pinned to ~1 ULP against published references; these live in `core/`.
- **GLM-engine, nilearn-0.10.4-pinned** (`01` group maps, `cv_04` group fROI) — frozen to
  exact reproduction (`atol=1e-13` on the group maps; exact fROI counts + Dice=1.0),
  measured bit-identical across nilearn 0.10.4 and 0.12.1 on a bigmem node.

The READMEs and `data/PROVENANCE.md` record every deviation from the dev repo, including a
corrected stale `speech_vs_nonspeech → vocal_vs_nonvocal` contrast label.

### §3 Port rationale (why values are pinned)

- **Fig. B3b model-bar provenance.** The dev fixture's point-estimate literals
  (`topo=0.594 n=29 / nontopo=0.126 n=120`) were never the published figure — they are
  orphaned literals from a *secondary, non-figure* computation in the untracked dev
  `src/05_island_morans_i.py`. The published Fig. B3b (dev `06`) is drawn from the
  **per-island distributions**:

  | Model | file (`data/model_island_morans_i/`) | n islands | mean I | SE |
  |---|---|---|---|---|
  | Topo-Omni | `topoomni_vocals_fwhm4.0_island_morans_I.json` | 79 | 0.574898 | 0.041570 |
  | Non-Topo | `nontopo_vocals_fwhm4.0_island_morans_I.json` | 418 | 0.235253 | 0.012036 |

  Both JSONs are vendored and sha256-pinned; `06` reproduces the mean/SE and
  `tests/test_b3b_comparison.py` locks them.

- **GLM golden masters** (`01`, `cv_04`) are calibrated and frozen —
  `test_group_analysis_golden.py` freezes `atol=1e-13`; `test_cv_group_froi_golden.py`
  pins exact fROI counts.

- **Environment pins.** `environment/analysis_env_pernet.yml` is the exact `conda env
  export` from the dev machine (python 3.9.19, numpy 2.0.2, scipy 1.13.1, nibabel 5.2.1,
  nilearn 0.10.4, scikit-learn 1.5.1, statsmodels 0.14.2, matplotlib 3.9.1, seaborn 0.13.2,
  esda 2.5.1, libpysal 4.8.1), trimmed of jupyter/aws/pybids tooling the figures don't
  import. numpy 2.0.2 + nilearn 0.10.4 is the real, tested combination.

- **Seeded permutations.** `core.spatial_stats.compute_island_morans_i` takes an optional
  `seed=` (default `None`, unchanged for other datasets); `05` sets `PERM_SEED=42` (distinct
  per hemisphere). esda 2.5.1's `Moran` draws from the global NumPy RNG, so this makes the
  permutation p-values reproducible; the deterministic Moran's I values are unaffected
  (`core/tests/test_spatial_stats_seed.py`).

### §4 Known gotchas and runnability limits

- **The cut is contrast-level, by design.** Unlike the sibling datasets there is no
  preprocessed BOLD to host — FSL hands smoothed BOLD to the GLM in memory — so the OSF
  tier is inherently contrast-level (`00_volumetric_GLM/` + `04_cross_validation/per_subject/`),
  matching what `config.PRECOMPUTED` declares.
- **Raw / Stage-0 needs site glue.** `--input-source raw` requires FSL on `PATH`; the SLURM
  emitter leaves env/FSL activation as placeholders (~30–60 min/subject × 218). The
  precomputed cut is the intended entry point.
- **Deprecated secondary fixture.** `05` still reads `data/fig_b3b_model_island_morans_i.json`
  for a secondary (non-figure) t-test; that fixture and t-test are superseded by the
  per-island model bars in §3 (docs/DESIGN.md §10).

### §5 Cross-dataset parity

Relative to the Marvi and Jung ports, Pernet leads on inspectable internals (the
`compute()` discipline) and on the environment lock (its `.yml` is a real `conda env
export`). It does not yet commit run-free reference figures; because all three figures are
deterministic and derive from pinned goldens (`02`, `cv_05`, `06`), they can be rendered
once and committed so the output is inspectable without a run.

### §6 Verification baseline

At port time `pytest Pernet_2015/ core/` ran 60 passed, 10 skipped — the skips being
cleanly data-gated / opt-in-heavy (`PERNET_RUN_HEAVY=1`). `06` was smoke-tested against a
real brain island-Moran's-I JSON and renders the 3-bar chart with the correct model bars.

### §7 Data-release hosting (two-tier model)

Pernet's FSL preproc is never written to disk (in-memory hand-off, §1 of the code index),
so its cut is already contrast-level — roughly 3 GB total, with no BOLD/anat to exclude or
deface.

**Tier 1 — host on OSF (~3 GB):**
- `results/00_volumetric_GLM/` — per-subject first-level **contrast maps** (218 subj, MNI
  2mm, ~5.5 MB/subj, ~1.2 GB). Feeds `01` group → Fig. 3b map, and the **brain bar** of Fig. B3b.
- `results/04_cross_validation/per_subject/` — half-split GLMs (~1.7 GB). Feeds
  `cv_04` → `cv_05` → Fig. 3b 2-bar profile.

Fig. B3b's **model bars** are vendored in-repo, not on OSF (§3).

**Tier 2 — regenerate Tier 1 from raw** (Stage 0 ported; needs FSL/SLURM glue):
1. Pull the 218-subject set from **Edinburgh DataShare `10283/818`** (CC-BY 4.0; cite
   Pernet et al. 2015).
2. `preprocessing/run_stage0.py` — FSL preproc → Nilearn first-level GLM (`glm`) + the CV
   split GLMs (`fold-split` / `cv-split`) → regenerates `00_volumetric_GLM/` +
   `04_cross_validation/per_subject/`. The engine is byte-faithful; SLURM/FSL activation is
   placeholder glue (~30–60 min/subject × 218).
3. Compare against the hosted Tier-1 maps: spatial-r (Fig. 3b map) / scalar
   (`cv_responses.csv`).

**Caveat.** Unlike Stage 1 (bit-identical across nilearn 0.10.4/0.12.1 — §2), a Tier-2 FSL
rerun matches Tier 1 only *within tolerance*, not bitwise.

**Read-set / cut layout.** The precomputed path re-runs the 218-subject group GLM, so
`fig3b_map` (`01` `SecondLevelModel`) and `fig3b_profile` (`cv_04`, 2× half-split fits)
re-fit and need **SLURM bigmem** (they OOM on a login node). The traced reads are exactly
the cut:
- `01` reads only `00_volumetric_GLM/sub*_Ed/sub*_Ed_contrast_estimates.nii.gz`;
- `cv_04` reads only `04_cross_validation/per_subject/sub*_Ed/half-{A,B}_contrast.nii.gz`;
- `cv_05` reads only `per_subject/...{vocal,nonvocal}_beta.nii.gz` (+ masks `cv_04` regenerates).

The 2026-07-06 calibration (SLURM 65405112) ran `01`+`cv_04` against this exact on-disk cut
and reproduced the published maps/counts, so a from-clean-folder SLURM rerun is the only
remaining reproduction step, not a correctness risk.

**Fig. B3b is not self-contained.** `05_island_morans_i.py` reads
`02_surface_projection/surface_data_fsaverage6.npz`, an **output of `02`** (the `fig3b_map`
lineage), not of the `figB3b` handler. So `make_figures --figures figB3b_morans_i` against a
Tier-1-only cut fails unless `fig3b_map` (heavy) ran first, or that 1.3 MB npz is shipped
alongside. The chosen resolution ships the npz as an opt-in extra so Fig. B3b reproduces on
a login node standalone; the alternative is to keep the cut strictly contrast-level and
always run `fig3b_map` before `figB3b`.
