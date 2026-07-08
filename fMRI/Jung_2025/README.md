# Jung 2025 — naturalistic movie / cluster discovery (Spacetop)

Reanalysis code for the Jung movie-watching dataset, producing the paper's Jung-side
figures (AlKhamissi & Mehrer et al., 2026).

> **Pinned source index:** `20251211_fMRI_movie_watching_spacetop/docs/OMNI_PAPER_JUNG_CODE_INDEX.md`
> @ commit **`4066746`** (2026-07-06). Allow/deny list tracks that frozen index
> (docs/DESIGN.md §10).

## Figures & lineages (allow-list — docs/DESIGN.md §2.4)

All six brain-validation maps are **single discovered clusters** from the one **54-cluster
(new54)** partition. (Appendix D's "14 clusters" is a typo for 54 — confirmed by the
authors, Mehrer + AlKhamissi, 2026-07-07; the 14-/21-/22-cluster branches are **not** in
the paper. Implementation notes §4.)

| Paper figure | Cluster IDs (new54) | Networks | Dev lineage | Golden master (docs/DESIGN.md §6) |
|---|---|---|---|---|
| Fig. 6 / Fig. D4 | 5, 32, 49 | animals · natural landscapes · **faces** (D4c) | `44` → `45` → `47` | group t/p/mean maps: max\|Δ\| + Pearson r |
| Fig. D5 | 6, 30, 31 | planetearth · mountainbike ×2 | `44` → `45` → `47` | group t/p/mean maps: max\|Δ\| + Pearson r |

> The **faces network is cluster 49** (`normativeprosocial` videos → right IT near FFA) —
> strong (maxT 4.38, ~64.7k FDR survivors), **not** the null talking-head clusters
> (`angrygrandpa`/`harrymetsally`). Semantic labels are the authors'; the auto-derived
> CSV labels are video-name-based.

> **Cluster numbering is 0-based (IDs 0–53)** — identical across the model JSON, the CSV,
> the on-disk `group_cluster-NN` filenames, `config.FIGURES`, and the paper's cluster
> numbers (no off-by-one; verified via the authors' "cluster 7 is the only non-sig one",
> which is our null `cluster-07`). Full ID → content → figure table:
> [`data/cluster_assignments/CLUSTER_INDEX.md`](data/cluster_assignments/CLUSTER_INDEX.md).

Driver: `make_figures.py` (`fig6_d4`, `figD5`). Both figures share one local,
pure-numpy/scipy/nibabel engine (**no nilearn** at analysis time) and the identical
cluster-assignment CSV, differing only in which cluster IDs are rendered, so the maps
reproduce the published derivatives near-bitwise (docs/DESIGN.md §4).

### Release file map (`analysis/`)
| Release module | Dev source | Role |
|---|---|---|
| `parse_clusters.py` | `44` | new54 model JSON + sub-0001 events → cluster-assignment CSV |
| `regressors.py` | `create_regressors_unified.py` (cluster path) | HRF-convolved target/others design regressors |
| `glm_engine.py` | `glm_unified.py` (cluster path) + `confounds_standard`/`canonical_subjects` | per-subject OLS GLM → group t-test → FDR; **hosts the n=78 drop** |
| `cluster_contrasts.py` | `45` | driver: one cluster of the new54 family, path-agnostic |
| `visualize_clusters.py` | `17_v2` (lean) + `47` | top-10%-of-FDR inflated HTML (viridis); Moran's/PNG/multi-variant dropped |

### Golden-master status (docs/DESIGN.md §6)
- **Parse CSV — BITWISE (green):** the new54 family reproduces the on-disk cluster CSV
  exactly (`tests/test_parse_clusters_golden.py`).
- **n=78 drop — characterization (green):** exactly `{0035,0044,0061,0084,0131}` drop, with
  the missing-confound reason documented (`tests/test_n78_confound_drop.py`).
- **Per-subject GLM — BITWISE (verified on SLURM):** per-subject t-maps reproduce the
  published `subject_level/` maps at **max|Δ| = 0.0, r = 1.0** under the pinned env
  (numpy 2.2.5 / scipy 1.15.3 / nibabel 5.3.2 / nilearn 0.12.1).
- **Group maps — BITWISE (frozen):** all six published clusters (5/32/49 + 6/30/31)
  reproduce the on-disk group `tstat/pval/mean` GIFTIs at **max|Δ| = 0.0, Pearson r = 1.0**
  under the pinned env — calibrated on SLURM bigmem 2026-07-07, tolerance frozen at
  `ATOL=1e-9`, `MIN_PEARSON_R=1−1e-9` (`tests/test_group_maps_golden.py`, gated
  `JUNG_RUN_HEAVY=1`). Harness: `tests/calibrate_heavy_golden.py` — see Implementation notes §3/§5.

## ⚠ Reproduction pin: n=78 / df=77 (docs/DESIGN.md §7 — DECIDED)

`load_confounds` hard-requires all 24 named columns and **drops 5 subjects**
(`0035, 0044, 0061, 0084, 0131`) whose runs legitimately have fewer
`cosine`/`tCompCor` columns. **All six published maps are n=78 — this drop is what
produced the paper.** We therefore:

- **KEEP the n=78 loader behavior**, pinned by a characterization test asserting
  exactly those 5 drop and documenting why;
- **correct only the paper text** df=82 → 77.

Padding the loader to n=83 is a *different analysis* that will not reproduce the
published maps — **not built** (no `--confounds-tolerant` deviation). docs/DESIGN.md §7/§10.

## Precomputed cut & running

**fMRIPrep fsaverage6 GIFTIs (L/R) + `desc-confounds` TSVs.** Resolves via
`--derivatives-root`. The GLM also needs run-relative video onsets from the raw
`task-alignvideo` `events.tsv` (`--raw-root`; defaults to the derivatives-root's parent).

```bash
# default: reproduce all six published maps from the shipped derivatives (no raw, no fMRIPrep)
python make_figures.py --input-source precomputed \
    --derivatives-root <DERIV> --raw-root <BIDS>

# regenerate the cluster CSVs from raw events first, then run Stage 1
python make_figures.py --input-source raw \
    --derivatives-root <DERIV> --raw-root <BIDS>

# GLM only (skip HTML rendering, which needs nilearn)
python make_figures.py --derivatives-root <DERIV> --raw-root <BIDS> --skip-visualize
```

**Raw path (two-part).** `--input-source raw` still starts from the **fMRIPrep derivatives**
`<DERIV>` — it re-derives the cluster CSVs from raw events and re-fits the per-subject GLM from
the fsaverage6 BOLD, but it does **not** produce `<DERIV>` itself. To build the derivatives from
raw scans first (DataLad fetch + containerized fMRIPrep), run the jobs in
[`preprocessing/`](preprocessing/) (see [`preprocessing/README.md`](preprocessing/README.md));
then point `--input-source raw` at the result. Stage 0 is containerized and not
bitwise-reproducible (docs/DESIGN.md §2.5/§6).

> ⚠ The group GLM loads 78×13-run fsaverage6 BOLD (~5 GB/subject) — run the full
> reproduction on a compute node, not a login node (see Implementation notes §3). Stage 0
> (raw download + fMRIPrep) lives under `preprocessing/` (container-based, faithful port,
> not golden-mastered).

## Model-derived inputs (vendored)

The new54 cluster-assignment JSON comes from the topographic model (`topo-omni`) —
vendored fixture under `data/cluster_assignments/`, pinned by commit + hash in
`data/PROVENANCE.md`. This release regenerates the **vendored-fixture** version of
Figs. 6 / D4 / D5; swapped for live `topo-omni` output at merge (docs/DESIGN.md §5).

## Not ported (deny-list — index §7 / docs/DESIGN.md §8)

The 14-cluster branch (App. D typo for 54 — see above), individual-video family, original
50-cluster + 21/22/supercluster sets, cluster-vs-rating family, Moran's I / IoU /
random-baseline machinery (`27`–`39`), full-program launchers. The repo's **stale
duplicate `spatial_stats.py` is dropped** (Moran's I is not in any Spacetop paper figure
— index §7).

## Data license

OpenNeuro `ds005256` — **CC0** (redistribution permitted). CC0 does not waive ethics:
ship surface-space + defaced anat only on the derivative tier (docs/DESIGN.md §5(b) / §10).

## Resolved & open items (index §8)

**✅ Resolved (authors Mehrer + AlKhamissi, 2026-07-07 — Implementation notes §4):**
- **All six maps are 54-cluster single discovered clusters.** Appendix D's "14 clusters"
  is a typo for 54; the 14-/21-/22-cluster branches are not in the paper and are dropped.
- **Faces control (Fig. D4c) = cluster 49** (`normativeprosocial`), strong (maxT 4.38,
  ~64.7k FDR survivors, right IT near FFA) — **not** the null talking-head clusters. It
  reproduces from the vendored new54 CSV (byte-identical to the dev original); no new-code
  vs old-code discrepancy exists (the earlier "faces null" was a mislabeling).

**Notes:**
- Model-JSON run provenance: the new54 assignment is an emergent output of the
  modeling-side discovery pipeline (`topo-discover/`); which specific `topo-omni` run
  produced it is not recoverable from the vendored fixture and lives on the modeling side,
  not this release (described in `data/PROVENANCE.md`).
- Precomputed-tier minimum subset; top-10%-of-FDR is the single published viz threshold.

> **STATUS:** Stage-1 ported and wired (54-cluster-only; Figs. 6/D4 + D5); parse + n=78 goldens green;
> per-subject GLM bitwise-verified; **group-map golden frozen (all six clusters bitwise, max|Δ|=0.0)**.
> Stage-0 preprocessing ported (faithful, not GM'd). Port-time detail: see Implementation notes below.

## Implementation notes

Port-time implementation detail, retained for provenance. Folded from the port working
notes; the section numbers §3–§5 below are stable citation anchors referenced from the
code and tests.

### Key facts (subjects, design, engine)

**Subjects / design.** 83 canonical subjects → 78 used after the confound-loader drop
(dropped: sub-0035, 0044, 0061, 0084, 0131; `df = 77`). TR = 0.46 s; 13 runs/subject
(ses-01×4, ses-02×4, ses-03×3, ses-04×2); fsaverage6 = 40,962 vtx/hemi.

**Drop mechanism (verified empirically).** `load_confounds` hard-requires all 24 named
confounds in *every* run. fMRIPrep legitimately emits fewer columns for some runs →
`ValueError` → caught by `run_subject_analysis`'s blanket `except → return (None, None)` →
subject dropped. Confirmed: sub-0035 ses-01_run-02 has only 2 `cosine` columns (missing
cosine02/03); sub-0044 ses-04 runs miss `t_comp_cor_02`. The on-disk
`subject_level/cluster-02` has exactly 78 dirs, the 5 named absent. This n=78 is the
published analysis — KEEP it (pinned by a characterization test; paper text corrected
df=82→77; no `--confounds-tolerant` n=83 deviation is built).

**The engine** (`analysis/glm_engine.py` + `regressors.py`) — pure
numpy/scipy/nibabel/pandas, no nilearn at analysis time:
- per-subject OLS GLM (`np.linalg.inv`) → per-subject contrast t-map (target − others) →
  group one-sample t-test (`scipy.stats.ttest_1samp`) → one-tailed BH-FDR q<0.05 → group
  `tstat/pval/mean` GIFTIs + `summary.json`.
- regressors: canonical double-gamma HRF (gamma(6) − gamma(16)/6, peak-normalized),
  `signal.fftconvolve(mode='same')`, z-scored. Cluster regressor = target-cluster
  snippet-TRs vs. all other video-TRs (rating TRs = unmodeled baseline).
- confounds: 24 (Friston-24 motion + 5 aCompCor + 3 tCompCor + 4 cosine), z-scored per
  column, NaN→0.
- Because no nilearn touches the analysis path, the port reproduces the on-disk group
  GIFTIs near-bitwise.

**Only the cluster-vs-others contrast is in the paper.** Not ported: video contrasts,
cluster-vs-rating, random baselines, Moran's I / IoU, alternate granularities.

**Cluster sets (verified row counts).** new54 CSV: 2572 rows, IDs 0–53. Published (single
discovered clusters): c5 = animals/snakes (61 rows), c32 = landscapes, c49 =
faces/`normativeprosocial` (131), c6 = planetearth (74), c30/c31 = mountainbike. Labels
are auto-derived from the dominant video (`derive_label`); the semantic reading of c49 as
"faces" is the authors'. **Indexing is 0-based** (IDs 0..53) and identical across the JSON
`cluster_id`, CSV `cluster_id`, on-disk `group_cluster-NN`, `config.FIGURES`, and the paper
— proof: the authors' "cluster 7 is the only non-sig one" ⇒ our `group_cluster-07` has 0
FDR survivors while 5/6 are strong. Full ID→content→figure table:
`data/cluster_assignments/CLUSTER_INDEX.md`. Vendored new54 JSON
(`20260607_supercluster_assignment_individual_clusters.json`) sha256 `882173ec…ffd1`, see
`data/PROVENANCE.md`; the source `topo-omni` model-run commit is still to be pinned.

### §3 — Golden-master plan (Tier-1 per lineage)

1. **parse CSV (cheap, bitwise):** `parse_clusters` fed the vendored new54 JSON + sub-0001
   events equals the on-disk `cluster_assignments_new54clusters.csv` exactly — pins the
   model-input → design seam.
2. **n=78 characterization (cheap, loader-only):** iterate all 83 subjects' confound TSVs
   through the ported `load_confounds` → exactly `{0035,0044,0061,0084,0131}` raise,
   documenting *why* (missing cosine/tCompCor). No BOLD, no GLM.
3. **group map (heavy, SLURM):** full engine on the on-disk fsaverage6 derivatives →
   reproduce the six published `group_cluster-{05,32,49}` (Fig. 6/D4) and
   `group_cluster-{06,30,31}` (Fig. D5) `tstat/pval/mean` GIFTIs. Metric: spatial Pearson r
   and max|Δ|, plus the n=78/df=77 invariant. Calibrate, then freeze.

**SLURM note.** The login node's per-user cgroup cap is ~8 GB; the group GLM loads
78×13-run BOLD → run on a compute node (`--partition=bigmem --qos=bigmem`); never retry
heavy fits on the login node.

### §4 — Cluster-ID resolution (authoritative figure → new54 mapping)

All six published brain-validation maps are single discovered clusters from the one
54-cluster (new54) partition (confirmed with the authors). The 14-cluster branch is dropped
entirely (Appendix D "14 clusters" is a typo for 54); the 21/22-cluster `_v2`/`_v3`
"separate_face_clusters" JSONs are irrelevant.

- **Fig. 6 / Fig. D4:** cluster 5 = animals (lateral PFC), cluster 32 = natural landscapes,
  cluster 49 = faces (D4c).
- **Fig. D5:** clusters 6, 30, 31 (30/31/32 are near-identical landscape maps; the authors
  picked 32 for D4 and 30/31 for D5).

The faces network is cluster 49 (`normativeprosocial`, 131 rows): prosocial
human-interaction videos → strong activity in right inferior temporal cortex near FFA. This
is **not** the null talking-head clusters (`angrygrandpa` #33–38, `harrymetsally` in #48 —
maxT ≲ 0.9, 0 FDR survivors) that an earlier label-based guess wrongly took for "faces."

On-disk group maps, one-tailed BH-FDR q<.05 survivors (df=77): c5 maxT 4.73 / c32 4.45 /
c49 4.38 / c6 4.17 / c30 4.32 / c31 4.32 — all six strong. There is no new-code/old-code
discrepancy for c49: the vendored new54 CSV is byte-identical to the dev original, the
per-subject GLM is bitwise, and the group step is deterministic.

Port consequence: 54-cluster-only. `config` carries one new54 family with
`fig6_d4 = [5,32,49]` and `figD5 = [6,30,31]`; the 14-cluster JSON, CSV, and parse
mode/tests are dropped. The model-run provenance (which `topo-omni` run produced the new54
JSON) is an emergent output of the modeling side and is not recoverable from the vendored
fixture (see `data/PROVENANCE.md`); the precomputed-tier minimum subset and the
top-10%-of-FDR viz threshold are documented above.

### §5 — Heavy golden: calibration & frozen tolerance

Reference environment (the stack the published maps were produced under, and the
golden-master pin): numpy 2.2.5 / scipy 1.15.3 / nibabel 5.3.2 / nilearn 0.12.1.

- **Per-subject gate:** the engine reproduces the on-disk `subject_level` t-maps bitwise
  (max|Δ| = 0.0, r = 1.0, all kept subjects × both hemis) under the pinned env, with the
  n=78 drop live (sub-0035 on cosine02, sub-0044 on missing tCompCor). Since per-subject is
  bitwise and the group step is pure `ttest_1samp` + FDR, the group maps are expected
  bitwise too.
- **Group calibration (SLURM bigmem, pinned env):** across all six clusters ×
  {tstat,pval,mean} × {L,R} (36 maps) the worst max|Δ| = 0.0 (bitwise, every map) and worst
  Pearson r = 1 − 2e-16 (float64 rounding). Per-cluster group maxT matches
  `CLUSTER_INDEX.md`; all n=78 / df=77. **Frozen:** `ATOL=1e-9`, `MIN_PEARSON_R=1−1e-9`
  baked into `tests/test_group_maps_golden.py` (gated by `JUNG_RUN_HEAVY=1`). Harness:
  `tests/calibrate_heavy_golden.py`. SLURM resources: `--partition=bigmem --qos=bigmem
  --cpus-per-task=4 --mem=52G`.

### Precomputed (render-only) path — gotcha

`make_figures.py --input-source precomputed` must **not** re-run the full 78-subject GLM
(that would read the excluded fsaverage6 BOLD). The precomputed path instead uses
`glm_engine.load_subject_tmaps()` + `run_full_analysis(reuse_subject_maps=True,
subject_maps_root=…)` to load the shipped per-subject t-maps (in `CANONICAL_SUBJECTS`
order, so the group summation is order-stable) and run only the deterministic group
ttest + FDR; threaded via `cluster_contrasts.py --from-subject-maps --subject-maps-root`.
**Float32 floor:** the shipped subject maps are float32 GIFTI, so re-deriving the group
maps from them matches the published `group_level/` to ~3.3e-6 on t (r = 1 − 6.3e-13), not
the 1e-9 the from-BOLD golden hits (which recomputes subject maps in float64) —
scientifically identical (thresholded figure unchanged), pinned by
`tests/test_render_only_from_subject_maps.py`.
