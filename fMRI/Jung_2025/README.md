# Jung 2025 — naturalistic movie / cluster discovery (Spacetop)

Reanalysis code for the Jung movie-watching dataset, producing the paper's Jung-side
figures (AlKhamissi & Mehrer et al., 2026).

> **Pinned source index:** `20251211_fMRI_movie_watching_spacetop/docs/OMNI_PAPER_JUNG_CODE_INDEX.md`
> @ commit **`4066746`** (2026-07-06). Allow/deny list tracks that frozen index
> (docs/DESIGN.md §10).

## Figures & lineages (allow-list — docs/DESIGN.md §2.4)

The revised paper (Nature Communications revision, Nov 2026) shows **two single discovered
clusters of the 16-cluster (`new16`) partition** of the revised checkpoint, with the four
runs whose events file is a copy of another run **excluded** (`--exclude-broken-runs`,
section "Runs with a copied events file" below). Both maps are n=78 / df=77.

| Paper figure | Family | Cluster ID | Content | Dev lineage |
|---|---|---|---|---|
| Fig. 6a | new16, broken runs excluded | 7 | television news interviews (120 segments, 3 news reports) | `44` → `45` → `47` |
| Fig. 6b | new16, broken runs excluded | 10 | natural landscapes (155 segments: mountainbike, freesolo, skiing) | `44` → `45` → `47` |

> **Cluster numbering is 0-based** — identical across the model JSON, the CSV, the on-disk
> `group_cluster-NN` filenames, `config.FIGURES`, and the paper's cluster numbers.

### arXiv v1/v2 maps (`new54`)

The preprint showed six single clusters of the **54-cluster (`new54`)** partition of the
earlier strip model: IDs 5/32/49 (old Fig. 6 / D4: animals, natural landscapes, faces) and
6/30/31 (old Fig. D5), with all runs kept. They remain reproducible (bitwise goldens below)
with `analysis/cluster_contrasts.py --family new54`; they are no longer paper figures.
Full ID → content table: [`data/cluster_assignments/CLUSTER_INDEX.md`](data/cluster_assignments/CLUSTER_INDEX.md).

Driver: `make_figures.py` (`fig6`). One local, pure-numpy/scipy/nibabel engine (**no
nilearn** at analysis time) for every family; families differ only in the
cluster-assignment CSV.

### Release file map (`analysis/`)
| Release module | Dev source | Role |
|---|---|---|
| `parse_clusters.py` | `44` | new54 model JSON + sub-0001 events → cluster-assignment CSV |
| `regressors.py` | `create_regressors_unified.py` (cluster path) | HRF-convolved target/others design regressors |
| `glm_engine.py` | `glm_unified.py` (cluster path) + `confounds_standard`/`canonical_subjects` | per-subject OLS GLM → group t-test → FDR; **hosts the n=78 drop** |
| `cluster_contrasts.py` | `45` | driver: one cluster of the new54 family, path-agnostic |
| `visualize_clusters.py` | `17_v2` (lean) + `47` | top-10%-of-FDR inflated HTML (viridis); Moran's/PNG/multi-variant dropped |
| `summarize_family.py` | new (Sep 2026) | one row per cluster of a family: max t, FDR survivors, n, content |
| `compare_families.py` | new (Sep 2026) | a family vs a reference family: clip overlap + t-map r / Dice per cluster pair; `--ids` / `--all-pairs` for targeted tables |

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
# default: reproduce the two Fig. 6 maps from the shipped subject-level t-maps (no raw, no fMRIPrep)
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

The cluster-assignment JSONs (new54, new27, new16) come from the topographic model
(`topo-omni`) — vendored under `data/cluster_assignments/`, pinned by hash in
`data/PROVENANCE.md`.

## Rebuttal rerun (Sep 2026): the new-model partition (`new27`)

The same cluster-vs-others validation, run on **every** cluster of the partition produced
by the new Topo-Omni checkpoint (27 raw dendrogram leaves; provenance in
`data/PROVENANCE.md`). Engine, design, confounds, subjects and thresholds are the
published ones, with one decided change: the four runs with a copied events file are
dropped (next section). Not a paper figure; the paper uses its coarser cut `new16`.

```bash
cd fMRI/Jung_2025 && mkdir -p slurm/logs
# 1. smoke test: one cluster, 3 subjects, to a scratch results root (~10 min)
sbatch --array=0 --time=01:00:00 --export=ALL,DERIV=<DERIV>,BIDS=<BIDS>,PY=<python>,\
RESULTS_ROOT=<scratch>,SUBJECTS="sub-0001 sub-0002 sub-0003" slurm/cluster_contrasts_array.sbatch
# 2. all 27 clusters (bigmem, ~2 h each, in parallel); broken runs excluded by default
sbatch --export=ALL,DERIV=<DERIV>,BIDS=<BIDS>,PY=<python> slurm/cluster_contrasts_array.sbatch
#    -> <DERIV>/cluster_contrasts_new27clusters_exclbrokenruns/
# 3. the six paper clusters under the same remedy (robustness check vs the published maps)
sbatch --array=5,6,30,31,32,49 --export=ALL,FAMILY=new54,DERIV=<DERIV>,BIDS=<BIDS>,PY=<python> \
    slurm/cluster_contrasts_array.sbatch
#    -> <DERIV>/cluster_contrasts_new54clusters_exclbrokenruns/
# 4. one row per cluster: max t, joint-hemisphere FDR survivors, n, content, best new54 match
python analysis/summarize_family.py --family new27 --exclude-broken-runs --results-root <DERIV> \
    --reference-family new54 --reference-ids 5 32 49 6 30 31
```

Reading the table:
- `n_fdr_survivors` counts vertices at or above the one-tailed BH-FDR q<.05 threshold
  computed jointly over both hemispheres — the threshold the figures use.
- `n_unmodeled_clips`: the published design marks as "others" only videos **not** in the
  target cluster, so clips of a target video that belong to other clusters are in neither
  regressor. Near zero for single-video clusters; large for mixed ones.
- `status` is `done` only if the group test used the expected 78 subjects. The engine drops
  a subject on any error. With the broken runs excluded, clusters 8 and 14 are expected at
  n=77 (next section); any other value means a task needs a look before rerunning.
- `excluded_runs` lists the dropped runs that belonged to subjects in that cluster's test.
- Each cluster writes its own `group_cluster-NN_…_summary.json` with the subject list and
  the excluded runs. The old shared `group_level/summary.json` is overwritten by every
  cluster and kept only for back-compat.

### The coarser cut (`new16`) and the comparison with new27 / new54

Badr's second partition (2026-09-24, `clusters_tvals_disjoint_min60.json` →
`data/cluster_assignments/16_cluster.json`) is the same run cut higher in the dendrogram
(min cluster size 60). It is an exact coarsening of new27 (`config.NEW16_FROM_NEW27`):
8 clusters are unchanged, 8 merge 2–4 new27 clusters. Because the design depends only on the
target clips, the 8 unchanged clusters must reproduce their new27 maps (`map_r` ≈ 1 in the
comparison, which warns otherwise). Same engine, remedy and thresholds as new27.

```bash
cd fMRI/Jung_2025
# 1. all 16 clusters (same resources as new27)
sbatch --array=0-15 --export=ALL,FAMILY=new16,DERIV=<DERIV>,BIDS=<BIDS>,PY=<python> \
    slurm/cluster_contrasts_array.sbatch
#    -> <DERIV>/cluster_contrasts_new16clusters_exclbrokenruns/
# 2. per-cluster table, best match among the six paper clusters
python analysis/summarize_family.py --family new16 --exclude-broken-runs --results-root <DERIV> \
    --reference-family new54 --reference-ids 5 32 49 6 30 31
# 3. comparisons: vs new27 (each new16 cluster vs the new27 clusters it contains), vs all 54
#    published new54 maps, and vs the six paper clusters under the same broken-runs remedy
python analysis/compare_families.py --family new16 --reference-family new27 \
    --exclude-broken-runs --results-root <DERIV>
python analysis/compare_families.py --family new16 --reference-family new54 \
    --exclude-broken-runs --results-root <DERIV> \
    --reference-group-dir <DERIV>/cluster_contrasts_new54clusters/group_level
python analysis/compare_families.py --family new16 --reference-family new54 \
    --exclude-broken-runs --results-root <DERIV> --reference-ids 5 32 49 6 30 31 \
    --out-prefix <DERIV>/cluster_contrasts_new16clusters_exclbrokenruns/compare_new54paper_exclbrokenruns
# 4. figures: top-10%-of-FDR inflated HTML for every cluster with FDR survivors
python analysis/visualize_clusters.py \
    --contrast-dir <DERIV>/cluster_contrasts_new16clusters_exclbrokenruns/group_level \
    --output-dir plots/cluster_contrasts_new16clusters_exclbrokenruns --figure-tag new16 \
    --cluster-labels <ID=LABEL ...>
# 5. the two FDR-surviving clusters (7, 10), each against the six paper maps and against the
#    new27 clusters they merge (7 = 10+11+12, 10 = 16+17), all pairs whether or not they
#    share clips; one table per cluster x reference
OUT=<DERIV>/cluster_contrasts_new16clusters_exclbrokenruns
for c in 07 10; do
  python analysis/compare_families.py --family new16 --ids $c --reference-family new54 \
      --exclude-broken-runs --results-root <DERIV> --reference-ids 5 32 49 6 30 31 \
      --all-pairs --no-plot --out-prefix $OUT/compare_c${c}_vs_new54paper_exclbrokenruns
  python analysis/compare_families.py --family new16 --ids $c --reference-family new27 \
      --exclude-broken-runs --results-root <DERIV> --reference-ids 10 11 12 16 17 \
      --all-pairs --no-plot --out-prefix $OUT/compare_c${c}_vs_new27_exclbrokenruns
done
```

Reading the comparison (`compare_<ref>.csv`, one row per pair sharing clips;
`compare_<ref>_per_cluster.csv`, one row per cluster; `compare_<ref>_map_r.png`):
- `relation`: `identical`, `a_contains_b` (the new16 cluster merges that reference
  cluster), `b_contains_a`, `partial`, or `disjoint` (no shared clips; only with
  `--all-pairs`).
- `map_r` is the whole-brain correlation of the two unthresholded group t-maps. Every
  cluster-vs-others map shares a common video-vs-video component, so `map_r` stays at
  about 0.6–0.8 even between clusters with no clips in common.
- `dice_top10` is the Dice of the top-10%-of-FDR-survivor masks, i.e. what the figures
  show. `dice_top10_all` uses the top 10% of all 81,924 vertices instead, so it is also
  defined when one map is null; the two agree within ~0.04 for the maps here. Both separate
  matching from unrelated maps far better than `map_r` (≈0.3 unrelated vs 0.7–1.0 matched).
- `dice_fdr` (the FDR-survivor masks) discriminates poorly: a strong map has 55–80k of the
  81,924 vertices above threshold, so any two strong maps overlap at Dice ≈0.85–0.97.
- A Dice against a null map (`ref_n_fdr` = 0) is 0 by construction, not evidence of a
  mismatch; use `map_r` or `dice_top10_all` there.
- `best_map_ref_id` is the most-correlated reference map over all reference clusters,
  whether or not they share clips.

### new16 results (runs of 2026-09-24/25, broken runs excluded)

All 16 clusters ran at n=78. Only two survive FDR (joint-hemisphere one-tailed BH, q<.05):

| new16 | content (new27 parts) | max t | FDR t | FDR survivors | HTML figures (`plots/cluster_contrasts_new16clusters_exclbrokenruns/`) |
|---|---|---|---|---|---|
| 7 | normativeprosocial 1+2+3 (10+11+12) | 4.42 | 1.77 | 65,379 | `new16_cluster07_normativeprosocial_{lh,rh}_top10pct_inflated.html` |
| 10 | freesolo + mountainbike + skiing (16+17) | 4.14 | 1.71 | 74,487 | `new16_cluster10_freesolo-mountainbike-skiing_{lh,rh}_top10pct_inflated.html` |

The other 14 are null; that includes new16 13, which absorbs paper clusters 5 (animals) and
6 (planetearth) and is identical to the null new27 21. The 8 unchanged clusters reproduce
their new27 maps exactly (max |Δt| = 0).

How the two clusters compare with the six paper maps and with their new27 parts. The
reference maps are the broken-runs-excluded reruns. r = correlation of the unthresholded
t-maps. Dice = top 10% of FDR survivors (figure definition) / top 10% of all vertices.
Written by step 5 above to
`<DERIV>/cluster_contrasts_new16clusters_exclbrokenruns/compare_c{07,10}_vs_{new54paper,new27}_exclbrokenruns.{csv,md}`.

| new16 7 vs new54 paper | r | Dice | | new16 10 vs new54 paper | r | Dice |
|---|---|---|---|---|---|---|
| 5 animals | .70 | .34 / .38 | | 5 animals | .80 | .55 / .57 |
| 32 natural landscapes | .66 | .34 / .39 | | 32 natural landscapes | **.95** | **.73 / .74** |
| **49 faces** | **1.00** | **.98 / .98** | | 49 faces | .57 | .29 / .32 |
| 6 planetearth | .70 | .32 / .38 | | 6 planetearth | .77 | .45 / .48 |
| 30 mountainbike | .67 | .35 / .39 | | **30 mountainbike** | **.95** | **.73 / .75** |
| 31 mountainbike | .67 | .35 / .39 | | **31 mountainbike** | **.96** | **.77 / .78** |

| new16 7 vs new27 | r | Dice | | new16 10 vs new27 | r | Dice |
|---|---|---|---|---|---|---|
| 10 normativeprosocial1 (part) | .89 | .53 / .57 | | 10 normativeprosocial1 | .63 | .34 / .38 |
| 11 normativeprosocial3 (part, null) | .83 | – / .52 | | 11 normativeprosocial3 (null) | .61 | – / .40 |
| **12 normativeprosocial2 (part)** | **.94** | **.67 / .69** | | 12 normativeprosocial2 | .63 | .36 / .38 |
| 16 freesolo (null) | .25 | – / .13 | | 16 freesolo (part, null) | .58 | – / .31 |
| 17 mountainbike+skiing | .63 | .33 / .37 | | **17 mountainbike+skiing (part)** | **.97** | **.81 / .81** |

- new16 7 is the paper's faces map (49): 120 of 49's 131 clips, r = 1.00, Dice .98.
- new16 10 carries the three mountainbike/landscape paper maps (it contains all their clips),
  r ≈ .95, Dice .73–.78, and is essentially new27 17; freesolo adds little.
- Unrelated pairs still correlate at r ≈ .57–.80 but reach only Dice ≈ .29–.55.

### Runs with a copied events file (ds005256 v1.1.0 data defect)

**The defect.** Four of the 78 analyzed subjects have one run whose `task-alignvideo`
`events.tsv` contains only video rows, identical in onset, duration and file to another
run's, and no rating rows. The other run's file is complete. Found 2026-09-23 while
explaining subject drops; pinned by `tests/test_exclude_broken_runs.py`.

| Subject | Broken run | Video rows repeat | Broken run's scan length vs. canonical |
|---|---|---|---|
| sub-0021 | `ses-01 run-03` | `run-04` | 1016 = 1016 |
| sub-0065 | `ses-04 run-02` | `run-01` | 926 ≈ 930 |
| sub-0094 | `ses-04 run-01` | `run-02` | 862 < 1266 |
| sub-0105 | `ses-02 run-02` | `run-01` | 1582 < 1855 |

The scans are the listed runs; only their events are wrong. So the true video timing of
these four runs is unknown. The raw behavioral logs are not in the public release.

**Impact on the published analysis.** The published maps modeled these four runs with the
wrong videos. In every cluster this puts some wrong timepoints into the "others" regressor
for these subjects: 4 of 1,014 runs. Where a cluster's target clips sat in the broken run,
the target was empty, the design singular, and the subject dropped. sub-0021's copied
`angrygrandpa` event also ends after its run-03 scan does, so its timepoints spill into
run-04 and collide with that run's target clips, which trips the overlap check. Result: 16 non-paper new54 clusters at n=77. All six paper clusters
kept n=78. A simulation from events and confound run lengths, without BOLD, reproduces
exactly which subject is missing from which cluster in the on-disk published results.

**Decision (authors, 2026-09-23): drop exactly these four runs.** Each of the four subjects
contributes its other 12 runs; runs are removed from events, run boundaries, confounds and
BOLD alike (`cluster_contrasts.py --exclude-broken-runs`, the SLURM script's default).

| Option | Subjects | Runs | Scan hours |
|---|---|---|---|
| Published (broken runs mis-modeled) | 78 | 1014 | 151.3 |
| **Drop the 4 broken runs (chosen)** | 78 | 1010 | 150.8 |
| Drop the 4 subjects everywhere | 74 | 962 | 143.7 |

Also considered: asking the Spacetop authors for the correct events files.

**What the remedy cannot fix.** A subject whose only showing of a cluster's videos is in
its broken run has no data for that cluster and still drops, now with the explicit message
"no target timepoints". Predicted with the exclusion:

| Family | Clusters at n=77 | All others |
|---|---|---|
| new27 | 8 (mediabias; sub-0105), 14 (universe; sub-0094) | n=78 (cluster 22 recovers sub-0021) |
| new16 | none (clusters 6 and 9 contain new27 8 and 14 and recover those subjects) | n=78 |
| new54 paper clusters 5/32/49/6/30/31 | none | n=78 |

**Reproducibility.** The published analysis is untouched: without the flag, the engine is
the published one and `make_figures.py` reproduces the paper maps bitwise. Remedied outputs
always go to a separate `*_exclbrokenruns` directory, so the two are never mixed.
`EXCLUDE_BROKEN_RUNS=0` runs the published variant with the SLURM script.

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
- **Open (2026-09-23):** four runs in ds005256 v1.1.0 have an events file that only repeats
  another run's video rows (see "Runs with a copied events file"). Worth reporting to the
  Spacetop authors, who may hold the correct files. The paper's six maps include these runs
  mis-modeled; the remedied rerun of those six clusters quantifies the effect.

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
