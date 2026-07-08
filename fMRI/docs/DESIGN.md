# Omni Paper — Unified fMRI Code Release: Design

**Paper:** AlKhamissi & Mehrer et al., 2026, *"Discovering Functionally Selective
Brain Regions with a Deep Topographic Multimodal Model"* (Nature Communications).

Architecture reference for the **human-fMRI** reanalysis code. Optional read — to
reproduce figures see the top-level `README.md`. Section numbers are stable anchors cited
throughout the code as `(docs/DESIGN.md §N)`.

**This document covers:** the goal and end state (§0), the three-dataset overview (§1),
the design principles (§2), the repository layout (§3), the shared `core/` package (§4),
the OSF data-hosting model and what each precomputed cut contains (§5), the
testing / golden-master strategy (§6), known fixes including the Jung n=78 reproduction
(§7), what is deliberately **not** ported (§8), the build order (§9), and the settled
design decisions (§10).

---

## 0. Goal & end state

Consolidates the fMRI code producing the paper's brain-side figures from three dev repos
into one release, rebuilt **characterization-first** (§2.6), shipping only what the paper
uses. Datasets: **Pernet 2015** (voice localizer), **Marvi 2025 / EMFL** (multifunction
localizer), **Jung 2025 / Spacetop** (naturalistic movie / cluster discovery); each ported
against its authoritative **code-to-figure index** (pinned by commit in the dataset
README). Built and tested standalone, it folds later into the modeling repo
(`https://github.com/epflneuroailab/topo-omni`), so `core/` and vendored model inputs are
self-contained and relocatable. **Scope lock:** only code producing paper results;
everything else is excluded (§8).

---

## 1. The three datasets at a glance

All three share one architecture: **Stage 0 preprocessing** (container-based, run once) →
**Stage 1 analysis** (Nilearn/Python) → **figures**, with a `--input-source
{precomputed, raw}` switch. They differ in GLM engine, target surface, and where the cut
sits.

| | Pernet 2015 | Marvi 2025 (EMFL) | Jung 2025 (Spacetop) |
|---|---|---|---|
| Paper figs | Fig. 3b map + 2-bar profile; Fig. B3b Moran's I | Fig. A2 fROI profiles; Figs 2&3 surface maps (+Fig. 1b human) | Fig. 6 / Fig. D4; Fig. D5 |
| Raw source (not hosted) | Edinburgh DataShare `10283/818` (218 subj) | OpenNeuro `ds006179` (6 subj) | OpenNeuro `ds005256` v1.1.0 (83 subj → **78 used**) |
| Stage 0 tooling | FSL 6.0.7 (+Nilearn) | fMRIPrep 24.0.1 + FreeSurfer 7.3.2 (+CVS warps) | fMRIPrep 24.0.1 + FreeSurfer 7.x |
| GLM engine (stays local) | FSL preproc→Nilearn, **volumetric MNI 2mm** | Nilearn, MNI 2mm (Branch A) + native T1w→fsnative (Branch B) | Nilearn unified engine, **fsaverage6** |
| **Precomputed cut** (shipped) | **contrast maps** — FSL/fMRIPrep output **never on disk** | **fMRIPrep derivatives** (BOLD 3 spaces + FreeSurfer + CVS `.m3z`) | **fMRIPrep fsaverage6 GIFTIs + confounds** |
| Surface target | fsaverage6 | fsnative (paper) | fsaverage6 |
| Uses island Moran's I? | **Yes** (Fig. B3b) — imports shared `spatial_stats.py` | Hosts `spatial_stats.py`; not in an EMFL fig | No |

Pernet's FSL preproc runs in a temp dir and hands smoothed BOLD to the Nilearn GLM **in
memory**, so its cut is contrast-level; Marvi/Jung ship actual fMRIPrep derivatives. A
per-dataset parameter, not a structural difference.

---

## 2. Design principles

1. **One template, three instances.** All dataset folders share one skeleton (§3); only
   content and the cut vary.
2. **`core/` stays minimal.** Only shared, stable utilities in `core/` (§4); GLM engines
   stay local (legitimately different — FSL-vol / Nilearn-MNI+T1w / Nilearn-fsaverage6).
3. **Two entry points everywhere.** `--input-source {precomputed, raw}` with `--raw-root`,
   `--derivatives-root`, `--results-root` (Pernet has no `--derivatives-root`; its cut is
   contrast-level).
4. **Port only paper lineages.** Each index's allow-list is ported; its "not used" section
   is the deny-list (§8).
5. **The testable boundary is `precomputed-cut → figure`.** Stage 0 (containerized
   fMRIPrep/FSL) is not bitwise-deterministic and Pernet's preproc is never on disk, so it
   is not golden-masterable — ported faithfully + provenance-spot-checked; Stage 1 is
   golden-master regression. Hence the precomputed tier is **load-bearing for correctness,
   not a convenience**.
6. **Characterization-first, not TDD, for the port.** Pin existing behavior; genuine TDD
   only for new code (`core/paths.py`; any opt-in loader-tolerance deviation — §7).
7. **Reproduce, don't silently re-analyze.** Default build reproduces the **published**
   results (e.g. Jung n=78, §7). Science-altering changes are separate, labeled deviations.
8. **Relocatable.** `core/` is self-contained; model-derived inputs are clearly-marked
   vendored fixtures, swappable for `topo-omni` outputs at merge.

---

## 3. Repository layout

```
fMRI/
├── docs/DESIGN.md               ← this file
├── README.md                    ← what this is, how to reproduce all figs
├── make_all_figures.py          ← META: runs all three Stage-1 pipelines → every paper fig
├── environment/                 ← pinned envs (Stage-1 stacks DIVERGE — §6):
│   ├── analysis_env_pernet.yml  ←   nilearn 0.10.4 / numpy 2.0.2 / py3.9
│   ├── analysis_env_marvi.yml   ←   nilearn 0.12.1 / numpy 1.26.4
│   ├── analysis_env_jung.yml    ←   nilearn 0.12.1 / numpy 2.2.5 / py3.10
│   └── stage0_*/                ←   Stage-0 toolchains (FSL 6.0.7; fMRIPrep + FreeSurfer)
├── core/                        ← shared pip package (pyproject.toml + pip install -e core/)
│   ├── spatial_stats.py         ← island Moran's I (single source of truth)
│   ├── surface.py               ← vol→surf projection + fsaverage/fsnative plotting
│   ├── paths.py                 ← --input-source / --*-root resolution
│   ├── froi_cv.py               ← cross-validated fROI math (Pernet & Marvi, if it factors)
│   └── tests/
├── Pernet_2015/                 ← preprocessing/ · analysis/ · config.py (cut = contrast maps)
│                                  · data/ (vendored + Fig B3b fixtures) · tests/ · make_figures.py · README.md
├── Marvi_2025/                  ← same skeleton (cut = fMRIPrep derivatives; Branch A + Branch B)
└── Jung_2025/                   ← same skeleton (cut = fsaverage6 GIFTIs + confounds; glm_unified engine)
```

Each dataset folder is one instance of the skeleton (preprocessing = Stage 0, analysis =
Stage 1). **`make_all_figures.py`** dispatches each dataset's `make_figures.py` in
`precomputed` mode by default, one child process per dataset with cwd set to the dataset
folder — so each bare `import config` resolves locally and each runs under its own
divergent Stage-1 env (`--python NAME=PATH`). `--derivatives-root <BASE>` maps to
`<BASE>/<Dataset>`; `--datasets`, per-dataset overrides, and `--dry-run` are supported;
each `make_figures.py` also runs standalone. `core/` ships a minimal `pyproject.toml` and
each dataset env runs `pip install -e core/` — the orchestrator deliberately does **not**
inject `PYTHONPATH` (that is what breaks on relocation).

---

## 4. `core/` — what goes in, what stays out

**In `core/` (shared, stable):**
- `spatial_stats.py` — island Moran's I. **The #1 consolidation win**: formerly Marvi
  hosted it, Pernet imported it cross-repo by absolute path, Jung carried a stale
  duplicate; now one copy.
- `surface.py` — `vol_to_surf` + fsaverage6/fsnative plotting (Pernet Fig 3b, Marvi Figs
  2&3, Jung Fig 6).
- `paths.py` — `--input-source` / `--*-root` resolution.
- `froi_cv.py` — cross-validated fROI math, shared by Pernet & Marvi only where it factors
  cleanly; where they diverge each stays local.

**Stays LOCAL:** GLM engines (different spaces/tools, intentionally separate); dataset
config; preprocessing/SBATCH scripts.

**nilearn-version constraint (from §6):** `core/` imports under **both nilearn 0.10.4
(Pernet) and 0.12.1 (Marvi/Jung)**. `spatial_stats.py`/`froi_cv.py` are pure numpy/scipy,
so version-robust; `surface.py` is a thin version-tolerant wrapper (nilearn's
`vol_to_surf`/plotting API changed across 0.10→0.12), tested under both.

---

## 5. Data & hosting strategy

**Two-tier data model (§5.1).** Raw and full per-subject preprocessed data are **not**
hosted (Marvi 120 GB, Jung 1.5 TB — infeasible on OSF).
- **Tier 1** — the small "late cut" of derived stat maps (~10 GB total) hosted on **OSF**,
  pulled by `download_precomputed.py`, consumed by `make_figures.py --input-source
  precomputed`. The default reproduction path.
- **Tier 2** — code only: download raw and regenerate Tier 1 locally, with Tier 1 as the
  **tolerance-checked reference** (`compare_to_reference.py`).

**What's in the cut, per dataset (§5.1-C).** Pernet ships **contrast maps** (FSL/fMRIPrep
preproc is never written to disk — in-memory hand-off); Marvi/Jung ship **fMRIPrep
derivatives**; Jung's cut is **subject-level t-maps (~185 MB)**, NOT the 1.5 TB fsaverage6
BOLD. Per-dataset inventories live in each dataset README. The cut is **load-bearing**
(§2.5): the fixed input that makes the golden masters meaningful, since Stage 0 is not
golden-masterable.

**Licensing:** Pernet `10283/818` is **CC-BY 4.0** (cite Pernet et al. 2015); Marvi/Jung
`ds006179`/`ds005256` are **CC0**. Raw stays on OpenNeuro/DataShare (cited by DOI); only
derivatives go to OSF. The full-res Marvi warpable T1w is excluded from the cut; Jung is
surface-only (no anat volume). Details in the dataset READMEs.

**Vendored model-input provenance (§7):** each vendored model file (Jung cluster JSONs;
Pernet Fig-B3b bars) records source commit + content hash in
`<dataset>/data/PROVENANCE.md`; the release regenerates the **vendored-fixture** version
until swapped for live `topo-omni` outputs at merge.

---

## 6. Testing strategy (characterization-first port)

This is a **port**: the primary net is **characterization / golden-master regression**
against the originals' outputs (not TDD — §2.6). Only testable boundary is
**precomputed-cut → figure** (§2.5).

**Tolerance policy — summary statistics over fragile per-vertex dumps** (a single-vertex
shift or Nilearn projection-default change breaks a vertex sample without telling you the
*science* moved). Per lineage, in order of preference: (1) **map-level summary** — spatial
correlation `r ≥ 0.999` (or Dice ≥ threshold for masks); (2) **pinned scalars** (fROI means
± ε; cluster t at named vertices; Moran's I ± tol); (3) **exact integer/string invariants**
where deterministic (subject counts, df, cluster IDs). Bitwise equality is claimed only
where genuinely deterministic. Thresholds are **calibrated then frozen** (measure →
freeze): the achievable floor under a possibly-different Nilearn is unknown until the first
faithful port runs green, so calibrate, record in the README, then freeze as the bar.

**Three Stage-1 envs, not one** — the dev stacks diverge across breaking boundaries:

| | Pernet | Marvi | Jung |
|---|---|---|---|
| nilearn | **0.10.4** | 0.12.1 | 0.12.1 |
| numpy | 2.0.2 | 1.26.4 | 2.2.5 |
| scipy | 1.13.1 | 1.11.1 | 1.15.3 |
| python | 3.9 | — | 3.10/3.12 |

Pernet 0.10.4 vs Marvi/Jung 0.12.1 is a breaking surface/GLM-API gap; each dataset gets
its own env pinned to the versions its golden masters were produced under ("within
tolerance" is unfalsifiable across machines otherwise), which is why `core/` is kept
version-robust (§4).

**Tiers.** *Tier 1 — Golden-master (Stage 1, the real net):* Pernet `cv_responses.csv`
scalars, `island_morans_i_results.json` (± tol), Fig-3b map (spatial-r); Marvi
`condition_responses_details_*.csv` (means ± ε), fROI masks (Dice), Figs 2&3 (spatial-r);
Jung per-cluster group t-maps (spatial-r) + **n=78 / df=77** as a hard invariant (§7).
*Tier 2 — Unit / TDD (new code):* `core.spatial_stats`, `core.surface`, `core.paths`; plus
the **Jung confound loader as characterization** — a test pinning the loader dropping
exactly {0035,0044,0061,0084,0131} → n=78. *Tier 3 — Integration:* one subject / one
contrast, precomputed → figure.

**Stage 0 — NOT golden-mastered** (not bitwise reproducible; Pernet writes no preproc to
disk): faithful port, provenance spot-checks (container version, expected output
spaces/files), and a smoke test that `--input-source raw` parses and dispatches. **Raw
path = best-effort, not CI-covered.**

---

## 7. Known fixes applied during the port

Reproduction fixes (align with the published result) are distinct from deviations (change
the science); the default build is reproduction only (§2.7).

- **Jung — reproduce n=78 / df=77 (not re-analyzed to n=83).** `load_confounds`
  hard-requires all 24 named columns and drops 5 subjects (0035, 0044, 0061, 0084, 0131)
  with fewer `cosine`/`tCompCor` columns. **All six published maps are n=78 — this drop
  behavior *is* what produced the paper.** The release keeps the n=78 loader (pinned by a
  characterization test) and corrects only the paper text df=82→77. Padding to n=83 is a
  different analysis; if wanted it is an opt-in labeled deviation (`--confounds-tolerant`)
  with its own TDD tests and outputs — never the default.
- **Stage-1 env pinned** per dataset — golden masters depend on the exact
  Nilearn/numpy/scipy stack (§6).
- **Marvi — ship concat GLM (08), not per-run (07)** (matches the paper text); **individual-
  subject surface only** (no group-level EMFL map is in the paper).
- **Pernet — `spatial_stats.py` vendored into `core/`** (removes the cross-repo import in
  `05`; stale docstring corrected); **Fig. B3b model bars** generated here from vendored
  model JSONs, swapped for live outputs at merge.

Open reconciliations (tracked, not blocking): Jung cluster-ID labelling (Appendix D text
vs Fig. D5); Marvi canonical CSV timestamp; **Fig. B3b provenance** — the vendored
`topo_omni_I`/`nontopo_I` (0.5939 n=29 / 0.1260 n=120) are hard-coded literals in an
untracked dev script and don't reproduce from the related per-island JSONs (which lack
island sizes), so bytes are pinned by sha256 in `Pernet_2015/data/PROVENANCE.md`; the fix
is to regenerate both from a known `topo-omni` run at merge. The brain bar and its golden
master are unaffected.

---

## 8. Not ported (the deny-list)

Everything in each source index's **"Scripts NOT used for the paper"** section is excluded
(global/alternative Moran's I variants, exploratory whole-brain pipelines, group
random-effects, abandoned routes, backups/launchers) — see each dataset README for the
per-dataset list. The internal `METHODS_DRAFT.md` files describe more than the paper
reports; the release tracks to the paper (Appendices A/B/D), not those drafts.

---

## 9. Build order

1. **Pernet first** — smallest, self-contained; exercises `core/spatial_stats` and
   `core/surface`.
2. **Marvi** — validates the `core/froi_cv` sharing decision and native-surface pipeline
   (Branch A fROI profiles, then Branch B surface maps).
3. **Jung** — largest; pins n=78 / df=77 as reproduction (§7).
4. **Meta script + top-level README** — `make_all_figures.py` orchestrates all three; the
   README documents how to reproduce every figure.
5. **Hosting** — carve the Tier-1 cut, upload to OSF, ship `download_precomputed.py`;
   licensing/ethics gate clears first, then a path-agnostic sweep.

Per dataset: port in index order; write the Tier-1 test for each stage before porting it.
The release is then handoff-ready for folding into `topo-omni`.

---

## 10. Design decisions & open questions

**Settled:**
- **Three Stage-1 envs, not one** — Pernet 0.10.4 vs Marvi/Jung 0.12.1 breaking gap (§6, §4).
- **`core/` is pip-installed & version-robust** — `pip install -e core/` per env, relocates
  cleanly into `topo-omni` (§3, §4).
- **Jung n=78 reproduction only** — characterization-pinned, paper text df=82→77; no n=83
  deviation built (§7).
- **Vendored-fixture model inputs** (Fig. B3b bars; Jung cluster JSONs) carry pinned commit
  + hash, regenerated from live `topo-omni` at merge (§7).

**Open:** does Pernet+Marvi CV math factor cleanly into `core/froi_cv.py`; does Marvi
Branch B need `cvs_transforms/` (24 GB) or can it consume already-projected surface maps
(preferred — §5.1-C); where `core/` lands after the `topo-omni` merge.
