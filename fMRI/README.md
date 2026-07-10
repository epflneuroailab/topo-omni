# fMRI analyses — brain-side pipelines

Reanalysis code for the three human-fMRI datasets in:

> AlKhamissi & Mehrer et al., 2026, *"Discovering Functionally Selective Brain
> Regions with a Deep Topographic Multimodal Model"* (Nature Communications).

This directory holds the fMRI code that produces the paper's brain-side figures: the
three datasets (Pernet 2015, Marvi 2025, Jung 2025) each go from a precomputed fMRI cut
to the published figures. It is a clean, tested, path-agnostic pipeline — distinct from
the model-side analyses in the parent `topo-omni` repo, which run the trained model on
the same localizer stimuli. See **[docs/DESIGN.md](docs/DESIGN.md)** for the full design
rationale; this README is the operational entry point.

Each dataset's `make_figures.py` runs standalone and its Stage-1 figure lineages are
pinned by golden-master tests; `make_all_figures.py` orchestrates all three.

---

## What's here

| Folder | Dataset | Paper figures |
|---|---|---|
| [`Pernet_2015/`](Pernet_2015/) | Voice localizer (Pernet 2015) | Fig. 3b map + 2-bar profile; Fig. B3b Moran's I |
| [`Marvi_2025/`](Marvi_2025/) | Multifunction localizer / EMFL (Marvi 2025) | Fig. A2 fROI profiles; Figs. 2 & 3 surface maps |
| [`Jung_2025/`](Jung_2025/) | Naturalistic movie / cluster discovery (Jung 2025) | Fig. 6 / Fig. D4; Fig. D5 |
| [`core/`](core/) | Shared utilities (island Moran's I, surface plotting, path resolution) | — |
| [`environment/`](environment/) | Pinned conda envs (Stage-1 stacks **diverge** — one per dataset) | — |

Each dataset folder has the **same skeleton**: `preprocessing/` (Stage 0),
`analysis/` (Stage 1), `config.py`, `data/`, `tests/`, `make_figures.py`,
`README.md`.

---

## Two-stage architecture

Every dataset runs the same two stages, selected with `--input-source`:

```
Stage 0  preprocessing   (containerized fMRIPrep / FSL — run once, not reproducible bitwise)
   │
   ▼   precomputed cut  ← the shippable, testable boundary (docs/DESIGN.md §2.5)
Stage 1  analysis        (Nilearn / Python)  →  figures
```

- `--input-source precomputed` — **default & recommended.** Start from our hosted
  precomputed cut (download from OSF) and regenerate figures. No raw data, no fMRIPrep.
- `--input-source raw` — full pipeline from the raw BIDS dataset (external download +
  Stage 0). Best-effort, not CI-covered.

## Reproduce every figure

**`make_all_figures.py` renders figures from a precomputed cut that must already be on disk — it
does not download or preprocess anything.** So before you run it, get each dataset's cut in place
**one of two ways**:

1. **Download it from OSF** (default & recommended) with `download_precomputed.py` — public, no
   token, sha256-verified. *(No raw data, no fMRIPrep.)*
2. **Produce it yourself** from the raw scans via Stage-0 preprocessing (`--input-source raw`; see
   [Reproduce from raw data](#reproduce-from-raw-data-instead---input-source-raw) below).

Fresh clone → cut from OSF → figures.

**One command (recommended).** [`reproduce_precomputed_figures.sh`](reproduce_precomputed_figures.sh)
chains the two steps below and also fetches the authors' published renders into
`_reference_figures/` so you can compare yours side-by-side. Run from this `fMRI/` dir:

```bash
./reproduce_precomputed_figures.sh                 # all three datasets
./reproduce_precomputed_figures.sh Jung_2025       # one dataset (fast; login-node OK)
PYTHON=/path/to/python ./reproduce_precomputed_figures.sh   # pick the interpreter (render stack)
```

**Or the two steps manually.** Run from this `fMRI/` dir:

```bash
python download_precomputed.py --dest .          # STEP 1 — pull each cut from OSF into ./<Dataset>/
python make_all_figures.py --derivatives-root .  # STEP 2 — render (needs STEP 1 done first)
```

**Figures land in each dataset's `plots/` dir (`./<Dataset>/plots/`)** — the default figure output,
independent of where the cut lives. The cut itself and all intermediate stat maps stay under
`--derivatives-root`. The bulky cut files (`*.gii`, `*.nii.gz`, …) are `.gitignore`d, so the working
tree stays clean. (The OSF components are public — no token needed.) Pernet's Fig-3b load is
memory-heavy — run it on a compute node with several GB of RAM, not a memory-capped login node.

**Runtime.** **Jung** (minutes) and **Pernet** (minutes, on a bigmem node) are fast. **Marvi is
by far the slowest — allow multiple hours.** Its Branch-A Fig-A2 lineage re-derives and
cross-validates ~108 fROIs for each of the 6 subjects (thousands of `resample_to_img` calls),
and `cross_validation` emits no intermediate files, so a long-quiet terminal is normal — not a
hang. Running the three datasets in parallel (one per shell/node) is fine. To spot-check Marvi
faster, restrict it — one subject (`--subjects sub-kaneff01`) or only the surface maps
(`--figures fig2_surface fig3_surface`, which skips the heavy fROI cross-validation).

**Compute resources.** The Stage-1 pipelines are essentially **single-threaded** (the nilearn
GLMs run `n_jobs=1`), so **RAM is the binding resource, not core count** — extra cores mainly let
you run the three datasets concurrently. RAM peaks on **Pernet** (its Fig-3b group step loads the
218-subject contrast set and fits a `SecondLevelModel`; measured peak **≈ 9.5 GB**). Recommended
per-dataset allocation for the precomputed reproduction:

| Dataset | RAM | Cores | Wall-time |
|---|---|---|---|
| Jung_2025 | ~8 GB | 1–2 | minutes (login-node OK) |
| Pernet_2015 | **≥ 16 GB** (peak ≈ 9.5 GB) | 1–2 | minutes (needs a non-tiny node — OOMs on a capped login node) |
| Marvi_2025 | ~8–16 GB | 1–4 | **multiple hours** (Branch-A fROI CV) |
| **All three at once (one node)** | **≥ 32 GB** (48 GB comfortable) | **4–8** | dominated by Marvi |

Validated end-to-end on a single compute node with **80 GB / 20 cores** (a generous allocation,
not a minimum). Cores beyond a handful are only useful for running datasets in parallel.

Point the cut anywhere with `--derivatives-root <CUT_BASE>` (the cut is then read from
`<CUT_BASE>/<Dataset>/` — e.g. a tmp/scratch dir, so intermediates never touch the repo); figures
still land in `./<Dataset>/plots/`. To send figures elsewhere, run a dataset's `make_figures.py`
directly with its `--plots-root`.

`make_all_figures.py` **dispatches each dataset's `make_figures.py` as a subprocess**
(one per dataset, run in its own folder). Subprocess — not in-process import — because
each dataset does a bare `import config` (name collides across the three) and the three
Stage-1 envs diverge (nilearn 0.10.4 vs 0.12.1), so each dataset needs its own
interpreter (`--python NAME=PATH`, default: this interpreter).

It expects each dataset's precomputed cut under `<CUT_BASE>/<Dataset>/` and maps it to
that dataset's own cut flag — **Marvi/Jung use `--derivatives-root`, Pernet uses
`--results-root`** (its cut is contrast-level). Point a dataset at an arbitrary path with
`--root NAME=PATH`; preview the exact commands without running via `--dry-run`. Each
`make_figures.py` is also runnable standalone from within its folder.

## Reproduce from raw data instead (`--input-source raw`)

The precomputed cut above is the recommended entry point. If you'd rather rebuild it yourself
from the public raw scans (OpenNeuro / Edinburgh DataShare), use `--input-source raw`. **How
many steps this takes depends on the dataset's Stage 0** — because `--input-source raw` on each
`make_figures.py` regenerates the *GLM-level* cut and figures, but for Marvi & Jung it starts
from **fMRIPrep derivatives**, which you must produce first:

| Dataset | Stage 0 — raw scans → fMRIPrep derivatives | Stage 1 — derivatives → figures |
|---|---|---|
| **Pernet** | folded into the one command below (FSL preproc runs in-memory, nothing to pre-run) | `make_figures.py --input-source raw --raw-root <BIDS> --results-root <DIR>` |
| **Marvi** | run the container jobs in [`Marvi_2025/preprocessing/`](Marvi_2025/preprocessing/) (fMRIPrep **+ CVS**) → `<DERIV>` | `make_figures.py --input-source raw --derivatives-root <DERIV> --raw-root <BIDS>` |
| **Jung** | run the container jobs in [`Jung_2025/preprocessing/`](Jung_2025/preprocessing/) (DataLad fetch + fMRIPrep) → `<DERIV>` | `make_figures.py --input-source raw --derivatives-root <DERIV> --raw-root <BIDS>` |

So for **Marvi and Jung the raw path is two-part**: first run the manual, containerized Stage-0
jobs (SLURM; see each `preprocessing/README.md`), then point `--input-source raw` at the
resulting `<DERIV>`. For **Pernet it's a single command** — its Stage 0 is lightweight enough to
run inside `make_figures.py`. Stage 0 is containerized fMRIPrep/FSL/FreeSurfer and **not
bitwise-reproducible** across hosts, so raw reruns are validated by provenance spot-checks, not
golden masters — this is why the precomputed cut is the tested boundary. Each dataset README's
"Precomputed cut / running" section has the dataset-specific details.

---

## Getting the data

Raw fMRI is **not hosted here** — it lives at OpenNeuro / Edinburgh DataShare (see
each dataset README). Our **precomputed tier is hosted on OSF** (umbrella
[`ehrt6`](https://osf.io/ehrt6/), DOI
[10.17605/OSF.IO/EHRT6](https://doi.org/10.17605/OSF.IO/EHRT6); one component per dataset) as a
single per-dataset zip under `fMRI_data/`, pulled + sha256-verified by `download_precomputed.py`
(see [`docs/OSF_DATA.md`](docs/OSF_DATA.md)). Each cut carries a `_fMRI_data_contents.txt` you can
preview on OSF before downloading. **The components are public — no token is needed** to
download; the optional `--token` / `$OSF_TOKEN` remains for anyone hosting a private mirror.

**Citing the data.** The precomputed fMRI data release has a DOI —
[`10.17605/OSF.IO/EHRT6`](https://doi.org/10.17605/OSF.IO/EHRT6) (*Topo-Omni fMRI data release*).
Please also cite the **original source studies** when reusing a dataset (per-study citation +
license in each dataset README; Pernet CC-BY 4.0, Marvi & Jung CC0).

## Environments

Stage-1 dependency stacks **diverge across breaking boundaries** (Pernet is on
nilearn 0.10.4; Marvi/Jung on 0.12.1), so there is **one env per dataset**:

```bash
conda env create -f environment/analysis_env_pernet.yml   # or _marvi / _jung
pip install -e core/                                       # shared utilities, per env
```

---

## Model-derived inputs & provenance

Some figures depend on model outputs from the topographic model
([`topo-omni`](https://github.com/epflneuroailab/topo-omni)). Those are currently
**vendored fixtures** (Jung cluster-assignment JSONs; Pernet Fig. B3b model bars),
each pinned by source commit + hash under `<dataset>/data/PROVENANCE.md`. **This
release regenerates the vendored-fixture version of those figures** — swapped for
live `topo-omni` outputs when this code is folded into that repo (docs/DESIGN.md §0).

## License

Code is released under the parent `topo-omni` repository's license. Data licenses are
per-source and documented in each dataset README (Pernet CC-BY 4.0; Marvi & Jung CC0).
