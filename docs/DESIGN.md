# Design: reproducing the model-side figures

This documents how the **model side** of the repo (`src/`, `scripts/`, `topo-discover/`) turns a
fresh clone into the paper's figure panels. It mirrors the philosophy of the brain-side
[`fMRI/`](../fMRI/) subproject: **two clearly separated paths**, a hosted precomputed cut as the
default, and full recompute as an opt-in.

## Two paths

```
                 ┌─────────────────────── precomputed (default) ───────────────────────┐
 OSF cut  ──►  download_precomputed.py  ──►  make_all_figures.py --input-source precomputed  ──►  figures_out/
                 └─ small per-figure intermediates (stats / profiles / JSONs / t-maps)   ┘
                                                                                      (no GPU, no stimuli)

                 ┌──────────────────────────── raw (opt-in) ───────────────────────────┐
 HF model ──►  load_topo_omni()  ──►  run on stimuli (eval/*)  ──►  same plotters       ──►  figures_out/
                 └─ epfl-neuroai/topo-omni + stimulus banks                              ┘
                                                                                      (needs a GPU)
```

* **precomputed** — `make_all_figures.py` runs only the *plot* steps, which read the cut's small
  intermediates (a few hundred MB) and render the panels. This is the default and needs neither a
  GPU nor any stimuli.
* **raw** — `--input-source raw` first runs the *compute* steps (download the model from
  HuggingFace via [`src/core/model_loading.py`](../src/core/model_loading.py), run it on stimuli to
  regenerate the intermediates), then the same plot steps.

The orchestrator ([`make_all_figures.py`](../make_all_figures.py)) dispatches one script per figure
under [`scripts/figures/`](../scripts/figures/); each declares its steps (see
[`_common.py`](../scripts/figures/_common.py)). Both are runnable standalone.

## The precomputed cut

Built by [`scripts/build_precomputed_cut.py`](../scripts/build_precomputed_cut.py) (the single
source of truth for its contents) and hosted on OSF (DOI 10.17605/OSF.IO/EHRT6). The manifest
[`data/precomputed_manifest.json`](../data/precomputed_manifest.json) carries the OSF `osf_guid`,
the zip name, and a per-file sha256 (the integrity guarantee; the zip is just transport).
`download_precomputed.py` fetches + extracts + verifies it (idempotent).

It contains **only the small plot inputs**: per-category selectivity stats
(`*_all_selectivity_stats.pkl`), response-profile pickles, ablation/stimulation accuracy JSONs,
the model-vs-fMRI Moran's-I summary, per-cluster t-maps, and the discovery selectivity scores.
It deliberately **excludes** the large intermediates — the ON/OFF feature caches, the
retinotopy/tonotopy `cortical_sheets.npy` (~2 GB), and the NSD feature HDF5s — which are instead
regenerated on the raw path.

## Per-figure input availability

Stimuli are only *partly* public, so not every panel is recomputable by an external clone. The
precomputed path covers every panel below; the "raw" column says what a from-scratch rerun needs.

| Panel | precomputed | raw needs |
|-------|:-----------:|-----------|
| 2c model-guided discovery (scores, model-vs-fMRI corr, cluster maps) | ✅ | HF model + SpaceTop movies (not redistributable) |
| 3a–d visual selectivity map + response profiles | ✅ | HF model + MARVI localizer stimuli (not redistributable) |
| 3e–f retinotopy | — (raw only) | HF model + **self-generating** bank (`generate_retinotopy`) |
| 4a speech / 4b voice (PerNet) response profiles | ✅ | HF model + speech / PerNet stimuli |
| 4c tonotopy | — (raw only) | HF model + **self-generating** bank (`generate_frequencies`) |
| 5a–c language / multiple-demand / theory-of-mind profiles | ✅ | HF model + cognitive-localizer text |
| 6a–c driving / suppression accuracy | ✅ | HF model + localizer stimuli + LLM judge |
| 7a–b animal / landscape network maps | ✅ | HF model + SpaceTop movies (collages need the movie frames) |

**Self-generating** banks (retinotopy, tonotopy) make those panels fully reproducible on the raw
path by anyone with a GPU — no data download required:

```bash
python -m src.utils.generate_retinotopy    # -> $STIMULI_DIR/retino_bank/{*.png, manifest.csv}
python -m src.utils.generate_frequencies    # -> $STIMULI_DIR/tone_bank/{*.wav, manifest.csv}
```

For the panels whose stimuli can't be redistributed, point `$STIMULI_DIR` at your own copy laid
out as the eval configs expect (`marvi_videos_<condition>/`, `pernet_vocal_nonvocal/`, …), produce
the intermediates with the building-block scripts (`scripts/*_response_profiles.sh`,
`scripts/run_ablation.sh`, …), then render. In `--input-source raw`, the orchestrator treats the
self-generating panels (tonotopy/retinotopy) as required and every other plot step as
**best-effort**: if its intermediate isn't present it warns and continues rather than aborting the
figure, so a machine with only the public inputs still gets those panels.

## Model loading

All evaluation/extraction code loads the model through the single helper
`src.core.model_loading.load_topo_omni`, which pulls `epfl-neuroai/topo-omni` from HuggingFace by
default (override with `$TOPO_OMNI_MODEL` or `model.model` in a config, e.g. a local checkpoint).
Inference sets `apply_spatial_loss=True` so each forward assembles the unified 304×512 cortical
sheet, but the spatial *loss* — and hence the training-only neighborhood/position files — is never
touched (no `labels` are passed), so a fresh clone needs only the model weights.
