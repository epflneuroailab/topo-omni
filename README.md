# Topo-Omni: Discovering Functionally Selective Brain Regions with a Deep Topographic Multimodal Model

[![arXiv](https://img.shields.io/badge/arXiv-2606.09770-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2606.09770)
[![Project Page](https://img.shields.io/badge/Project%20Page-EPFL%20site-E60028.svg?logo=googlechrome&logoColor=white)](https://topo-omni.epfl.ch)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-FFD21E.svg?logo=huggingface)](https://huggingface.co/epfl-neuroai/topo-omni)


![graphical abstract figure](data/topo-omni-graphical-abstract.png)

Inducing topographic organization in a multimodal language model (Qwen2.5-Omni) using a spatial smoothness loss inspired by biological cortical maps.

The model learns a 2-D *cortical sheet*, a fixed spatial layout of neurons shared across all transformer layers, and is trained to encourage nearby neurons to respond similarly. The result is emergent functional regions (face patches, scene areas, auditory regions) that spatially cluster on the sheet, mirroring organization in the human brain.

---

## Overview

Standard neural networks have no spatial structure: any neuron can interact with any other regardless of position. The brain, by contrast, organizes neurons topographically: neurons with similar tuning are physically close. TopoOmni injects this inductive bias into Qwen2.5-Omni-3B by:

1. Assigning every neuron in the network a fixed 2-D coordinate on a shared cortical sheet.
2. Adding a spatial loss term during training that penalizes nearby neurons for responding dissimilarly.
3. Evaluating the learned sheet for selectivity (face/scene/object/body areas), brain alignment (NSD), and causal intervention (ablation/stimulation).
4. Discovering new functionally selective clusters and validating it on a heldout brain dataset (SpaceTop fMRI).

The approach builds on [TDANN](https://github.com/neuroailab/TDANN) and [TopoLM](https://topolm.epfl.ch).

---

## Repository Structure

```
topo-omni/
├── make_all_figures.py          # orchestrator: reproduce all model-side figures (precomputed|raw)
├── download_precomputed.py      # fetch the precomputed cut from OSF (+ sha256 verify)
├── reproduce_precomputed_figures.sh  # one-command precomputed reproduction
├── .env.example                 # environment template (copy to .env)
├── src/
│   ├── core/
│   │   ├── model_loading.py      # load_topo_omni(): single HuggingFace/local model loader
│   │   └── precomputed.py        # OSF cut fetch + sha256 verify
│   ├── models/
│   │   ├── qwen2_5_omni.py       # Modified Qwen2.5-Omni with cortical sheet output
│   │   └── spatial_utils.py      # LayerPositions, spatial loss, neighborhood utilities
│   ├── eval/
│   │   ├── run/                  # Run evaluations against a trained model
│   │   │   ├── run_selectivity.py
│   │   │   ├── run_ablation.py
│   │   │   ├── run_clusters_iou.py
│   │   │   ├── run_tonotopy.py
│   │   │   ├── run_retinotopy.py
│   │   │   ├── run_retinotopy_angle.py
│   │   │   └── run_retinotopy_ecc.py
│   │   ├── extract/              # Extract model activations for external datasets
│   │   │   ├── extract_nsd.py
│   │   │   ├── extract_spacetop.py
│   │   │   ├── extract_spacetop_rating.py
│   │   │   └── extract_clusters.py
│   │   └── analysis/             # Post-hoc analysis and comparisons
│   │       ├── marvi_response_profiles.py
│   │       ├── cognitive_response_profiles.py
│   │       ├── ablation_similarity.py
│   │       ├── contrast_spacetop.py
│   │       ├── explore_cluster_iou.py
│   │       ├── no_ablation_judge.py
│   │       └── stimulation_perception.py
│   ├── visualize/                # Plotting scripts: selectivity/ablation/cluster maps,
│   │                             # retinotopy & tonotopy maps, brain-alignment plots/tables,
│   │                             # response profiles, topo vs. non-topo comparisons, ...
│   ├── utils/
│   │   ├── island_morans_I.py    # Spatial autocorrelation (Moran's I) for patchy maps
│   │   ├── smoothing.py          # Gaussian smoothing over the cortical sheet
│   │   ├── spatial_stats.py
│   │   ├── connected_components.py
│   │   ├── generate_frequencies.py # Pure-tone stimulus bank for tonotopy mapping
│   │   └── generate_retinotopy.py  # Localized stimulus bank for retinotopy mapping
│   ├── configs/
│   │   ├── train.yml             # Training hyperparameters
│   │   ├── eval_marvi.yml        # Selectivity eval config
│   │   ├── eval_retinotopy.yml   # Retinotopy mapping eval config
│   │   ├── eval_tonotopy.yml     # Tonotopy mapping eval config
│   │   ├── init_coords.yml       # Cortical coordinate initialization config
│   │   └── accelerate_config.yml
│   ├── train.py                  # Training entry point
│   ├── distill.py                # Knowledge distillation entry point
│   ├── data_utils.py             # Dataset loading and collation
│   └── init_coords.py            # Pre-compute neighborhood structures
├── scripts/                      # Shell launchers, batch sweeps, result-compilation utilities
│   ├── train.sh
│   ├── run_ablation.sh
│   ├── run_iou.sh
│   ├── marvi_response_profiles.sh
│   ├── cognitive_response_profiles.sh
│   ├── pernet_response_profiles.sh
│   ├── plot_spacetop.sh
│   ├── compile_iou.py
│   ├── compile_moran_I.py
│   ├── run_selectivity.py        # Batch launcher for selectivity sweeps
│   ├── submit_train.py
│   └── figures/                  # Scripts that reproduce individual paper figures
├── topo-discover/                # Unsupervised discovery of functionally selective clusters
│   │                             # from video-driven model embeddings (extract → cluster → label)
│   ├── extract_video_embeddings.py
│   ├── compile_embeddings.py
│   ├── agglomerative_early_stopping.py
│   ├── plot_dendrogram.py
│   ├── merge_clusters.py
│   ├── create_cluster_manifest.py
│   ├── calculate_selectivity.py
│   ├── plot_model_selectivity.py
│   ├── make_cluster_collages.py
│   └── scripts/
├── data/                         # Small tracked artifacts (summary CSVs, figures);
│                                 # large stimuli/checkpoints/datasets are gitignored
├── fMRI/                         # ⚠ SEPARATE SUBPROJECT — brain-side, not the model.
│                                 # Reanalyzes the 3 human-fMRI datasets (Pernet/Marvi/Jung) to
│                                 # reproduce the paper's brain figures from a precomputed OSF cut.
│                                 # Own Nilearn stack + per-dataset envs (no torch/transformers);
│                                 # shares dataset names with src/ but no data or code. See below.
├── requirements.txt
└── README.md
```

---

## Cortical Sheet

The cortical sheet is a **304 × 512** 2-D grid that spans the entire model:

| Rows | Component |
|------|-----------|
| 0 – 159 | Vision encoder layers |
| 0 – 159 (cols 256–511) | Audio encoder layers |
| 160 – 303 | Thinker (LM) layers |

Each unit (neuron) has a fixed coordinate on this grid. The spatial loss computes pairwise response similarity within local Chebyshev-distance neighborhoods and penalizes dissimilarity between neighbors. The loss weight `alpha` (default: 20) controls how strongly the spatial constraint is applied relative to the task loss.

---

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` at the repo root (loaded via `python-dotenv`) and fill in the paths
you need. **For the default figure-reproduction path (precomputed) you need none of them** — only
`SAVE_DIR` (where the downloaded cut lands) is used, and the reproduce script sets it for you. The
other paths matter only for the `raw` recompute path and for training. The trained model is loaded
from HuggingFace (`epfl-neuroai/topo-omni`) by default, so no checkpoint download is required.

---

## System requirements

**Operating system.** Developed and tested on Ubuntu Linux 22.04 (x86_64).
The code is standard cross-platform Python but has only been tested on Linux.

**Python.** Python 3.10.

**Software dependencies.** All Python dependencies, with pinned versions, are listed in
[`requirements.txt`](requirements.txt) and installed with the command in [Setup](#setup). Key packages:
`torch`, `transformers`, `trl`, `accelerate`, `deepspeed`, `qwen_omni_utils`, `omegaconf`, `wandb`,
plus `numpy`, `scipy`, `scikit-learn` and `matplotlib`. The brain-side [`fMRI/`](fMRI/) subproject has
its own separate Nilearn-based environment (see [`fMRI/README.md`](fMRI/README.md)).

**Tested on.** Ubuntu Linux, Python 3.10, with the package versions pinned in `requirements.txt`.

**Typical install time.** A few minutes on a normal desktop (~5–15 minutes), dominated by downloading the PyTorch / CUDA wheels.

**Hardware.**
- *Precomputed figure reproduction (default path):* **no GPU required** — runs on a standard CPU
  desktop/laptop.
- *Raw recompute / training:* one or more CUDA-capable NVIDIA GPUs.
  The model was trained on 4× H200 --> Training uses multi-GPU via `accelerate` + `deepspeed`.

No other non-standard hardware is required.

---

## Reproducing the paper figures

The model-side figure panels (Fig 2b/c, 3, 4, 5, 6, 7) reproduce via **two clearly separated
paths**, mirroring the brain-side [`fMRI/`](fMRI/) subproject — see [`docs/DESIGN.md`](docs/DESIGN.md)
for the rationale and a per-panel input-availability table.

**1. Precomputed (default, recommended)** — plot from a hosted cut of model outputs. No GPU, no
stimuli, no model download.

```bash
./reproduce_precomputed_figures.sh            # download cut from OSF + render all figures
./reproduce_precomputed_figures.sh 3 4        # a subset

# …or the two steps by hand:
python download_precomputed.py --dest _precomputed_cut
python make_all_figures.py --input-source precomputed --derivatives-root _precomputed_cut
```

Panels land in `figures_out/figure_<id>/`.

> **This precomputed path also serves as the demo** required by the software checklist. It runs on a
> small hosted cut of model outputs (downloaded from OSF), needs no GPU, no stimuli and no model download.
> - **Demo dataset:** the precomputed cut fetched by `download_precomputed.py` from OSF
>   (DOI [10.17605/OSF.IO/EHRT6](https://doi.org/10.17605/OSF.IO/EHRT6)).
> - **Expected output:** rendered panels written to `figures_out/figure_<id>/`
>   (e.g. `figures_out/figure_3/`).
> - **Expected run time:** roughly ~5 minutes on a normal desktop for the full set
>   (less for a subset such as `./reproduce_precomputed_figures.sh 3 4`).

**2. Raw (opt-in)** — download the model from HuggingFace and recompute from stimuli (needs a GPU).
Fully wired where inputs are public (the retinotopy/tonotopy stimulus banks self-generate); panels
whose localizer stimuli aren't redistributable read from a `STIMULI_DIR` you supply.

```bash
python make_all_figures.py --input-source raw --derivatives-root results --figures 3,4
```

Everything below documents the individual building blocks these two paths orchestrate.

---

## Usage

Evaluation / figure commands are run from the **repo root** (module form, e.g.
`python -m src.eval.run.…`). The **training** commands (steps 1–2) are run from inside `src/`.

### 1. Initialize cortical coordinates *(training only)*

Pre-computes neighborhood structures for the spatial loss. Run once before training.

```bash
cd src && python init_coords.py -c configs/init_coords.yml
```

This generates `.pkl` files under `src/neighborhoods/` keyed by model name, radius, and neighborhood count.

### 2. Train *(training only)*

```bash
cd src && bash ../scripts/train.sh configs/train.yml <num_gpus> accelerate_config.yml
```

Key config options (`configs/train.yml`):

| Key | Default | Description |
|-----|---------|-------------|
| `train.apply-spatial-loss` | `true` | Enable the topographic loss |
| `topo-params.alpha` | `20` | Spatial loss weight |
| `topo-params.position-dir` | `neighborhoods/...` | Pre-computed neighborhood directory |
| `topo-params.cortical-init` | `0.001` | Cortical output projection init scale |
| `topo-params.identity-init` | `true` | Initialize output projection as identity |
| `train.n-epochs` | `5` | Training epochs |
| `train.batch-size` | `4` | Per-device batch size |
| `train.grad-accum` | `4` | Gradient accumulation steps |
| `train.learning-rate` | `5e-5` | Learning rate |

Training logs to Weights & Biases when `--wandb` is passed.

### 3. Run selectivity evaluation

Tests whether specific regions on the cortical sheet are selective for a category (faces, bodies, objects, scenes, speech, etc.) versus control stimuli.

```bash
python -m src.eval.run.run_selectivity --config src/configs/eval_marvi.yml
```

The model is loaded from HuggingFace (`epfl-neuroai/topo-omni`) by default; set `$TOPO_OMNI_MODEL`
or `model.model` in the config to use a local checkpoint instead.

Key config options (`src/configs/eval_marvi.yml`):

| Key | Description |
|-----|-------------|
| `model.model` | *(optional)* HF id or local checkpoint dir (default: `$TOPO_OMNI_MODEL`) |
| `data.mode` | `video`, `image`, or `text` |
| `data.stimuli_root` | Path to ON/OFF stimuli folders |
| `data.lm_reduce` | Reduction over LM sheet (`mean`/`max`) |
| `stats.alpha` | p-value threshold for selectivity |
| `stats.smooth` | Apply Gaussian smoothing before testing |
| `stats.fwhm_mm` | Smoothing kernel FWHM (mm) |

### 4. Ablation and stimulation

Ablates (zeros out) or stimulates (amplifies) a localizer-defined region and measures the effect on downstream responses.

```bash
bash scripts/run_ablation.sh
```

### 5. Brain alignment

Extract model activations for NSD (images) or SpaceTop (videos) fMRI datasets, then compare the topographic layout to human brain responses.

```bash
# NSD
python -m src.eval.extract.extract_nsd

# SpaceTop
python -m src.eval.extract.extract_spacetop --group-index 0
```

Run correlation analysis:

```bash
python -m src.eval.analysis.contrast_spacetop --cluster_id 32 --topk 1
```

### 6. Response profiles

Compute and plot how the top-k% most selective neurons respond across all stimulus categories. Separate sweeps are provided per localizer family:

```bash
bash scripts/marvi_response_profiles.sh [top_k_pct] [fwhm_mm] [anatomical_constraint]      # faces, scenes, objects, vwfa, bodies, speech
bash scripts/cognitive_response_profiles.sh [top_k_pct] [fwhm_mm]                          # language, theory of mind, multiple demand
bash scripts/pernet_response_profiles.sh [top_k_pct] [fwhm_mm] [anatomical_constraint]     # PerNet vocal/non-vocal folds
```

These call into `eval/analysis/marvi_response_profiles.py` / `eval/analysis/cognitive_response_profiles.py` and plot via `visualize/plot_response_profiles.py`, `visualize/plot_cog_response_profiles.py`, and `visualize/plot_pernet_response_profiles.py`.

### 7. Retinotopy and tonotopy mapping

Map preferred visual-field position (eccentricity / polar angle) and preferred sound frequency across the cortical sheet, masking units that fail an ANOVA tuning test (FDR-corrected). The stimulus banks **self-generate** (no data download), which makes these panels fully reproducible on the raw path with only a GPU:

```bash
# 1. generate the (public) stimulus banks into $STIMULI_DIR
python -m src.utils.generate_retinotopy     # -> $STIMULI_DIR/retino_bank/{*.png, manifest.csv}
python -m src.utils.generate_frequencies    # -> $STIMULI_DIR/tone_bank/{*.wav, manifest.csv}

# 2. run the model over them (produces the maps directly)
python -m src.eval.run.run_retinotopy --config src/configs/eval_retinotopy.yml   # polar angle + eccentricity
python -m src.eval.run.run_tonotopy   --config src/configs/eval_tonotopy.yml     # preferred frequency
```

(The single-axis variants `run_retinotopy_ecc` / `run_retinotopy_angle` expect `eccentricity/` and `angle/` sub-banks instead of the combined `retino_bank`.)

### 8. Cluster IoU

Measure the spatial overlap (IoU) of thresholded activation masks across stimulus groups, with a random-sampling null distribution for significance.

```bash
bash scripts/run_iou.sh
```

---

## Evaluation metrics

| Metric | Script | Description |
|--------|--------|-------------|
| Selectivity (t-test) | `eval/run/run_selectivity.py` | ON vs OFF contrast, FDR-corrected |
| Moran's I | `utils/island_morans_I.py` | Spatial autocorrelation of the selectivity map |
| Brain alignment (r) | `eval/analysis/contrast_spacetop.py` | Correlation with fMRI t-maps |
| Ablation accuracy | `eval/analysis/ablation_similarity.py` | Cross-category response similarity under ablation |
| Cluster IoU | `eval/run/run_clusters_iou.py` | Overlap of activation masks across groups |
| Retinotopic tuning | `eval/run/run_retinotopy*.py` | Preferred eccentricity/polar-angle per unit, ANOVA-tested |
| Tonotopic tuning | `eval/run/run_tonotopy.py` | Preferred sound frequency per unit, ANOVA-tested |

---

## Cluster discovery

`topo-discover/` runs an unsupervised pipeline to find functionally selective clusters from video-driven activations: extract per-clip embeddings (`extract_video_embeddings.py`, `compile_embeddings.py`), agglomeratively cluster them (`agglomerative_early_stopping.py`, `plot_dendrogram.py`, `merge_clusters.py`), then label and inspect the resulting clusters (`create_cluster_manifest.py`, `calculate_selectivity.py`, `plot_model_selectivity.py`, `make_cluster_collages.py`).

---

## fMRI analyses (`fMRI/`)

**[`fMRI/`](fMRI/) is a self-contained subproject and is quite different from the rest of this repository.** Everything above (`src/`, `scripts/`, `topo-discover/`) builds, trains, and analyzes the *model*; `fMRI/` instead reanalyzes the *human brain data*. It holds the brain-side pipelines for the three human-fMRI datasets validated in the paper — **Pernet 2015** (voice), **Marvi 2025** (localizers), and **Jung 2025 / SpaceTop** (naturalistic movie) — reproducing the paper's brain figures. It has its **own dependency stack (Nilearn — not torch/transformers), its own per-dataset conda environments, its own data (hosted on OSF, not in git), and its own docs**; it shares only dataset *names* with `src/`, no data and no code. Each dataset runs a two-stage pipeline (Stage 0 preprocessing → Stage 1 analysis → figures) with two entry points — start from the hosted **precomputed cut** (default, recommended) or rebuild from **raw** scans (`--input-source raw`). Full setup + details: [`fMRI/README.md`](fMRI/README.md); design rationale: [`fMRI/docs/DESIGN.md`](fMRI/docs/DESIGN.md).

Data and figures are **not** in git — the precomputed cut is hosted on OSF (umbrella https://osf.io/ehrt6/, DOI [10.17605/OSF.IO/EHRT6](https://doi.org/10.17605/OSF.IO/EHRT6); one component per dataset, all public). **`make_all_figures.py` renders from a cut that must already be on disk — it does not download or preprocess anything**, so you must first either download the cut from OSF (`download_precomputed.py`) **or** produce it yourself via Stage-0 preprocessing (`--input-source raw`; see [`fMRI/README.md`](fMRI/README.md)). Reproduce from a clean checkout:

```bash
cd fMRI
./reproduce_precomputed_figures.sh               # one command: download cut + render + fetch reference renders
# …or the two steps manually:
python download_precomputed.py --dest .          # STEP 1 — cut -> ./<Dataset>/  (public; no token needed)
python make_all_figures.py --derivatives-root .  # STEP 2 — figures -> ./<Dataset>/plots/  (needs STEP 1)
```

**Runtime note:** Jung and Pernet finish in minutes (Pernet needs a compute node with several GB RAM); **Marvi can take multiple hours** (its Branch-A fROI cross-validation). See [`fMRI/README.md`](fMRI/README.md) → "Compute resources" for the RAM/core guide.

---

## Key dependencies

| Package | Role |
|---------|------|
| `transformers` | Qwen2.5-Omni base model |
| `trl` | SFT training loop (`SFTTrainer`) |
| `accelerate` + `deepspeed` | Multi-GPU training |
| `qwen_omni_utils` | Multimodal preprocessing |
| `omegaconf` | YAML config management |
| `wandb` | Experiment tracking |
