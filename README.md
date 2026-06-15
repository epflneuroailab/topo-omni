# Topo-Omni: Discovering Functionally Selective Brain Regions with a Deep Topographic Multimodal Model

[![arXiv](https://img.shields.io/badge/arXiv-2606.09770-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2606.09770)
[![Project Page](https://img.shields.io/badge/Project%20Page-EPFL%20site-E60028.svg?logo=googlechrome&logoColor=white)](https://epflneuroailab.github.io/topo-omni/)
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
├── src/
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

Create a `.env` file at the repo root (loaded via `python-dotenv`) with the following paths:

```env
WANDB_API_KEY=...
DATA_DIR=/path/to/video/data
STIMULI_DIR=/path/to/stimuli
CKPT_DIR=/path/to/checkpoints
SAVE_DIR=/path/to/save/outputs
REPO_DIR=/path/to/repo
```

---

## Usage

All commands below are run from `src/`.

### 1. Initialize cortical coordinates

Pre-computes neighborhood structures for the spatial loss. Run once before training.

```bash
python init_coords.py -c configs/init_coords.yml
```

This generates `.pkl` files under `src/neighborhoods/` keyed by model name, radius, and neighborhood count.

### 2. Train

```bash
bash scripts/train.sh configs/train.yml <num_gpus> accelerate_config.yml
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
python -m eval.run.run_selectivity --config configs/eval_marvi.yml
```

Or launch a sweep across all categories:

```bash
bash scripts/run_ablation.sh
```

Key config options (`configs/eval_marvi.yml`):

| Key | Description |
|-----|-------------|
| `model.run_dir` | Checkpoint directory |
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
python -m eval.extract.extract_nsd

# SpaceTop
python -m eval.extract.extract_spacetop
```

Run correlation analysis:

```bash
python -m eval.analysis.contrast_spacetop
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

Map preferred visual-field position (eccentricity / polar angle) and preferred sound frequency across the cortical sheet, masking units that fail an ANOVA tuning test (FDR-corrected). Stimulus banks are generated with `utils/generate_retinotopy.py` and `utils/generate_frequencies.py`.

```bash
# Retinotopy (eccentricity + polar angle, or each individually)
python -m eval.run.run_retinotopy       --config configs/eval_retinotopy.yml
python -m eval.run.run_retinotopy_ecc   --config configs/eval_retinotopy.yml
python -m eval.run.run_retinotopy_angle --config configs/eval_retinotopy.yml

# Tonotopy
python -m eval.run.run_tonotopy --config configs/eval_tonotopy.yml
```

Plots are produced via `visualize/tonotopy.py` (and the corresponding retinotopy plotting code in `eval/run/run_retinotopy*.py`).

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

## Key dependencies

| Package | Role |
|---------|------|
| `transformers` | Qwen2.5-Omni base model |
| `trl` | SFT training loop (`SFTTrainer`) |
| `accelerate` + `deepspeed` | Multi-GPU training |
| `qwen_omni_utils` | Multimodal preprocessing |
| `omegaconf` | YAML config management |
| `wandb` | Experiment tracking |
