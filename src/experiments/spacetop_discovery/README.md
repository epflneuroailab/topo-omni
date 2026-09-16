# SpaceTop discovery

Finds category-selective regions on the unified cortical sheet *without* a
hand-built localizer: cluster naturalistic movie clips, then ask which clusters
the model's sheet is selective for.

Everything here used to be split across two trees — the discovery half in
`topo-discover/` at the repo root (paths relative to that directory) and the
sheet-mapping half in this package. They are one package now; every path resolves
through `common.py`, and every script runs from the repo root as
`python -m src.experiments.spacetop_discovery.<script>`.

## Layout on disk

```
$SPACETOP_DIR/task-alignvideo/*.mp4              52 movie runs
$SPACETOP_DIR/spacetop_embeddings_v2/*.mp4       the 2s clips + *_nemotron_embeddings.npz
$SAVE_DIR/<model_name>/spacetop_discovery/       one directory per checkpoint
```

The output directory is keyed on the model because the tree is cut using *that
model's* cortical sheets. Two checkpoints share the clip embeddings that seed the
tree, and nothing else.

`TOPO_SHEET_LAYOUT` must match the checkpoint (`tiled32` for the 800×576 runs). It
is read at import time, so export it in the environment — setting it only in
`.env` is too late and a random front-end adaptor is applied silently.

## Upstream — discover the clusters

| # | Step | Output |
|---|------|--------|
| 1 | `extract_video_embeddings` | per-movie `*_nemotron_embeddings.npz` + the 2s clips |
| 2 | `compile_embeddings` | `video_clip_vectors.npy`, `clips_manifest.json` |
| 3 | `agglomerative_early_stopping` | `cortical_sheets/*.npy`, `linkage_tree_tvals_min=10_max=500.pkl` |
| 4 | `plot_dendrogram` | `clusters_tvals.json`/`.txt`, `dendrogram_tvals.png` |
| 5 | `make_cluster_collages` | `cluster_previews/cluster_NNN_nM.jpg` |
| 6 | `merge_clusters` | `merged_clusters_tvals.json` (needs a hand-written merge map) |
| 7 | `create_cluster_manifest` | `merged_clusters_tvals_with_video_info.json` |

Steps 1–2 are stimulus-side only and do not involve the checkpoint: **the clip
embeddings are reusable across runs**, so point `--vectors_path` /
`--manifest_path` at an existing pair rather than re-extracting.

Step 3 is the expensive one: one forward pass per clip (~2.7k clips) to cache the
mean unified sheet, then a top-down walk of the Ward tree that keeps splitting a
node only while both children are *more* selective than the parent (median Welch t
of the cluster's sheets against every other clip's). Sheets are cached per clip, so
the step is resumable; `--extract_only` stops after the GPU half.

Steps 5–6 are the labelling loop and need a human: read the collages, then

```bash
python -m src.experiments.spacetop_discovery.merge_clusters --model_name <run> --init_map
# edit <output_dir>/merge_map.json — rename keys, group ids
python -m src.experiments.spacetop_discovery.merge_clusters --model_name <run>
```

Early stopping deliberately over-splits, so a single region typically arrives as
several adjacent leaves. The old code hardcoded one run's grouping; it now lives in
a JSON map per run, because cluster ids do not survive a change of checkpoint.

## Selectivity of the discovered clusters

| Step | Output |
|------|--------|
| `calculate_selectivity` | `cluster_selectivity_scores_v1.json` |
| `plot_model_selectivity` | `figures/selectivity_scores_*.png` |

Scored on the **raw** clusters from step 4, not the merged ones, so the number
reflects what discovery actually found. `--brain_clusters` (which points are drawn
green) is run-specific — read the ids off the collages.

## Downstream — put the clusters on the sheet

`extract` → `contrast` → `plot_selectivity` → `correlate_t_values` /
`run_clusters_iou` / `plot_corr`. These consume cluster clips laid out as
`$STIMULI_DIR/spacetop_audio/cluster_clips/<cluster_id>/*.mp4` and compare the
resulting t-maps against the fMRI contrasts.
