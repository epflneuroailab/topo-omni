# Jung_2025/data — provenance

Small vendored inputs for the Jung lineages. Large data (fsaverage6 GIFTIs +
confounds) is never committed here — it lives on OSF and is pulled by the download
script (docs/DESIGN.md §5/§6).

## Vendored model input — cluster-assignment JSON (from `topo-omni`)

The 54-cluster (new54) assignment comes from the topographic model (`topo-omni`). All six
published brain-validation maps are single discovered clusters of this one partition
(App. D "14 clusters" is a typo for 54 — authors Mehrer + AlKhamissi, 2026-07-07;
README §4). The 14-/21-/22-cluster JSONs are NOT in the paper and are not vendored.
Vendored now as a fixture; the release regenerates the **vendored-fixture** version of
Figs. 6 / D4 / D5, swapped for live `topo-omni` output at merge (docs/DESIGN.md §5).

Copied verbatim from the dev repo `20251211_fMRI_movie_watching_spacetop/src/analysis/`
(commit `4066746`) on 2026-07-07.

| File | Dev source name | sha256 | Notes |
|---|---|---|---|
| `cluster_assignments/54_cluster.json` | `20260607_supercluster_assignment_individual_clusters.json` | `882173ec67a649700f22ce1ff0173cd82d2c4c36db037a3d237b84800362ffd1` | IDs 0–53; drives Fig. 6 / D4 (IDs 5/32/49) + Fig. D5 (IDs 6/30/31) |

> **Source model run.** The bytes of this JSON are pinned by the sha256 above. Its
> *scientific* provenance — which `topo-omni` run produced it — is not recorded here,
> and by design cannot be recovered from this file alone. The 54-cluster partition is not
> a fixed parameter but the emergent output of a discovery run: the agglomerative
> early-stopping pipeline in the modeling side of the repo (`topo-discover/`) merges until
> a stopping criterion fires, yielding IDs 0–53. Reconstructing which run this was requires
> data from the modeling part of the repo that is not vendored alongside this fixture —
> the model checkpoint / video-embedding set fed to `topo-discover/agglomerative_early_stopping.py`,
> the early-stopping (and any merge) parameters, and the run output
> (`clustering_v2/clusters_tvals.json`) that this JSON derives from. That output file is
> not committed, and the discovery code addresses it by hardcoded relative path with only
> an informal version tag (`clustering_v2` / `_v3`) and this file's `20260607` datestamp as
> markers. So the discovery pipeline is reproducible in principle, but the specific run
> behind this partition lives on the modeling side, not in the fMRI release.

## Derived Stage-1 input — cluster-assignment CSV

Generated from the JSON above by `analysis/parse_clusters.py` (port of dev `44`), using
sub-0001 `events.tsv` for run-relative video onsets (all subjects saw the same stimulus
order). Vendored so the `--input-source precomputed` path needs no raw BIDS; the parse
step is exercised (and golden-tested bitwise against this) on the `raw` path.

| File | sha256 | Rows / clusters |
|---|---|---|
| `cluster_assignments/cluster_assignments_new54clusters.csv` | `adeed168d7d2d51b29806ffae76620c145990be288e17ebc8cc0ea86cf1a274e` | 2572 rows, IDs 0–53 |
