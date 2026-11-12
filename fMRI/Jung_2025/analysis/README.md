# Jung_2025/analysis — Stage 1 (cluster-vs-others GLM on fsaverage6)

Testable boundary: precomputed fsaverage6 GIFTIs + confounds → paper figures. The full
description (figures, goldens, running, the Sep 2026 rebuttal reruns) is in the dataset
[`README.md`](../README.md); this file only maps the modules.

The revised paper's maps are two single clusters of the 16-cluster partition (new16),
broken runs excluded:

- `44` → `45` → `47` (new16 IDs 7/10, `--exclude-broken-runs`) ...... Fig. 6a/b

The arXiv v1/v2 maps came from the 54-cluster partition (new54): IDs 5/32/49 (old Fig. 6 /
D4) and 6/30/31 (old Fig. D5).

| Module | Role |
|---|---|
| `parse_clusters.py` | model JSON + sub-0001 events → cluster-assignment CSV |
| `regressors.py` | HRF-convolved target/others design regressors |
| `glm_engine.py` | per-subject OLS GLM → group t-test → FDR; hosts the n=78 drop |
| `cluster_contrasts.py` | driver: one cluster of a family (new54 / new27 / new16); `--exclude-broken-runs` |
| `visualize_clusters.py` | top-10%-of-FDR inflated HTML |
| `summarize_family.py` | one row per cluster: max t, joint-hemisphere FDR survivors, n, content |
| `compare_families.py` | a family vs a reference family: clip overlap, t-map r, Dice (FDR, top 10% of FDR, top 10% of all); `--ids` / `--all-pairs` for targeted tables |

Hard invariant pinned by characterization test: **n=78 / df=77**. The confound loader
drops exactly {0035, 0044, 0061, 0084, 0131}, which is what produced the published maps
(docs/DESIGN.md §7). This is behavior to preserve, not a bug to fix.
