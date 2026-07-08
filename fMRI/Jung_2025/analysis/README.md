# Jung_2025/analysis — Stage 1 (Nilearn, glm_unified fsaverage6)

Testable boundary: precomputed fsaverage6 GIFTIs + confounds → paper figures. Ported
in index order, each lineage pinned by a golden master first (docs/DESIGN.md §6, §9).

- `40` → `41` → `17_v2` (IDs 2/12/13) ...... Fig. 6 / Fig. D4  (14-cluster)
- `44` → `45` → `47`   (IDs 6/30/31) ....... Fig. D5           (54-cluster)

Hard invariant pinned by characterization test: **n=78 / df=77** — the confound
loader drops exactly {0035, 0044, 0061, 0084, 0131}, which is what produced the
published maps (docs/DESIGN.md §7). Not a bug to fix; a behavior to preserve.

STATUS: scaffold.
