# Pernet_2015/data — provenance

Small vendored inputs for the Pernet lineages. Large data is never committed here
(docs/DESIGN.md §5/§6) — it lives on OSF and is pulled by the download script.

## Vendored model input — Fig. B3b model bars

Fig. B3b compares the **brain** island-Moran's-I (computed here by `05`) against two
**model** distributions (Topo-Omni and Non-Topo), drawn as mean ± SE by `06`. The model
bars come from the vendored **per-island distributions** below — the exact artifacts the
published figure (dev `06_plot_island_morans_i_comparison.py`) was drawn from. They are
swapped for a live `topo-omni` run at merge (docs/DESIGN.md §7, §10).

### Primary source — per-island distributions (what `06` plots)

| File (`data/model_island_morans_i/`) | sha256 | n islands | mean I | SE |
|---|---|---|---|---|
| `topoomni_vocals_fwhm4.0_island_morans_I.json` | `d434c2533a67035e9697b3566bb66cc49dd93bea14235b30721727469314f5ff` | 79 | 0.574898 | 0.041570 |
| `nontopo_vocals_fwhm4.0_island_morans_I.json` | `6bb61fa999acc5090901ea3133d7c331b7845e30d8224ba87fbc571d0090b581` | 418 | 0.235253 | 0.012036 |

Each JSON maps island id → `{moran_I, p_value}` (no island sizes). Copied verbatim from
the Marvi/topo-omni dev repo at
`20251030_Marvi_2025_efficient_fMRI_localizer/src/omni_clustering_map_analysis_topo_vs_non_topo_models/data/`
(both untracked there; the sha256 pins are the citeable identity). `06.compute_bars`
reproduces the mean/SE above; `tests/test_b3b_comparison.py` locks them.

### Deprecated — the orphaned point-estimate literals (NOT the figure)

`fig_b3b_model_island_morans_i.json`
(sha256 `9b50bfd96a4d6585d0166783788b0ce79b1d50f583b18e045cc49074f02336a1`) holds
`topo_omni_I = 0.5938924589442801` / `nontopo_I = 0.12599509990429125` (n=29 / 120).
**These are not the published Fig. B3b bars.** Traced to ground truth (2026-07-06): they
are hard-coded literals at lines 117–118 of the **untracked** dev script
`src/05_island_morans_i.py` (sha256
`1694a1ce875b0d90a6633209e94b47e69b478ea632de2533edac6c83f0eb9535`), used only for a
secondary t-test *inside* dev `05` — a computation that never fed the chart. They do not
reproduce from the per-island JSONs by any simple aggregation (`mean(all)` = 0.5749 /
0.2353; `p<0.05` counts 64 / 251 ≠ 29 / 120), so the 29 / 120 reflect a min-island-size
filter over sizes absent from those JSONs — a richer upstream we do not have and do not
need. **The port no longer uses these for the figure.** The old fixture is retained only
because the ported `05` still reads it for its (non-figure) secondary t-test; that too
should be dropped or re-pointed at merge.

> **STATUS: Fig. B3b model bars REPRODUCIBLE.** `06` derives them (mean ± SE) from the
> sha256-pinned per-island distributions above — matching the published chart. The one
> remaining §10 item is to **regenerate them from a live `topo-omni` run at merge**
> (island Moran's I, vocals contrast, fwhm 4.0) and record that run's commit + config
> here, replacing the vendored copies. The brain bar (`05`) is independently golden.

## Other vendored inputs

- `fold_split.json`, parcels, etc. — TODO(port): copy from the dev repo with their
  origin noted here.
