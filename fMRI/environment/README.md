# Environments

Stage-1 dependency stacks **diverge across breaking boundaries**, so there is one
env per dataset (docs/DESIGN.md §6). Do **not** try to unify them.

| | Pernet | Marvi | Jung |
|---|---|---|---|
| nilearn | **0.10.4** | 0.12.1 | 0.12.1 |
| numpy | 2.0.2 | 1.26.4 | 2.2.5 |
| scipy | 1.13.1 | 1.11.1 | 1.15.3 |
| python | 3.9 | 3.10 | 3.10 |

Pernet on nilearn 0.10.4 vs Marvi/Jung on 0.12.1 is a **breaking** surface/GLM API
gap; Marvi vs Jung differ in numpy. Each env is pinned to the versions its golden
masters were produced under — "within tolerance" is unfalsifiable otherwise.

```bash
conda env create -f analysis_env_pernet.yml    # or _marvi / _jung
conda activate omni-fmri-pernet
pip install -e ../core
```

## Stage-0 toolchains

`stage0_pernet_fsl/`, `stage0_marvi_fmriprep/`, `stage0_jung_fmriprep/` hold the
**preprocessing** toolchain specs (FSL 6.0.7; fMRIPrep 24.0.1 + FreeSurfer). Stage 0
is containerized and **not** bitwise reproducible — it is ported faithfully and
provenance-spot-checked, not golden-mastered (docs/DESIGN.md §2.5 / §6). Default
reproduction uses the `precomputed` path and never runs Stage 0.

> **STATUS: scaffold.** The three `analysis_env_*.yml` carry the captured version
> pins; full transitive locks + Stage-0 container specs are frozen during each
> dataset's port.
