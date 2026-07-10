# Marvi_2025/data — provenance

Small vendored inputs for the Marvi lineages. Large data (fMRIPrep derivatives) is
never committed here — it lives on OSF and is pulled by the download script
(docs/DESIGN.md §5/§6).

## Vendored inputs

| File/dir | Source | Commit | Notes |
|---|---|---|---|
| `PARCELS/` | dev repo `src/aux/emfl_analysis-main/PARCELS/` | `ef1da34` | anatomical parcellation maps (used subdirs only); 65 NIfTIs, **bitwise-identical** to the dev source (verified per-file, 2026-07-07) |

**`PARCELS/` contents** (65 `*.nii.gz`, used subdirs only — exploratory
`old_parcels/physics/steel` not vendored):

| Subdir | Files | Category |
|---|---|---|
| `julian/` | 23 | face / body / scene / object (Julian et al.) |
| `language/` | 13 | language network |
| `md/` | 20 | multiple-demand |
| `tom/` | 7 | theory-of-mind |
| `speech/` | 1 | speech |
| `vwfa/` | 1 | visual word form area (LH only) |

**Integrity.** All 65 files are bitwise-identical to the dev source @ `ef1da34`
(verified by per-file `sha256`, 2026-07-07). Per-file digests are regenerable with:

```bash
find data/PARCELS -name '*.nii.gz' | sort | xargs sha256sum
```

Aggregate digest (sha256 of the sorted per-file `sha256sum` output), as a single
tamper-evident anchor:

```
b82827eb0aff92560ea29f112c04217d5815954b810b32d7dc2da8e4de2dec99
```

Marvi has **no** model-derived fixture (unlike Pernet Fig. B3b / Jung cluster JSONs).
