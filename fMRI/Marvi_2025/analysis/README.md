# Marvi_2025/analysis — Stage 1 (Nilearn)

Testable boundary: precomputed fMRIPrep derivatives → paper figures. Ported in index
order, each lineage pinned by a golden master first (docs/DESIGN.md §6, §9).

- **Branch A** `06` → splits → frois → CV → extract → plot ...... Fig. A2 fROI profiles
- **Branch B** `08` → `09` → `11` → `12/18` → `10/19` ........... Figs. 2 & 3 surface maps
  (individual-subject fsnative; concat GLM `08`, not per-run `07`)

## Modules

Branch A: [`first_level_glm.py`](first_level_glm.py) · [`glm_splits.py`](glm_splits.py)
(raw-path GLM producers) → [`define_frois.py`](define_frois.py) →
[`cross_validation.py`](cross_validation.py) →
[`extract_condition_responses.py`](extract_condition_responses.py) →
[`plot_figure_a2.py`](plot_figure_a2.py).

Branch B: [`concatenated_glm.py`](concatenated_glm.py) (08) →
[`project_to_native_surface.py`](project_to_native_surface.py) (09) →
[`convert_inflated_surfaces.py`](convert_inflated_surfaces.py) (11) →
[`project_parcels_to_surface.py`](project_parcels_to_surface.py) (12/18) →
[`visualize_native_surface.py`](visualize_native_surface.py) (10/19).

Which numbers are frozen CI goldens vs one-off spot-checks: see the reproducibility
table in [`../README.md`](../README.md).

## `core.froi_cv` sharing decision — RESOLVED: kept local (docs/DESIGN.md §4, §10)

Pernet's `cv_*` fROI math and Marvi's `emfl.roi.{definition,validation,extraction}` do
**not** factor into a clean shared kernel, so each stays local. They diverge on the parts
that matter: Pernet does a half-split fROI on a single voice contrast over one parcel
pair (A/B); Marvi takes the top-10%-t within each of ~30 anatomical parcels, even/odd
split, and reports Dice + spatial pattern-correlation. Forcing a shared `core/froi_cv.py`
would have coupled two genuinely different designs — the "leave local if they diverge"
lean in docs/DESIGN.md §4 applies.

STATUS: complete — see the Implementation notes section of `../README.md` for per-stage records.
