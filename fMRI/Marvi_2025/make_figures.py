#!/usr/bin/env python
"""Marvi 2025 (EMFL) — Stage 1 driver: precomputed derivatives -> paper figures.

Figure lineages ported here (allow-list; docs/DESIGN.md §2.4):
  - Fig. A2 fROI profiles (Branch A): 06 -> splits -> frois -> CV -> extract -> plot
  - Figs. 2 & 3 surface maps (Branch B): 08 -> 09 -> 11 -> 12/18 -> 10/19
    (individual-subject fsnative; concat GLM 08, NOT per-run 07 — docs/DESIGN.md §7)

Precomputed cut = fMRIPrep derivatives -> --derivatives-root.

--input-source {precomputed, raw}:
  * precomputed (default): consume the shipped fMRIPrep-derivatives cut (BOLD/confounds +
    first_level_glm/ split GLM) and render figures. Branch A (figA2) regenerates fROIs from
    the split GLM (cheap, deterministic, bitwise per its golden master), cross-validates,
    extracts per-condition betas, and plots Fig. A2 — so the shipped cut need only carry the
    split first-level GLM (frois are reproduced, not shipped).
  * raw: regenerate the split first-level GLM cut from raw BIDS first (analysis/glm_splits
    into --derivatives-root, events from --raw-root), then render as above. Heavy nilearn GLM
    — run via SLURM bigmem; the GLM is NOT golden-mastered (docs/DESIGN.md §6), best-effort.

Branch A (figA2) rendering chain (docs/DESIGN.md §2.4):  frois -> CV -> extract -> plot
  define_frois -> cross_validation -> extract_condition_responses -> plot_figure_a2.
Branch B (fig2/fig3 native-surface maps) rendering chain (docs/DESIGN.md §2.4):
  09 project_to_native_surface -> 11 convert_inflated_surfaces -> [12 project_parcels_to_surface
  if a FreeSurfer container is available, else use the shipped native_surface_parcels/ contours]
  -> 10 visualize_native_surface (metric=signed_log_p, paper method).
  09 + 11 are deterministic (bitwise goldens) so they are regenerated from the shipped concat
  GLM (08) + FreeSurfer sourcedata; 12 is FreeSurfer-CLI (Stage-0-ish), so its parcel contours
  are shipped and reused by default. Raw path regenerates 08 (concat GLM) first (SLURM bigmem).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import config

_HERE = Path(__file__).resolve().parent

FIGURES = ("figA2_froi_profiles", "fig2_surface", "fig3_surface")

# Branch A cross-validated fROIs need the even/odd split GLMs (define_frois / cross_validation
# / extract_condition_responses all read the split dirs).
_BRANCH_A_SPLITS = ("even", "odd")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-source", choices=("precomputed", "raw"), default="precomputed")
    p.add_argument("--derivatives-root", default=None, help="fMRIPrep derivatives (the precomputed cut).")
    p.add_argument("--results-root", default=None, help="Where intermediate results (CSVs, surface projections) are written.")
    p.add_argument("--plots-root", default=str(_HERE / "plots"),
                   help="Directory for rendered figures (default: <dataset>/plots). Intermediate "
                        "CSVs/surface maps still go under --results-root/--derivatives-root.")
    p.add_argument("--raw-root", default=None, help="OpenNeuro ds006179 BIDS root (only for --input-source raw).")
    p.add_argument("--subjects", nargs="+", default=list(config.ALL_SUBJECTS),
                   help="Subject IDs (default: all 6 EMFL subjects).")
    p.add_argument("--space", default=config.DEFAULT_SPACE, help="Analysis space (Branch A: MNI 2mm).")
    p.add_argument("--figures", nargs="+", choices=FIGURES, default=list(FIGURES))
    p.add_argument("--exhaustive", action="store_true",
                   help="Branch B (fig2/fig3): render the FULL 6×9×2=108-panel surface grid "
                        "(the matrix the paper's panels were curated from). Default: only the "
                        "published Figs 2 & 3 panels (see visualize_native_surface.PAPER_PANELS).")
    return p


def regenerate_raw_cut(args) -> None:
    """Raw path: regenerate the precomputed cut from raw BIDS into --derivatives-root.

    Branch A (figA2): re-run the split first-level GLM (even/odd) -> the zmap+effect maps the
    Fig-A2 lineage consumes. Branch B (fig2/fig3): re-run the concatenated GLM (08) -> the
    native-T1w contrast maps the surface lineage consumes. Both are heavy nilearn fits (run via
    SLURM bigmem) and NOT golden-mastered (docs/DESIGN.md §6).
    """
    if not args.derivatives_root:
        raise SystemExit("--input-source raw requires --derivatives-root (where the cut is written).")
    if not args.raw_root:
        raise SystemExit("--input-source raw requires --raw-root (raw BIDS event TSVs).")

    if "figA2_froi_profiles" in args.figures:
        from analysis import glm_splits
        print(f"[raw] regenerating Branch-A split GLM cut (splits={_BRANCH_A_SPLITS}) "
              f"for {len(args.subjects)} subject(s) -> {args.derivatives_root}")
        glm_splits.run_glm_splits(
            subjects=args.subjects,
            derivatives_root=args.derivatives_root,
            raw_root=args.raw_root,
            splits=_BRANCH_A_SPLITS,
            space=args.space,
            save_outputs=True,
        )

    branch_b = [f for f in args.figures if f in ("fig2_surface", "fig3_surface")]
    if branch_b:
        from analysis import concatenated_glm
        print(f"[raw] regenerating Branch-B concatenated GLM (08) for {len(args.subjects)} "
              f"subject(s) -> {args.derivatives_root} (heavy; run via SLURM bigmem)")
        concatenated_glm.main(
            ["--derivatives-root", str(args.derivatives_root),
             "--raw-root", str(args.raw_root), "--subjects", *args.subjects])


def render_fig_a2(args) -> Path:
    """Branch A: frois -> CV -> extract -> plot from the split GLM cut. Returns the SVG path.

    Regenerates fROIs into --derivatives-root (deterministic, bitwise per its golden master —
    so the shipped cut need not carry frois), cross-validates (QC), extracts per-condition
    betas -> details CSV in --results-root, then renders Fig. A2 there.
    """
    if not args.derivatives_root:
        raise SystemExit("figA2 requires --derivatives-root (the split first-level GLM cut).")
    deriv = str(args.derivatives_root)
    results = str(args.results_root or args.derivatives_root)
    plots = str(args.plots_root)
    Path(results).mkdir(parents=True, exist_ok=True)  # CV/extract write CSVs here; may be fresh
    subjects = list(args.subjects)
    space = str(args.space)

    from analysis import (cross_validation, define_frois,
                          extract_condition_responses, plot_figure_a2, plot_contrast_bars)

    common = ["--derivatives-root", deriv, "--subjects", *subjects, "--space", space]
    print("[figA2] 1/4 define_frois -> <deriv>/<subj>/frois/")
    define_frois.main(common)
    print("[figA2] 2/4 cross_validation (QC) -> <results>/cross_validation_details.csv")
    cross_validation.main(common + ["--details-csv", str(Path(results) / "cross_validation_details.csv")])
    print("[figA2] 3/4 extract_condition_responses -> <results>/condition_responses_details.csv")
    extract_condition_responses.main(
        ["--derivatives-root", deriv, "--results-root", results, "--subjects", *subjects, "--space", space])
    print(f"[figA2] 4/5 plot Fig. A2 -> {plots}")
    details = Path(results) / "condition_responses_details.csv"
    plot_figure_a2.main(["--details-csv", str(details), "--output-dir", plots])
    # The group-level 2-bar contrast plots that accompany the main-text surface figures
    # (same CSV, no new GLM) — the bars beside each Fig. 2/3 panel.
    print("[figA2] 5/5 group contrast bars")
    plot_contrast_bars.main(["--details-csv", str(details), "--output-dir", plots])
    return Path(plots) / "replication_of_fig_4_from_Marvi_2025_with_indiv.svg"


def render_fig2_fig3(args) -> Path:
    """Branch B: 09 project -> 11 inflated -> [12 parcels] -> 10 render. Returns output root.

    09 (surface projection) and 11 (inflated GIFTIs) are deterministic (bitwise goldens), so
    they are regenerated into --derivatives-root from the shipped concat GLM (08) + FreeSurfer
    sourcedata. 12 (parcel contours) is FreeSurfer-CLI: if a container is on PATH AND the
    contours are not already shipped, it is run; otherwise the shipped native_surface_parcels/
    contours are reused, and if neither is present we render without contours (with a warning).
    Rendering uses metric=signed_log_p (paper method, ±3).
    """
    if not args.derivatives_root:
        raise SystemExit("fig2/fig3 require --derivatives-root (the concat-GLM + FreeSurfer cut).")
    deriv = str(args.derivatives_root)
    plots = str(args.plots_root)
    Path(plots).mkdir(parents=True, exist_ok=True)  # render writes here; may be fresh
    subjects = list(args.subjects)

    from analysis import (project_to_native_surface, convert_inflated_surfaces,
                          project_parcels_to_surface, visualize_native_surface)

    print("[fig2/3] 1/4 project_to_native_surface (09) -> <deriv>/native_surface_projections/")
    project_to_native_surface.main(["--derivatives-root", deriv, "--subjects", *subjects])
    print("[fig2/3] 2/4 convert_inflated_surfaces (11) -> <deriv>/<subj>/anat/*_inflated.surf.gii")
    convert_inflated_surfaces.main(["--derivatives-root", deriv, "--subjects", *subjects])

    parcels_present = (Path(deriv) / "native_surface_parcels").exists()
    fs_available = project_parcels_to_surface._freesurfer_tools_available()
    no_contours = False
    if fs_available and not parcels_present:
        print("[fig2/3] 3/4 project_parcels_to_surface (12; FreeSurfer CLI) -> native_surface_parcels/")
        project_parcels_to_surface.main(["--derivatives-root", deriv, "--subjects", *subjects])
    elif parcels_present:
        print("[fig2/3] 3/4 using shipped native_surface_parcels/ contours (12 not re-run)")
    else:
        no_contours = True
        print("[fig2/3] 3/4 ⚠ no parcel contours and no FreeSurfer container — rendering WITHOUT "
              "contours (ship native_surface_parcels/ or run step 12 in a container for contours)")

    sel = "full 108-panel grid" if getattr(args, "exhaustive", False) else "published paper panels"
    print(f"[fig2/3] 4/4 visualize_native_surface (10; metric=signed_log_p; {sel}) -> {plots}")
    argv = ["--derivatives-root", deriv, "--output-dir", plots,
            "--metric", "signed_log_p", "--subjects", *subjects]
    if args.exhaustive:
        argv.append("--exhaustive")
    if no_contours:
        argv.append("--no-contours")
    visualize_native_surface.main(argv)
    return visualize_native_surface.output_root(
        Path(plots), "signed_log_p", 3.0, 0.05, no_contours)


def render_figures(args) -> int:
    """Render the requested figures from the (now-present) precomputed cut."""
    produced = []
    if "figA2_froi_profiles" in args.figures:
        produced.append(render_fig_a2(args))

    # Fig. 2 (visual) and Fig. 3 (auditory-cognitive) share the one native-surface render pass.
    if any(f in args.figures for f in ("fig2_surface", "fig3_surface")):
        produced.append(render_fig2_fig3(args))

    print("\nFigures produced:")
    for p in produced:
        print(f"  - {p}")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.input_source == "raw":
        regenerate_raw_cut(args)
    return render_figures(args)


if __name__ == "__main__":
    sys.exit(main())
