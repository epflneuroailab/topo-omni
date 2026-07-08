#!/usr/bin/env python3
"""Render individual native-surface activation maps — Marvi Figs 2 & 3 (Branch B, step 10).

For each subject / modality / contrast / hemisphere, threshold the projected surface stat
map and render it on the subject's inflated (or pial) fsnative mesh, with the anatomical
ROI **contours** (step 12) overlaid as black lines and sulcal depth as background. Produces
interactive HTML (and, with --save-png, 6-view static PNGs).

Paper method (signed log-p):  |signed_log_p| >= 3  (p < 0.001), sequential black->viridis
colormap, contours drawn in black, ventral view for visual / lateral for auditory.

Lineage (README §9): terminal Branch-B step. Consumes 09 (`native_surface_projections/`),
11 (`anat/*_inflated.surf.gii` + `_sulc.shape.gii`), 12 (`native_surface_parcels/`).
  input : <derivatives-root>/native_surface_projections/<subj>/<modality>/…_signed_log_p.func.gii
          <derivatives-root>/<subj>/anat/<subj>_hemi-{L,R}_{inflated,pial}.surf.gii + _sulc.shape.gii
          <derivatives-root>/native_surface_parcels/<subj>/<contrast>/…_parcel_contour.func.gii
  output: <output-dir>/subject_level_native_surface_T1w_<contour_tag>_<threshold_tag>/
          <modality>/<contrast>__<label>/<subj>_<modality>_<contrast>__<label>_hemi-{L,R}_<surf>_<view>.html
                 (e.g. auditory/false_belief_vs_false_photo__ToM/…; label = FUNCTIONAL_LABEL)

PORT NOTES vs dev-repo `src/10_visualize_individual_native_surface.py` + wrapper
`src/19_visualize_native_surface.sh` (@ ef1da34):
  * Faithful port of the rendering (black_viridis colormap, binarized-sulc background, manual
    NaN thresholding, contour-as-0.001 overlay, `plotting.view_surf`, 6-view PNG).
  * ⚠ **Release default metric = `signed_log_p`** (the PAPER method, threshold ±3), whereas the
    dev default was the exploratory `t_fdr` (BH FDR q=0.05). Both paths are preserved; the
    release reproduces the published Figs 2 & 3 by default (PLAN §2.7 "reproduce, don't
    re-analyze"). `t_fdr` remains available via `--metric t_fdr`.
  * Parameterized `--derivatives-root` / `--output-dir` (were hard-coded dev paths).

DETERMINISM (docs/DESIGN.md §6): rendering step → **NOT golden-mastered** (render-dependent, nilearn
surface-plotting API is version-sensitive). The numeric golden is upstream (09 projection,
bitwise). Validation here = e2e smoke (the chain runs and writes the expected HTML/PNG).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — MUST precede nilearn import
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from nilearn import plotting  # noqa: E402

ALL_SUBJECTS = (
    "sub-kaneff01", "sub-kaneff06", "sub-kaneff07",
    "sub-kaneff08", "sub-kaneff09", "sub-kaneff21",
)

# Contrast -> display label (dev 10 inline).
VISUAL_CONTRASTS = {
    "faces_vs_objects": "Faces > Objects",
    "scenes_vs_objects": "Scenes > Objects",
    "bodies_vs_objects": "Bodies > Objects",
    "words_vs_objects": "Words > Objects",
    "objects_vs_words": "Objects > Words (LOC)",
}
AUDITORY_CONTRASTS = {
    "false_belief_vs_false_photo": "False Belief > False Photo",
    "english_vs_nonwords": "English > Nonwords",
    "nonwords_vs_quilted": "Nonwords > Quilted Audio",
    "math_vs_theory_of_mind": "Math > Theory of Mind",
}

# Contrast -> short functional label (authoritative, from emfl.config CONTRAST_INFO/PARCEL_*):
# folded into OUTPUT dir/file names so the network each map targets is legible at a glance
# (e.g. false_belief_vs_false_photo is the Theory-of-Mind network). Input paths are unchanged.
FUNCTIONAL_LABEL = {
    "faces_vs_objects": "FFA-face",
    "scenes_vs_objects": "PPA-scene",
    "bodies_vs_objects": "EBA-body",
    "words_vs_objects": "VWFA-word",
    "objects_vs_words": "LOC-object",
    "false_belief_vs_false_photo": "ToM",
    "english_vs_nonwords": "Language",
    "nonwords_vs_quilted": "Speech",
    "math_vs_theory_of_mind": "MD",
}


# All renderable contrasts (visual + auditory), in render order — used to expand defaults.
_ALL_CONTRAST_KEYS = tuple(VISUAL_CONTRASTS) + tuple(AUDITORY_CONTRASTS)

# --- Published Figs 2 & 3 panel selection (Branch B surface render) ---------------------
# The FULL grid is 6 subjects × 9 contrasts × L/R = 108 panels — the matrix the paper's
# panels were CURATED FROM, not the set the paper shows. Rendering all 108 by default made
# "files produced" ≫ "figures in paper". So the full grid is now OPT-IN (`--exhaustive`);
# by DEFAULT the render emits only the panels that actually appear in the published Figs 2 &
# 3, listed here as (subject, contrast, hemi) tuples with hemi ∈ {"L","R"}.
#
# Confirmed by the authors (2026-07-08). Per contrast: (subject numbers, hemispheres).
# Fig. 2 (visual) shows BOTH hemispheres; Fig. 3 (speech) shows the LEFT hemisphere only.
# Contrasts NOT listed (bodies_vs_objects/EBA, false_belief_vs_false_photo/ToM,
# english_vs_nonwords/Language, math_vs_theory_of_mind/MD) do not appear in Figs 2 & 3 and
# are therefore not rendered by default.
_PAPER_FIG_PANELS = {
    # Fig. 2 (visual; ventral view) — both hemispheres
    "faces_vs_objects":    ((1, 6, 8, 9),   ("L", "R")),   # FFA — face
    "scenes_vs_objects":   ((7, 8, 9, 21),  ("L", "R")),   # PPA — scenes
    "objects_vs_words":    ((1, 6, 7, 9),   ("L", "R")),   # LOC — objects
    "words_vs_objects":    ((1, 7, 8, 21),  ("L", "R")),   # VWFA
    # Fig. 3 (auditory-cognitive; lateral view) — LEFT hemisphere only
    "nonwords_vs_quilted": ((1, 6, 7, 8),   ("L",)),       # Speech
}
PAPER_PANELS = tuple(
    (f"sub-kaneff{n:02d}", contrast, hemi)
    for contrast, (subject_nums, hemis) in _PAPER_FIG_PANELS.items()
    for n in subject_nums
    for hemi in hemis
)


def labeled_contrast(contrast_name: str) -> str:
    """`faces_vs_objects` -> `faces_vs_objects__FFA-face` (append functional label; PLAN naming)."""
    label = FUNCTIONAL_LABEL.get(contrast_name)
    return f"{contrast_name}__{label}" if label else contrast_name


def benjamini_hochberg_threshold(pvals: np.ndarray, q: float = 0.05) -> float:
    """BH FDR p-threshold at level q (dev 10; only used for the `t_fdr` metric)."""
    pvals_clean = pvals[~np.isnan(pvals)]
    if len(pvals_clean) == 0:
        return 0.0
    sorted_pvals = np.sort(pvals_clean)
    n = len(sorted_pvals)
    thresholds = (np.arange(1, n + 1) / n) * q
    below = sorted_pvals <= thresholds
    if np.any(below):
        return sorted_pvals[np.where(below)[0][-1]]
    return 0.0


def create_black_viridis_colormap():
    """viridis with pure black at 0 (so contours at 0.001 render black). Dev 10."""
    viridis = plt.cm.viridis
    n_colors = 256
    colors = [[0.0, 0.0, 0.0, 1.0]]
    for i in range(1, n_colors):
        colors.append(viridis(0.05 + (i / n_colors) * 0.95))
    return LinearSegmentedColormap.from_list("black_viridis", colors, N=n_colors)


def binarize_sulcal_map(sulc_data: np.ndarray) -> np.ndarray:
    """Sulci (>0) -> 179/255 gray, gyri -> white. Dev 10."""
    return np.where(sulc_data > 0, 179 / 255, 1.0)


def save_static_png_views(surf_mesh, data_thresholded, bg_map, output_dir: Path,
                          base_filename: str, cmap, vmax: float, hemi: str, title: str):
    """6-view static PNGs with a side colorbar (dev 10)."""
    if bg_map is not None:
        bg_map = binarize_sulcal_map(bg_map)
    VIEWS = {
        "left_lateral": ("lateral", "left"),
        "right_lateral": ("lateral", "right"),
        "dorsal": ((90, 0), "both"),
        "ventral": ((-90, 0), "both"),
        "anterior": ((0, 90), "both"),
        "posterior": ((0, -90), "both"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for view_name, (view_spec, view_hemi) in VIEWS.items():
        if view_hemi in ("left", "right") and view_hemi != hemi:
            continue
        output_path = output_dir / f"{base_filename}_{view_name}.png"
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(121, projection="3d", position=[0.05, 0.1, 0.75, 0.8])
        plotting.plot_surf_stat_map(
            surf_mesh=surf_mesh, stat_map=data_thresholded, bg_map=bg_map, hemi=hemi,
            view=view_spec, colorbar=False, cmap=cmap, vmin=0, vmax=vmax,
            threshold=0.0001, title=f"{title} - {view_name.replace('_', ' ').title()}",
            axes=ax,
        )
        from matplotlib import cm
        from matplotlib.colors import Normalize
        cbar_ax = fig.add_axes([0.85, 0.25, 0.03, 0.5])
        sm = cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=vmax))
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def create_surface_visualization(
    surface_path: Path, data_path: Path, output_path: Path, title: str,
    threshold: float = 3.0, view: str = "ventral", hemi: str = "left", vmax: float = 10.0,
    surf_type: str = "inflated", parcel_contour_path: Path = None, show_contours: bool = True,
    save_png: bool = False, png_output_dir: Path = None, metric: str = "signed_log_p",
    fdr_q: float = 0.05,
):
    """Render one HTML (+ optional PNGs) for one subject/contrast/hemi/surface. Faithful to dev 10."""
    surf_img = nib.load(str(surface_path))
    coordinates = surf_img.darrays[0].data
    faces = surf_img.darrays[1].data

    if metric == "t_fdr":
        tmap_path = Path(str(data_path).replace("_signed_log_p.func.gii", "_tmap.func.gii"))
        pval_path = Path(str(data_path).replace("_signed_log_p.func.gii", "_pval.func.gii"))
        if not tmap_path.exists() or not pval_path.exists():
            raise FileNotFoundError("t_fdr metric needs _tmap/_pval func.gii (run step 09).")
        tmap = nib.load(str(tmap_path)).darrays[0].data
        pvals = nib.load(str(pval_path)).darrays[0].data
        p_threshold = benjamini_hochberg_threshold(pvals, q=fdr_q)
        data = tmap.copy()
        fdr_mask = (pvals <= p_threshold) & (tmap > 0)
    else:  # signed_log_p (paper method)
        data = nib.load(str(data_path)).darrays[0].data
        fdr_mask = None

    # Sulcal-depth background (binarized).
    bg_map = None
    sulc_path = surface_path.parent / surface_path.name.replace(
        f"_{surf_type}.surf.gii", "_sulc.shape.gii")
    if sulc_path.exists():
        try:
            bg_map = nib.load(str(sulc_path)).darrays[0].data
        except Exception as e:  # noqa: BLE001
            print(f"      ⚠ could not load sulc: {e}")
    if bg_map is not None:
        bg_map = binarize_sulcal_map(bg_map)

    # Contour overlay.
    contour_mask = None
    parcel_info = ""
    if show_contours and parcel_contour_path and Path(parcel_contour_path).exists():
        try:
            contour_data = nib.load(str(parcel_contour_path)).darrays[0].data
            contour_mask = contour_data > 0
            n_contour = int(np.sum(contour_mask))
            if n_contour > 0:
                parcel_info = f" | {n_contour} contour vertices"
        except Exception as e:  # noqa: BLE001
            print(f"      ⚠ could not load contour: {e}")

    # Threshold -> NaN below threshold.
    data_thresholded = data.copy()
    if metric == "t_fdr":
        data_thresholded[~fdr_mask] = np.nan
    else:
        data_thresholded[np.abs(data) < threshold] = np.nan
    if contour_mask is not None:
        data_thresholded[contour_mask] = 0.001  # black in black_viridis, > 0.0001 threshold

    surf_mesh = (coordinates, faces)
    cmap = create_black_viridis_colormap()

    fig = plotting.view_surf(
        surf_mesh=surf_mesh, surf_map=data_thresholded, bg_map=bg_map, threshold=None,
        cmap=cmap, symmetric_cmap=False, vmin=0, vmax=vmax,
        title=f"{title} ({surf_type}){parcel_info}", colorbar=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.save_as_html(str(output_path))
    print(f"      ✓ HTML: {output_path.name}")

    if save_png and png_output_dir is not None:
        base_filename = "_".join(output_path.stem.split("_")[:-1])  # drop trailing view
        save_static_png_views(
            surf_mesh, data_thresholded, bg_map, png_output_dir, base_filename, cmap,
            vmax, hemi, f"{title}",
        )
    return fig


def _threshold_tag(metric: str, threshold: float, fdr_q: float) -> str:
    if metric == "signed_log_p":
        return "signed_log_p_p0.001_thresh3.0" if threshold == 3.0 else f"signed_log_p_thresh{threshold}"
    return f"FDR_q{fdr_q}"


def output_root(base_output_dir: Path, metric: str, threshold: float, fdr_q: float,
                no_contours: bool) -> Path:
    """The dev 10 output-dir naming (encodes contour choice + threshold method)."""
    contour_tag = "without_anat_contours" if no_contours else "with_anat_contours"
    return base_output_dir / (
        f"subject_level_native_surface_T1w_{contour_tag}_"
        f"{_threshold_tag(metric, threshold, fdr_q)}"
    )


def visualize_subject(
    subject_surface_dir: Path, data_dir: Path, output_dir: Path, subject: str,
    modality: str, contrasts: dict, surf_types=("inflated",), parcel_dir: Path = None,
    show_contours: bool = True, save_png: bool = False, metric: str = "signed_log_p",
    fdr_q: float = 0.05, threshold: float = 3.0, panels=None,
):
    """Render contrasts/hemis for one subject/modality (faithful to dev 10).

    ``panels``: if not None, an allow-list of ``(subject, contrast, hemi)`` tuples — any
    panel not in it is skipped (the default paper-panel selection). None = render all.
    """
    view = "ventral" if modality == "visual" else "lateral"
    for contrast_name, display_name in contrasts.items():
        for hemi_short, hemi_long in [("L", "left"), ("R", "right")]:
            if panels is not None and (subject, contrast_name, hemi_short) not in panels:
                continue
            data_path = (
                data_dir / subject / modality
                / f"{subject}_{modality}_{contrast_name}_hemi-{hemi_short}_signed_log_p.func.gii"
            )
            if not data_path.exists():
                print(f"      ✗ data not found: {data_path.name}")
                continue
            parcel_contour_path = None
            if parcel_dir:
                parcel_contour_path = (
                    parcel_dir / subject / contrast_name
                    / f"{subject}_{contrast_name}_hemi-{hemi_short}_parcel_contour.func.gii"
                )
            for surf_type in surf_types:
                surface_path = subject_surface_dir / f"{subject}_hemi-{hemi_short}_{surf_type}.surf.gii"
                if not surface_path.exists():
                    print(f"      ✗ surface not found: {surface_path.name}")
                    continue
                labeled = labeled_contrast(contrast_name)  # OUTPUT only; inputs use contrast_name
                output_path = (
                    output_dir / modality / labeled
                    / f"{subject}_{modality}_{labeled}_hemi-{hemi_short}_{surf_type}_{view}.html"
                )
                png_dir = None
                if save_png and surf_type == "inflated":
                    png_dir = output_dir / "png" / modality / labeled
                try:
                    create_surface_visualization(
                        surface_path=surface_path, data_path=data_path, output_path=output_path,
                        title=f"{subject} | {display_name} | {hemi_long}", threshold=threshold,
                        view=view, hemi=hemi_long, surf_type=surf_type,
                        parcel_contour_path=parcel_contour_path, show_contours=show_contours,
                        save_png=save_png and surf_type == "inflated", png_output_dir=png_dir,
                        metric=metric, fdr_q=fdr_q,
                    )
                except Exception as e:  # noqa: BLE001 — dev per-render skip semantics
                    print(f"      ✗ FAILED ({surf_type}): {e}")
                    continue


def render_all(
    derivatives_root: Path, base_output_dir: Path, subjects, surf_type: str = "inflated",
    no_contours: bool = False, save_png: bool = False, metric: str = "signed_log_p",
    fdr_q: float = 0.05, threshold: float = 3.0, panels=None,
) -> Path:
    """Render Figs 2 & 3 surface maps. Returns the output root dir.

    ``panels``: allow-list of ``(subject, contrast, hemi)`` tuples (default paper selection),
    or None to render the full 6×9×2 grid (``--exhaustive``).
    """
    surf_types = {"both": ("pial", "inflated"), "pial": ("pial",)}.get(surf_type, ("inflated",))
    data_dir = derivatives_root / "native_surface_projections"
    parcel_dir = None if no_contours else derivatives_root / "native_surface_parcels"
    output_dir = output_root(base_output_dir, metric, threshold, fdr_q, no_contours)

    mode = "EXHAUSTIVE full grid" if panels is None else f"paper panels ({len(panels)})"
    print("=" * 70)
    print("BRANCH B / 10: native-surface render (Figs 2 & 3)")
    print(f"  subjects={len(subjects)}  metric={metric}  surf={surf_type}  contours={not no_contours}")
    print(f"  selection={mode}" + ("" if panels is None else "  (use --exhaustive for all 108)"))
    print(f"  output={output_dir}")
    print("=" * 70)

    for subject in subjects:
        subject_surface_dir = derivatives_root / subject / "anat"
        for modality, contrasts in (("visual", VISUAL_CONTRASTS), ("auditory", AUDITORY_CONTRASTS)):
            visualize_subject(
                subject_surface_dir, data_dir, output_dir, subject, modality, contrasts,
                surf_types=surf_types, parcel_dir=parcel_dir, show_contours=not no_contours,
                save_png=save_png, metric=metric, fdr_q=fdr_q, threshold=threshold, panels=panels,
            )
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render individual native-surface maps — Figs 2 & 3 (Branch B, 10)."
    )
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="Subjects to render (default: derived from the paper-panel "
                             "selection, or all 6 with --exhaustive).")
    parser.add_argument("--exhaustive", action="store_true",
                        help="Render the FULL 6×9×2=108-panel grid (the matrix the paper's "
                             "panels were curated from). Default: only the published Figs 2 & "
                             "3 panels (PAPER_PANELS).")
    parser.add_argument("--derivatives-root", type=str, required=True)
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Base output dir; a subject_level_native_surface_T1w_* subdir is created under it.",
    )
    parser.add_argument("--pial", action="store_true", help="Pial surface instead of inflated.")
    parser.add_argument("--both-surfaces", action="store_true", help="Both pial and inflated.")
    parser.add_argument("--no-contours", action="store_true")
    parser.add_argument("--save-png", action="store_true", help="Also 6-view PNGs (inflated).")
    parser.add_argument(
        "--metric", type=str, default="signed_log_p", choices=["signed_log_p", "t_fdr"],
        help="signed_log_p (paper method, default) or t_fdr (dev exploratory).",
    )
    parser.add_argument("--fdr-q", type=float, default=0.05)
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--test", action="store_true", help="First subject only.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    surf_type = "both" if args.both_surfaces else ("pial" if args.pial else "inflated")

    # Panel selection: default = published Figs 2 & 3 panels; --exhaustive = full grid.
    panels = None if args.exhaustive else set(PAPER_PANELS)

    # Subjects: explicit --subjects wins; else derive from the panel set (or all 6 if
    # exhaustive). When --subjects narrows an active panel set, intersect the two.
    if args.subjects:
        subjects = args.subjects
        if panels is not None:
            panels = {p for p in panels if p[0] in set(subjects)}
    elif panels is not None:
        subjects = sorted({subj for subj, _, _ in panels})
    else:
        subjects = list(ALL_SUBJECTS)
    if args.test:
        subjects = subjects[:1]
        if panels is not None:
            panels = {p for p in panels if p[0] in set(subjects)}

    out = render_all(
        Path(args.derivatives_root), Path(args.output_dir), subjects, surf_type=surf_type,
        no_contours=args.no_contours, save_png=args.save_png, metric=args.metric,
        fdr_q=args.fdr_q, threshold=args.threshold, panels=panels,
    )
    print("=" * 70)
    print(f"RENDER COMPLETE -> {out}")


if __name__ == "__main__":
    main()
