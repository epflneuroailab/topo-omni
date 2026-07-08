#!/usr/bin/env python3
"""Surface projection of the Pernet group maps -> Fig. 3b voice-selective map (step 02).

Projects the volumetric group maps to fsaverage6 (via the shared, version-robust
`core.surface`), saves the surface arrays consumed downstream by Fig. B3b, and renders
the Fig. 3b publication figure (FWE-corrected, positive-only voice-selective areas).

Lineage (docs/DESIGN.md §2.4):  01 group analysis  ->  **02 surface projection**.
  input : <results-root>/01_group_analysis/{t_map,z_map,thresholded_t_map}.nii.gz
  output: <results-root>/02_surface_projection/surface_data_fsaverage6.npz   (-> Fig. B3b)
          + fig3b_voice_selective_{left,right}_lateral.html  (Fig. 3b, interactive)

PORT NOTES vs dev-repo `src/02_surface_projection.py` (@ f842b1a):
  * The projection now calls `core.surface` (shared with Marvi/Jung) instead of an
    inline `vol_to_surf`; parameters are identical, so the npz reproduces the published
    file **bitwise** (golden master at atol=0).
  * npz keys match the dev file exactly: `t_map`, `z_map`, `thresholded` (the
    thresholded_t_map is stored under the shorter `thresholded` key, per the dev writer).
  * Scope-lock (docs/DESIGN.md §2.4/§8): only the paper figure (positive-only FWE map) is
    rendered. The dev script's percentile / Destrieux-A1-overlay / bidirectional /
    comprehensive variants are NOT ported.

The surface projection is nilearn-version-robust (calibrated: 0.12.1 reproduces the
0.10.4 npz exactly), so unlike step 01 this is safely testable under any pinned env.
STATUS: ported (Stage 1).
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from core import surface as surf

TEMPLATE = "fsaverage6"
# Volumetric group maps projected to surface, and the npz key each is stored under.
# (matches the published surface_data_fsaverage6.npz)
PROJECTED_MAPS = {"t_map": "t_map", "z_map": "z_map", "thresholded_t_map": "thresholded"}
FWE_THRESHOLD = 4.79    # Fig. 3b shows voice-selective vertices above this (from step 01)


def compute(results_root: Path, template=TEMPLATE):
    """Project the group maps to fsaverage and return ``{npz_key: {'lh','rh'}}``.

    Side-effect-free (reads inputs only) so the golden-master test can call it directly.
    """
    import nibabel as nib          # lazy
    from nilearn import datasets   # lazy

    results_root = Path(results_root)
    in_dir = results_root / "01_group_analysis"
    fsaverage = datasets.fetch_surf_fsaverage(template)

    surface_data = {}
    for nii_name, npz_key in PROJECTED_MAPS.items():
        img = nib.load(str(in_dir / f"{nii_name}.nii.gz"))
        surface_data[npz_key] = surf.project_to_fsaverage(img, template, fsaverage=fsaverage)
    return surface_data


def save_npz(surface_data, out_dir: Path, template=TEMPLATE) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_dict = {}
    for npz_key, hemis in surface_data.items():
        save_dict[f"{npz_key}_lh"] = hemis["lh"]
        save_dict[f"{npz_key}_rh"] = hemis["rh"]
    save_dict["_metadata_fsaverage_template"] = np.array([template], dtype="U20")
    save_dict["_metadata_creation_time"] = np.array([datetime.now().isoformat()], dtype="U50")
    out = out_dir / f"surface_data_{template}.npz"
    np.savez_compressed(out, **save_dict)
    return out


def load_npz(npz_path: Path):
    """Inverse of ``save_npz``: load surface_data_{template}.npz -> {key: {'lh','rh'}}.

    Lets Fig. 3b re-render from the SHIPPED surface npz (the precomputed cut) without
    re-running the heavy 218-subject group GLM (01) + projection (02).
    """
    data = np.load(str(npz_path))
    out = {}
    for full_key in data.files:
        npz_key, _, hemi = full_key.rpartition("_")   # 't_map_lh' -> ('t_map','lh')
        out.setdefault(npz_key, {})[hemi] = data[full_key]
    return out


def render_fig3b(surface_data, out_dir: Path, template=TEMPLATE):
    """Render Fig. 3b: positive-only FWE voice-selective map, both hemispheres, as
    **interactive HTML** (nilearn ``view_surf`` on the inflated fsaverage6 surface).

    Interactive HTML (not static SVG) so a reviewer can rotate/inspect the surface — matching
    the Marvi/Jung surface figures. The colormap/threshold/vmax mirror the former SVG look.
    """
    import nibabel as nib          # lazy (keep the module import-light — core invariant)
    from nilearn import datasets, plotting

    fsaverage = datasets.fetch_surf_fsaverage(template)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bg = {"lh": surf.binarize_sulcal_map(nib.load(fsaverage["sulc_left"]).darrays[0].data),
          "rh": surf.binarize_sulcal_map(nib.load(fsaverage["sulc_right"]).darrays[0].data)}
    t_surf = surface_data["t_map"]
    written = []
    for hemi, key, infl in [("left", "lh", "infl_left"), ("right", "rh", "infl_right")]:
        data = t_surf[key]
        # Positive-only FWE: show only voice-selective vertices above threshold.
        voice = np.where(data > FWE_THRESHOLD, data, np.nan)
        view = plotting.view_surf(
            surf_mesh=fsaverage[infl], surf_map=voice, bg_map=bg[key],
            threshold=None, cmap="viridis", symmetric_cmap=False, vmin=0, vmax=16,
            title=f"{hemi.capitalize()} hemisphere: Voice > Non-voice (FWE p<0.05)",
        )
        f = out_dir / f"fig3b_voice_selective_{hemi}_lateral.html"
        view.save_as_html(str(f))
        written.append(f)
    return written


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", type=Path, required=True,
                   help="Root holding 01_group_analysis/ (input) and 02_surface_projection/ (output).")
    p.add_argument("--template", default=TEMPLATE)
    p.add_argument("--no-figure", action="store_true", help="Write the npz only, skip the HTML render.")
    p.add_argument("--from-npz", action="store_true",
                   help="Render Fig. 3b from the already-shipped surface npz (skip the heavy 01->02 "
                        "recompute). Requires <results-root>/02_surface_projection/surface_data_*.npz.")
    p.add_argument("--plots-dir", type=Path, default=None,
                   help="Directory for the rendered Fig. 3b HTML (default: alongside the npz under "
                        "<results-root>/02_surface_projection). The npz always stays under --results-root.")
    args = p.parse_args(argv)

    out_dir = args.results_root / "02_surface_projection"
    plots_dir = args.plots_dir or out_dir
    if args.from_npz:
        npz_path = out_dir / f"surface_data_{args.template}.npz"
        if not npz_path.exists():
            raise SystemExit(f"--from-npz needs {npz_path} (the shipped surface cut).")
        surface_data = load_npz(npz_path)
        print(f"Surface data <- {npz_path} (render-only, no GLM recompute)")
    else:
        surface_data = compute(args.results_root, args.template)
        npz = save_npz(surface_data, out_dir, args.template)
        print(f"Surface data -> {npz}")
    if not args.no_figure:
        for f in render_fig3b(surface_data, plots_dir, args.template):
            print(f"Fig. 3b -> {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
