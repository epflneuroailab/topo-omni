#!/usr/bin/env python3
"""Project volumetric stat maps onto each subject's native surface (Branch B, step 09).

For each subject / modality / contrast, sample the T1w-space concatenated-GLM stat
maps (tmap, pval, signed_log_p) at the vertices of that subject's native **pial**
surface, producing one ``.func.gii`` per hemisphere. This is the "analysis in the
volume, projected to the surface for visualization only" step of Marvi et al. (2025)
Figs 2 & 3.

  T1w volume stat map  --sample at pial RAS vertices (trilinear)-->  fsnative .func.gii

Lineage (README §9):  08 concat GLM (T1w) -> **09 project_to_native_surface** ->
11 inflated->GIFTI + 12 parcels->surface -> 10 render (Figs 2 & 3).
  input : <glm-dir>/<subj>/<modality>/<subj>_<modality>_<contrast>_concat_space-T1w_<map>.nii.gz
          + <derivatives-root>/<subj>/anat/<subj>_hemi-{L,R}_pial.surf.gii
  output: <output-dir>/<subj>/<modality>/<subj>_<modality>_<contrast>_hemi-{L,R}_<map>.func.gii

PORT NOTES vs dev-repo `src/09_project_volume_to_native_surface.py` (@ ef1da34):
  * Faithful port. The sampler is byte-for-byte the dev `sample_volume_at_coordinates`
    (RAS -> inv(affine) -> voxel -> `scipy.ndimage.map_coordinates(order=1,
    mode='constant', cval=0.0)`).  ⚠ Despite the dev docstring, this is NOT FreeSurfer
    `mri_vol2surf` — it is trilinear sampling directly at the pial vertices (projfrac ~ 0).
  * `project_volume_to_surface_data()` is side-effect-free (returns the sampled float32
    vector) so the golden master can compare against the published `.func.gii` without
    touching disk; the GIFTI wrap + save move to `main()`.
  * Parameterized by `--derivatives-root` / `--glm-dir` / `--output-dir` (were hard-coded
    dev paths). Contrast lists come from `emfl.config` (single source of truth; the dev
    inline lists are identical members).

DETERMINISM (docs/DESIGN.md §6): pure numpy + `scipy.ndimage.map_coordinates` — no nilearn, no
FreeSurfer CLI. This is a version-robust Tier-1 golden master (like Pernet cv_05 / Marvi
extract_condition_responses): feed the published T1w GLM maps + published pial surfaces
and reproduce the published surface projections to float precision.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

# Make `Marvi_2025/` importable so `emfl` + `config` resolve when run as a script.
_DATASET_DIR = Path(__file__).resolve().parent.parent
if str(_DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(_DATASET_DIR))

from emfl.config import (  # noqa: E402
    ALL_SUBJECTS,
    VISUAL_CONTRASTS,
    AUDITORY_CONTRASTS,
)

# Stat maps projected to the surface (dev 09 map_types), in dev order.
MAP_TYPES = ("tmap", "pval", "signed_log_p")

# Modality -> its contrast list (from config; identical to the dev 09 inline lists).
CONTRASTS_BY_MODALITY = {
    "visual": list(VISUAL_CONTRASTS),
    "auditory": list(AUDITORY_CONTRASTS),
}


def sample_volume_at_coordinates(
    volume_img: "nib.Nifti1Image", coordinates: np.ndarray
) -> np.ndarray:
    """Sample a volume at RAS coordinates via trilinear interpolation.

    Byte-faithful to dev `09.sample_volume_at_coordinates`: RAS -> inv(affine) ->
    voxel indices -> `map_coordinates(order=1, mode='constant', cval=0.0)`.
    """
    from scipy.ndimage import map_coordinates

    volume_data = volume_img.get_fdata()
    affine_inv = np.linalg.inv(volume_img.affine)

    coords_homogeneous = np.hstack(
        [coordinates, np.ones((coordinates.shape[0], 1))]
    )
    voxel_coords = (affine_inv @ coords_homogeneous.T).T[:, :3]

    sampled_data = map_coordinates(
        volume_data,
        voxel_coords.T,  # map_coordinates wants (ndim, n_points)
        order=1,  # trilinear
        mode="constant",
        cval=0.0,
    )
    return sampled_data


def project_volume_to_surface_data(volume_path: Path, surface_path: Path) -> np.ndarray:
    """Sample ``volume_path`` at the vertices of ``surface_path``; return float32 vector.

    Side-effect-free — no disk write (the golden master compares this directly).
    """
    volume_img = nib.load(str(volume_path))
    surface_img = nib.load(str(surface_path))
    coordinates = surface_img.darrays[0].data  # (n_vertices, 3) RAS
    sampled = sample_volume_at_coordinates(volume_img, coordinates)
    return sampled.astype(np.float32)


def surface_gifti_from_data(data: np.ndarray) -> "nib.gifti.GiftiImage":
    """Wrap a per-vertex float32 vector in a GIFTI image (dev 09 layout)."""
    darray = nib.gifti.GiftiDataArray(
        data=np.asarray(data, dtype=np.float32),
        intent="NIFTI_INTENT_NONE",
        datatype="NIFTI_TYPE_FLOAT32",
    )
    return nib.gifti.GiftiImage(darrays=[darray])


def pial_surface_path(derivatives_root: Path, subject: str, hemi: str) -> Path:
    """Path to a subject's native pial surface GIFTI (``hemi`` in {'L','R'})."""
    return (
        derivatives_root / subject / "anat" / f"{subject}_hemi-{hemi}_pial.surf.gii"
    )


def volume_map_path(
    glm_dir: Path, subject: str, modality: str, contrast: str, map_type: str
) -> Path:
    """Path to a T1w concatenated-GLM stat map (dev 08 output layout)."""
    return (
        glm_dir
        / subject
        / modality
        / f"{subject}_{modality}_{contrast}_concat_space-T1w_{map_type}.nii.gz"
    )


def surface_output_path(
    output_dir: Path, subject: str, modality: str, contrast: str, hemi: str, map_type: str
) -> Path:
    """Single source of truth for the on-disk surface-projection layout (dev 09)."""
    return (
        output_dir
        / subject
        / modality
        / f"{subject}_{modality}_{contrast}_hemi-{hemi}_{map_type}.func.gii"
    )


def project_subject(
    derivatives_root: Path,
    subject: str,
    glm_dir: Path,
    modality: str,
    contrasts,
    map_types=MAP_TYPES,
) -> dict:
    """Project all (contrast, map_type, hemi) maps for one subject/modality.

    Returns ``{(contrast, map_type, hemi): np.ndarray}``. Side-effect-free (no writes);
    silently skips contrast/map combinations whose volume is absent (mirrors dev 09,
    which prints "Volume not found" and continues).
    """
    pial = {h: pial_surface_path(derivatives_root, subject, h) for h in ("L", "R")}
    for h, p in pial.items():
        if not p.exists():
            raise FileNotFoundError(f"Native pial surface not found for {subject}: {p}")

    out = {}
    for contrast in contrasts:
        for map_type in map_types:
            vol = volume_map_path(glm_dir, subject, modality, contrast, map_type)
            if not vol.exists():
                continue
            for hemi in ("L", "R"):
                out[(contrast, map_type, hemi)] = project_volume_to_surface_data(
                    vol, pial[hemi]
                )
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Project T1w concatenated-GLM stat maps onto the native surface (Branch B, 09)."
    )
    parser.add_argument("--subjects", nargs="+", default=list(ALL_SUBJECTS))
    parser.add_argument(
        "--derivatives-root",
        type=str,
        required=True,
        help="Derivatives root (holds <subj>/anat/<subj>_hemi-*_pial.surf.gii).",
    )
    parser.add_argument(
        "--glm-dir",
        type=str,
        default=None,
        help="Concatenated-GLM dir (default: <derivatives-root>/concatenated_glm).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output dir (default: <derivatives-root>/native_surface_projections).",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["visual", "auditory"],
        choices=["visual", "auditory"],
    )
    parser.add_argument(
        "--contrasts",
        nargs="+",
        default=None,
        help="Subset of contrast keys to project (default: all for each modality).",
    )
    parser.add_argument("--map-types", nargs="+", default=list(MAP_TYPES))
    parser.add_argument(
        "--test",
        action="store_true",
        help="Only the first subject / first contrast of each modality.",
    )
    args = parser.parse_args(argv)

    derivatives_root = Path(args.derivatives_root)
    glm_dir = Path(args.glm_dir) if args.glm_dir else derivatives_root / "concatenated_glm"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else derivatives_root / "native_surface_projections"
    )

    subjects = args.subjects[:1] if args.test else args.subjects

    print("=" * 70)
    print("BRANCH B / 09: project volumetric maps -> native surface")
    print(f"  subjects={len(subjects)}  glm_dir={glm_dir}")
    print(f"  output_dir={output_dir}")
    print("=" * 70)

    n_written = 0
    for subject in subjects:
        for modality in args.modalities:
            contrasts = args.contrasts or CONTRASTS_BY_MODALITY[modality]
            if args.test:
                contrasts = list(contrasts)[:1]
            try:
                results = project_subject(
                    derivatives_root,
                    subject,
                    glm_dir,
                    modality,
                    contrasts,
                    map_types=args.map_types,
                )
            except FileNotFoundError as e:
                print(f"  ✗ {subject} {modality}: {e}")
                continue
            for (contrast, map_type, hemi), data in results.items():
                out_path = surface_output_path(
                    output_dir, subject, modality, contrast, hemi, map_type
                )
                out_path.parent.mkdir(parents=True, exist_ok=True)
                nib.save(surface_gifti_from_data(data), str(out_path))
                n_written += 1
            print(f"  ✓ {subject} {modality}: {len(results)} maps")

    print("=" * 70)
    print(f"PROJECTION COMPLETE — {n_written} surface maps written to {output_dir}")


if __name__ == "__main__":
    main()
