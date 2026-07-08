#!/usr/bin/env python3
"""Convert FreeSurfer inflated surfaces to GIFTI display meshes (Branch B, step 11).

The Figs 2 & 3 renderer (step 10) draws the thresholded surface maps on each subject's
*inflated* fsnative mesh. FreeSurfer stores the inflated geometry in its own binary
format (`surf/{lh,rh}.inflated`); this step reads it and writes a GIFTI display mesh.

  FreeSurfer surf/{lh,rh}.inflated  --nibabel.read_geometry-->  <subj>_hemi-{L,R}_inflated.surf.gii

Lineage (README §9):  independent of the GLM; needed by step 10 (render). Runs on the
FreeSurfer recon-all output shipped under `derivatives/sourcedata/freesurfer/`.
  input : <freesurfer-dir>/<subj>/surf/{lh,rh}.inflated
  output: <output-dir>/<subj>/anat/<subj>_hemi-{L,R}_inflated.surf.gii

PORT NOTES vs dev-repo `src/11_convert_inflated_surfaces.py` (@ ef1da34):
  * Faithful port. Byte-for-byte the dev conversion: `nibabel.freesurfer.read_geometry`
    -> a GIFTI with a POINTSET(float32 coords) + TRIANGLE(int32 faces) darray.
    ⚠ Despite what one might assume, this is NOT `mris_convert` — it is a pure-Python
    nibabel read + GIFTI write (no FreeSurfer CLI, no container).
  * `inflated_gifti_from_freesurfer()` is side-effect-free (returns the GiftiImage) so
    the golden master can compare coords/faces against the published GIFTI without
    touching disk; the save moves to `main()`.
  * Parameterized by `--freesurfer-dir` / `--output-dir` (were hard-coded dev paths).

DETERMINISM (docs/DESIGN.md §6): pure nibabel geometry read + array cast — fully deterministic.
Tier-1 bitwise golden vs the published `anat/*_inflated.surf.gii`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import nibabel as nib

# Subjects default to the published 6-subject public subset.
ALL_SUBJECTS = (
    "sub-kaneff01",
    "sub-kaneff06",
    "sub-kaneff07",
    "sub-kaneff08",
    "sub-kaneff09",
    "sub-kaneff21",
)

# FreeSurfer hemi label -> BIDS hemi label.
HEMIS = (("lh", "L"), ("rh", "R"))


def inflated_gifti_from_freesurfer(fs_surf_path: Path) -> "nib.gifti.GiftiImage":
    """Read a FreeSurfer surface and return a GIFTI display mesh (dev 11 layout).

    Side-effect-free. POINTSET (float32 coords) + TRIANGLE (int32 faces), in that order.
    """
    coords, faces = nib.freesurfer.read_geometry(str(fs_surf_path))
    coord_array = nib.gifti.GiftiDataArray(
        data=coords.astype("float32"),
        intent="NIFTI_INTENT_POINTSET",
        datatype="NIFTI_TYPE_FLOAT32",
    )
    face_array = nib.gifti.GiftiDataArray(
        data=faces.astype("int32"),
        intent="NIFTI_INTENT_TRIANGLE",
        datatype="NIFTI_TYPE_INT32",
    )
    return nib.gifti.GiftiImage(darrays=[coord_array, face_array])


def freesurfer_inflated_path(freesurfer_dir: Path, subject: str, hemi_fs: str) -> Path:
    """Path to a FreeSurfer inflated surface (``hemi_fs`` in {'lh','rh'})."""
    return freesurfer_dir / subject / "surf" / f"{hemi_fs}.inflated"


def inflated_output_path(output_dir: Path, subject: str, hemi_bids: str) -> Path:
    """Single source of truth for the on-disk inflated-GIFTI layout (dev 11)."""
    return (
        output_dir / subject / "anat" / f"{subject}_hemi-{hemi_bids}_inflated.surf.gii"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert FreeSurfer inflated surfaces to GIFTI (Branch B, 11)."
    )
    parser.add_argument("--subjects", nargs="+", default=list(ALL_SUBJECTS))
    parser.add_argument(
        "--freesurfer-dir",
        type=str,
        default=None,
        help="FreeSurfer recon-all dir (default: <derivatives-root>/sourcedata/freesurfer).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output derivatives dir holding <subj>/anat/ (default: <derivatives-root>).",
    )
    parser.add_argument(
        "--derivatives-root",
        type=str,
        default=None,
        help="Derivatives root; used to default --freesurfer-dir and --output-dir.",
    )
    args = parser.parse_args(argv)

    if args.derivatives_root:
        root = Path(args.derivatives_root)
    elif args.output_dir:
        root = Path(args.output_dir)
    else:
        parser.error("provide --derivatives-root (or both --freesurfer-dir and --output-dir)")

    freesurfer_dir = (
        Path(args.freesurfer_dir)
        if args.freesurfer_dir
        else root / "sourcedata" / "freesurfer"
    )
    output_dir = Path(args.output_dir) if args.output_dir else root

    print("=" * 70)
    print("BRANCH B / 11: FreeSurfer inflated surfaces -> GIFTI")
    print(f"  subjects={len(args.subjects)}  freesurfer_dir={freesurfer_dir}")
    print(f"  output_dir={output_dir}")
    print("=" * 70)

    n_written = 0
    for subject in args.subjects:
        fs_surf_dir = freesurfer_dir / subject / "surf"
        if not fs_surf_dir.exists():
            print(f"  ✗ {subject}: FreeSurfer surf/ not found: {fs_surf_dir}")
            continue
        for hemi_fs, hemi_bids in HEMIS:
            fs_path = freesurfer_inflated_path(freesurfer_dir, subject, hemi_fs)
            if not fs_path.exists():
                print(f"  ✗ {subject} {hemi_fs}.inflated not found")
                continue
            gii = inflated_gifti_from_freesurfer(fs_path)
            out_path = inflated_output_path(output_dir, subject, hemi_bids)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(gii, str(out_path))
            n_written += 1
            print(f"  ✓ {subject} {hemi_bids}: {out_path.name}")

    print("=" * 70)
    print(f"CONVERSION COMPLETE — {n_written} inflated GIFTIs written to {output_dir}")


if __name__ == "__main__":
    main()
