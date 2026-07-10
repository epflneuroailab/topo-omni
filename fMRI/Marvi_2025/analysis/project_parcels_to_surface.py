#!/usr/bin/env python3
"""Project anatomical ROI parcels to each subject's native surface (Branch B, step 12).

The Figs 2 & 3 renders overlay independent-study ROI **contours** on the activation
maps. This step warps each parcel from its template space into the subject's T1w volume
(FreeSurfer CLI), samples it onto the pial surface, thresholds, unions the parcels for a
contrast, and traces the boundary contour.

  parcel (CVS / CVS-MNI152 / MNI152)  --mri_vol2vol (inverse warp)-->  T1w volume
  T1w volume  --mri_vol2surf --projfrac 0.5-->  pial-surface mask  --> boundary contour

Transform routing (per parcel subdir; see docs/PARCEL_TRANSFORMATION_FIX.md):
  * julian (FFA/OFA/fSTS/PPA/OPA/RSC/EBA/LOC)  -> CVS warp inverse (tocvs_avg35.m3z --inv-morph)
  * vwfa, md                                    -> CVS-MNI152 warp inverse (tocvs_avg35_inMNI152.m3z)
  * language, speech, tom                       -> affine inverse (talairach.lta --lta-inv)
  ⚠ MNI->T1w must use `--lta-inv` (resamples), NOT `--regheader` (header-only → misplaced
    ROIs); `--regheader` is correct ONLY for the already-native T1w->surface step.

Lineage (README §9): needs the CVS `.m3z` warps from Stage-0 step 16
(`cvs_transforms/`). Feeds step 10 (render) as the contour overlay.
  input : <parcels-dir>/<subdir>/<parcel>.nii.gz  (vendored data/PARCELS/)
          <freesurfer-dir>/<subj>/mri/{orig.mgz,transforms/talairach.lta}
          <cvs-transforms-dir>/<subj>/{tocvs_avg35,tocvs_avg35_inMNI152}/final_CVSmorph_*.m3z
          <derivatives-root>/<subj>/anat/<subj>_hemi-{L,R}_pial.surf.gii
  output: <output-dir>/<subj>/<contrast>/<subj>_<contrast>_hemi-{L,R}_parcel_contour.func.gii

PORT NOTES vs dev-repo `src/12_project_parcels_to_native_surface.py` + wrapper
`src/18_project_parcels_to_surface.sh` (@ ef1da34):
  * Faithful port. The FreeSurfer CLI command builders (`mri_vol2vol` CVS/affine,
    `mri_vol2surf`) are byte-for-byte, incl. flags (`--noDefM3zPath`, `--inv-morph`,
    `--lta-inv`, `--nearest`, `--projfrac 0.5`, `--surf pial`, `--noreshape`, `--sd`).
    The parcel routing maps (CONTRAST_PARCELS / SUBDIR_TRANSFORM / PARCEL_SUBDIR_OVERRIDE)
    are 12-specific and kept inline unchanged. `create_contour_from_mask` is pure numpy.
  * Parameterized paths; `--parcels-dir` defaults to the vendored `config.get_parcels_dir()`.
  * `contour_from_surface_masks()` (pure numpy: threshold>0.5, union, trace boundary) is
    split out so it is unit-testable without FreeSurfer.
  * ⚠ CONTAINER: dev wrapper 18 binds `fmriprep-24.0.1.simg` (bundles FreeSurfer), while
    the 12 docstring references the standalone `freesurfer_7.3.2.sif`. Either works (both
    provide mri_vol2vol / mri_vol2surf). Set `--container` / run inside one; the script
    pre-flights `which mri_vol2vol` and exits with the apptainer invocation if absent.

DETERMINISM (docs/DESIGN.md §6): FreeSurfer-CLI heavy → **Stage-0-ish, NOT bitwise golden-mastered**.
Deliverable = faithful port + provenance spot-check (contour vertices vs the published
`native_surface_parcels/`) when a FreeSurfer container is available; the pure-numpy contour
tracer has its own deterministic unit test.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import nibabel as nib

# Make `Marvi_2025/` importable so `config` resolves when run as a script.
_DATASET_DIR = Path(__file__).resolve().parent.parent
if str(_DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(_DATASET_DIR))

from emfl.config import get_parcels_dir  # noqa: E402

ALL_SUBJECTS = (
    "sub-kaneff01",
    "sub-kaneff06",
    "sub-kaneff07",
    "sub-kaneff08",
    "sub-kaneff09",
    "sub-kaneff21",
)

# ---------------------------------------------------------------------------
# Parcel -> contrast mapping + transform routing (dev 12, verbatim)
# ---------------------------------------------------------------------------
CONTRAST_PARCELS = {
    "faces_vs_objects":  {"L": ["lh.ffa", "lh.ofa", "lh.sts"], "R": ["rh.ffa", "rh.ofa", "rh.sts"]},
    "scenes_vs_objects": {"L": ["lh.ppa", "lh.opa", "lh.rsc"], "R": ["rh.ppa", "rh.opa", "rh.rsc"]},
    "bodies_vs_objects": {"L": ["lh.eba"], "R": ["rh.eba"]},
    "words_vs_objects":  {"L": ["lh.vwfa"], "R": []},  # no rh.vwfa parcel exists
    "objects_vs_words":  {"L": ["lh.loc"], "R": ["rh.loc"]},
    "false_belief_vs_false_photo": {"L": [], "R": ["rh.tpj"]},
    "english_vs_nonwords": {
        "L": ["lh.ifg", "lh.ifgorb", "lh.mfg", "lh.anttemp", "lh.posttemp", "lh.ag"],
        "R": [],
    },
    "nonwords_vs_quilted": {"L": ["speech"], "R": ["speech"]},
    "math_vs_theory_of_mind": {
        "L": ["lh.antparietal", "lh.midparietal", "lh.postparietal", "lh.supfrontal", "lh.midfrontal"],
        "R": ["rh.antparietal", "rh.midparietal", "rh.postparietal", "rh.supfrontal", "rh.midfrontal"],
    },
}

# Transform per parcel subdir: cvs=tocvs_avg35.m3z; cvs_mni=tocvs_avg35_inMNI152.m3z; affine=talairach.lta.
SUBDIR_TRANSFORM = {
    "julian": "cvs",
    "vwfa": "cvs_mni",
    "md": "cvs_mni",
    "language": "affine",
    "speech": "affine",
    "tom": "affine",
}

# Parcels present in multiple subdirs — force the correct one (dev 12).
PARCEL_SUBDIR_OVERRIDE = {
    "lh.vwfa": "vwfa",   # also in julian/ (CVS); must use vwfa/ (CVS-MNI152)
    "rh.sts": "julian",  # also in tom/ (ToM-STS); used only for faces -> julian/ (face-STS)
}


def find_parcel_file(parcel_name: str, parcels_dir: Path):
    """Return (path, subdir) for a parcel, honouring overrides then search order (dev 12)."""
    if parcel_name in PARCEL_SUBDIR_OVERRIDE:
        subdir = PARCEL_SUBDIR_OVERRIDE[parcel_name]
        parcel_file = parcels_dir / subdir / f"{parcel_name}.nii.gz"
        if parcel_file.exists():
            return parcel_file, subdir
    for subdir in ["vwfa", "language", "tom", "md", "speech", "julian"]:
        parcel_file = parcels_dir / subdir / f"{parcel_name}.nii.gz"
        if parcel_file.exists():
            return parcel_file, subdir
    return None, None


def _run(cmd, freesurfer_dir: Path, what: str):
    env = os.environ.copy()
    env["SUBJECTS_DIR"] = str(freesurfer_dir)
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{what} failed:\n{result.stderr}")


def transform_cvs_to_t1w(parcel_path, subject, freesurfer_dir, cvs_m3z_path, output_path):
    """CVS/CVS-MNI152 parcel -> T1w via `mri_vol2vol --m3z ... --inv-morph` (dev 12)."""
    orig_mgz = freesurfer_dir / subject / "mri" / "orig.mgz"
    if not orig_mgz.exists():
        raise FileNotFoundError(f"orig.mgz not found: {orig_mgz}")
    if not Path(cvs_m3z_path).exists():
        raise FileNotFoundError(f"CVS warp not found: {cvs_m3z_path}")
    cmd = [
        "mri_vol2vol", "--noDefM3zPath",
        "--mov", str(orig_mgz),
        "--targ", str(parcel_path),
        "--m3z", str(cvs_m3z_path),
        "--inv-morph", "--nearest",
        "--o", str(output_path),
        "--sd", str(freesurfer_dir),
    ]
    _run(cmd, freesurfer_dir, "mri_vol2vol (CVS)")


def transform_affine_to_t1w(parcel_path, subject, freesurfer_dir, output_path):
    """MNI152 parcel -> T1w via `mri_vol2vol --lta-inv talairach.lta` (dev 12)."""
    talairach_lta = freesurfer_dir / subject / "mri" / "transforms" / "talairach.lta"
    if not talairach_lta.exists():
        raise FileNotFoundError(f"talairach.lta not found: {talairach_lta}")
    cmd = [
        "mri_vol2vol",
        "--mov", str(parcel_path),
        "--targ", str(freesurfer_dir / subject / "mri" / "orig.mgz"),
        "--lta-inv", str(talairach_lta),
        "--nearest",
        "--o", str(output_path),
        "--sd", str(freesurfer_dir),
    ]
    _run(cmd, freesurfer_dir, "mri_vol2vol (affine)")


def project_t1w_to_surface(t1w_parcel_path, subject, hemi, freesurfer_dir, output_path):
    """T1w parcel -> pial surface via `mri_vol2surf --regheader --projfrac 0.5` (dev 12)."""
    cmd = [
        "mri_vol2surf",
        "--mov", str(t1w_parcel_path),
        "--regheader", subject,
        "--hemi", hemi,
        "--projfrac", "0.5",
        "--surf", "pial",
        "--o", str(output_path),
        "--noreshape",
        "--sd", str(freesurfer_dir),
    ]
    _run(cmd, freesurfer_dir, "mri_vol2surf")


def create_contour_from_mask(mask: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Mark the vertices of any triangle spanning a mask boundary (dev 12). Pure numpy."""
    contour = np.zeros_like(mask)
    for face in faces:
        v0, v1, v2 = face
        if len({mask[v0], mask[v1], mask[v2]}) > 1:
            contour[v0] = 1
            contour[v1] = 1
            contour[v2] = 1
    return contour


def contour_from_surface_masks(surf_datas, faces: np.ndarray, n_vertices: int) -> np.ndarray:
    """Threshold (>0.5), union, and trace the boundary of per-parcel surface samples.

    Pure numpy (unit-testable without FreeSurfer). ``surf_datas`` = iterable of the raw
    ``mri_vol2surf`` per-parcel surface vectors for one hemisphere.
    """
    combined = np.zeros(n_vertices, dtype=np.float32)
    for surf_data in surf_datas:
        mask = (np.asarray(surf_data).squeeze() > 0.5).astype(np.float32)
        combined = np.maximum(combined, mask)
    if np.sum(combined) == 0:
        return combined  # empty; caller skips
    return create_contour_from_mask(combined, faces)


def parcel_contour_output_path(output_dir: Path, subject: str, contrast: str, hemi: str) -> Path:
    """Single source of truth for the on-disk parcel-contour layout (dev 12)."""
    return (
        output_dir
        / subject
        / contrast
        / f"{subject}_{contrast}_hemi-{hemi}_parcel_contour.func.gii"
    )


def process_subject_contrast(
    subject, contrast, parcels_dir, derivatives_root, freesurfer_dir,
    cvs_transforms_dir, output_dir, temp_dir,
):
    """Project all parcels for one subject/contrast to native surface (faithful to dev 12)."""
    if contrast not in CONTRAST_PARCELS:
        print(f"    ⚠ {contrast}: no parcels defined")
        return
    parcel_map = CONTRAST_PARCELS[contrast]
    anat_dir = derivatives_root / subject / "anat"

    for hemi_short, hemi_fs in [("L", "lh"), ("R", "rh")]:
        parcel_names = parcel_map[hemi_short]
        if not parcel_names:
            continue
        pial_path = anat_dir / f"{subject}_hemi-{hemi_short}_pial.surf.gii"
        if not pial_path.exists():
            print(f"    ✗ surface not found: {pial_path.name}")
            continue

        surf_img = nib.load(str(pial_path))
        faces = surf_img.darrays[1].data
        n_vertices = len(surf_img.darrays[0].data)

        surf_datas = []
        for parcel_name in parcel_names:
            parcel_file, subdir = find_parcel_file(parcel_name, parcels_dir)
            if parcel_file is None:
                print(f"      ⚠ parcel not found: {parcel_name}")
                continue
            transform_type = SUBDIR_TRANSFORM.get(subdir, "affine")
            try:
                t1w_parcel = temp_dir / f"{subject}_{parcel_name}_T1w.mgz"
                if transform_type == "cvs":
                    m3z = cvs_transforms_dir / subject / "tocvs_avg35" / "final_CVSmorph_tocvs_avg35.m3z"
                    transform_cvs_to_t1w(parcel_file, subject, freesurfer_dir, m3z, t1w_parcel)
                elif transform_type == "cvs_mni":
                    m3z = cvs_transforms_dir / subject / "tocvs_avg35_inMNI152" / "final_CVSmorph_tocvs_avg35_inMNI152.m3z"
                    transform_cvs_to_t1w(parcel_file, subject, freesurfer_dir, m3z, t1w_parcel)
                else:
                    transform_affine_to_t1w(parcel_file, subject, freesurfer_dir, t1w_parcel)

                surf_parcel = temp_dir / f"{subject}_{parcel_name}_{hemi_fs}.mgh"
                project_t1w_to_surface(t1w_parcel, subject, hemi_fs, freesurfer_dir, surf_parcel)
                surf_datas.append(nib.load(str(surf_parcel)).get_fdata().squeeze())
                print(f"      ✓ {parcel_name} [{transform_type}]")
            except Exception as e:  # noqa: BLE001 — dev per-parcel skip semantics
                print(f"      ✗ failed {parcel_name}: {e}")
                continue

        contour = contour_from_surface_masks(surf_datas, faces, n_vertices)
        if np.sum(contour) == 0:
            print(f"    ⚠ {hemi_short}: no parcel vertices after projection")
            continue

        out_path = parcel_contour_output_path(output_dir, subject, contrast, hemi_short)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        darray = nib.gifti.GiftiDataArray(
            data=contour.astype(np.float32),
            intent="NIFTI_INTENT_NONE",
            datatype="NIFTI_TYPE_FLOAT32",
        )
        nib.save(nib.gifti.GiftiImage(darrays=[darray]), str(out_path))
        print(f"    ✓ {hemi_short}: {int(contour.sum())} contour verts -> {out_path.name}")


def _freesurfer_tools_available() -> bool:
    for tool in ("mri_vol2vol", "mri_vol2surf"):
        if subprocess.run(["which", tool], capture_output=True).returncode != 0:
            return False
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Project anatomical parcels to native surface (Branch B, 12; FreeSurfer CLI)."
    )
    parser.add_argument("--subjects", nargs="+", default=list(ALL_SUBJECTS))
    parser.add_argument("--derivatives-root", type=str, required=True)
    parser.add_argument(
        "--parcels-dir", type=str, default=None,
        help="Parcels dir (default: vendored config.get_parcels_dir()).",
    )
    parser.add_argument(
        "--freesurfer-dir", type=str, default=None,
        help="FreeSurfer recon-all dir (default: <derivatives-root>/sourcedata/freesurfer).",
    )
    parser.add_argument(
        "--cvs-transforms-dir", type=str, default=None,
        help="CVS warp dir (default: <derivatives-root>/cvs_transforms).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output dir (default: <derivatives-root>/native_surface_parcels).",
    )
    parser.add_argument("--contrasts", nargs="+", default=None)
    parser.add_argument("--test", action="store_true", help="First subject / first contrast.")
    args = parser.parse_args(argv)

    derivatives_root = Path(args.derivatives_root)
    parcels_dir = Path(args.parcels_dir) if args.parcels_dir else Path(get_parcels_dir())
    freesurfer_dir = (
        Path(args.freesurfer_dir) if args.freesurfer_dir
        else derivatives_root / "sourcedata" / "freesurfer"
    )
    cvs_transforms_dir = (
        Path(args.cvs_transforms_dir) if args.cvs_transforms_dir
        else derivatives_root / "cvs_transforms"
    )
    output_dir = (
        Path(args.output_dir) if args.output_dir
        else derivatives_root / "native_surface_parcels"
    )

    subjects = [args.subjects[0]] if args.test else args.subjects
    contrasts = args.contrasts or list(CONTRAST_PARCELS.keys())
    if args.test:
        contrasts = contrasts[:1]

    if not _freesurfer_tools_available():
        print("✗ mri_vol2vol / mri_vol2surf not on PATH — run inside a FreeSurfer container, e.g.:")
        print("  apptainer exec --cleanenv \\")
        print("    --bind {FS_LICENSE}:/opt/freesurfer/license.txt:ro \\")
        print("    --bind /work/upschrimpf1:/work/upschrimpf1 \\")
        print("    --env FS_LICENSE=/opt/freesurfer/license.txt \\")
        print("    --env SUBJECTS_DIR={freesurfer_dir} \\")
        print("    /work/upschrimpf1/mehrer/fmriprep-24.0.1.simg \\")
        print("    python3 analysis/project_parcels_to_surface.py --derivatives-root ...")
        sys.exit(2)

    temp_dir = Path(tempfile.mkdtemp(prefix="parcel_transform_"))
    print("=" * 70)
    print("BRANCH B / 12: parcels -> native surface (contours)")
    print(f"  subjects={subjects}  contrasts={len(contrasts)}  parcels_dir={parcels_dir}")
    print(f"  temp_dir={temp_dir}  output_dir={output_dir}")
    print("=" * 70)

    try:
        for subject in subjects:
            cvs_dir = cvs_transforms_dir / subject
            m3z_cvs = cvs_dir / "tocvs_avg35" / "final_CVSmorph_tocvs_avg35.m3z"
            m3z_mni = cvs_dir / "tocvs_avg35_inMNI152" / "final_CVSmorph_tocvs_avg35_inMNI152.m3z"
            if not m3z_cvs.exists() or not m3z_mni.exists():
                print(f"  ✗ {subject}: CVS warp missing (run Stage-0 step 16 first): {cvs_dir}")
                continue
            print(f"\n{subject}")
            for contrast in contrasts:
                try:
                    process_subject_contrast(
                        subject, contrast, parcels_dir, derivatives_root,
                        freesurfer_dir, cvs_transforms_dir, output_dir, temp_dir,
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"  ✗ FAILED {contrast}: {e}")
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("=" * 70)
    print("PARCEL PROJECTION COMPLETE")


if __name__ == "__main__":
    main()
