#!/usr/bin/env python3
"""Calibrate the heavy Pernet golden-master tolerances (steps 01 + cv_04).

The two heavy golden masters (test_group_analysis_golden.py, test_cv_group_froi_golden.py)
keep their tight assertions `@pytest.mark.skip`ped pending a *measured* tolerance, because
`SecondLevelModel` is a dataset-specific GLM engine pinned to Pernet's nilearn 0.10.4 and
is NOT assumed version-robust (docs/DESIGN.md §2.2/§6). This script produces the numbers needed
to freeze those tolerances by re-running both drivers against the published reference maps
and reporting, per output:

  step 01  : shape, Pearson r, max|Δ| for t/z/p maps; cluster count; peak t.
  step cv_04: per fold — r, max|Δ| on the group t-map; n_froi_voxels vs published
              (A=2185, B=2944 @ t>4.79); Dice vs the published fROI mask.

Run it under BOTH pinned envs and diff the JSON:
  * nilearn 0.10.4 (Pernet's env)  -> the calibration reference; its max|Δ| sets ATOL.
  * nilearn 0.12.1 (Marvi/Jung env)-> shows whether the engine is version-portable at all.

Usage:
  python calibrate_heavy_golden.py \
      --results-root /work/.../20241003_pernet_2015/results \
      --out /path/to/calib_<envtag>.json

Reads inputs only (drivers' compute() are side-effect-free); writes a single JSON report.
This is the harness referenced by the two heavy golden tests' TODO(calibrate-0.10.4).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_DATASET = Path(__file__).resolve().parent.parent
_ANALYSIS = _DATASET / "analysis"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _ANALYSIS / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _env_info() -> dict:
    import nibabel
    import nilearn
    import numpy
    import scipy
    return {
        "python": sys.version.split()[0],
        "nilearn": nilearn.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "nibabel": nibabel.__version__,
        "executable": sys.executable,
    }


def _compare_maps(got: np.ndarray, ref: np.ndarray) -> dict:
    """Pearson r + max|Δ| over jointly-finite voxels."""
    m = np.isfinite(got) & np.isfinite(ref)
    n = int(m.sum())
    r = float(np.corrcoef(got[m], ref[m])[0, 1]) if n > 1 else float("nan")
    max_abs_delta = float(np.max(np.abs(got[m] - ref[m]))) if n else float("nan")
    return {
        "shape": list(got.shape),
        "n_finite": n,
        "pearson_r": r,
        "max_abs_delta": max_abs_delta,
    }


def calibrate_01(results_root: Path) -> dict:
    import nibabel as nib
    driver = _load("pernet_01", "01_group_analysis.py")
    maps, clusters, summary = driver.compute(results_root)
    gold_dir = results_root / "01_group_analysis"
    out = {"n_subjects": summary["n_subjects"], "n_clusters": summary["n_clusters"], "maps": {}}
    for name in ("t_map", "z_map", "p_map"):
        got = maps[name].get_fdata()
        ref = nib.load(str(gold_dir / f"{name}.nii.gz")).get_fdata()
        out["maps"][name] = _compare_maps(got, ref)
    out["peak_stat"] = float(clusters["Peak Stat"].max()) if len(clusters) else None
    return out


def calibrate_cv04(results_root: Path) -> dict:
    import nibabel as nib
    driver = _load("pernet_cv04", "cv_04_group_froi_analysis.py")
    results = driver.compute(results_root)
    group = results_root / "04_cross_validation" / "group"
    published = {"A": 2185, "B": 2944}
    out = {}
    for fold in ("A", "B"):
        r = results[fold]
        got_t = r["t_map"].get_fdata()
        ref_t = nib.load(str(group / f"half-{fold}_t_map.nii.gz")).get_fdata()
        got_mask = r["mask"].get_fdata().astype(bool)
        ref_mask = nib.load(str(group / f"half-{fold}_fROI_mask.nii.gz")).get_fdata().astype(bool)
        denom = got_mask.sum() + ref_mask.sum()
        dice = float(2 * (got_mask & ref_mask).sum() / denom) if denom else float("nan")
        out[fold] = {
            "n_subjects": r["n_subjects"],
            "t_map": _compare_maps(got_t, ref_t),
            "n_froi_voxels": r["n_froi_voxels"],
            "published_voxels": published[fold],
            "voxel_delta": r["n_froi_voxels"] - published[fold],
            "dice_vs_published_mask": dice,
        }
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", type=Path, required=True,
                   help="Source root with 00_volumetric_GLM/, 01_group_analysis/, "
                        "04_cross_validation/{per_subject,group}/.")
    p.add_argument("--out", type=Path, required=True, help="Where to write the JSON report.")
    p.add_argument("--steps", default="01,cv04", help="Comma list: 01, cv04 (default both).")
    args = p.parse_args(argv)

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    report = {"env": _env_info(), "results_root": str(args.results_root), "started": _now()}
    print(f"[calibrate] env: nilearn {report['env']['nilearn']} / "
          f"numpy {report['env']['numpy']} / py {report['env']['python']}", flush=True)

    if "01" in steps:
        print("[calibrate] step 01 — 218-subject SecondLevelModel ...", flush=True)
        report["step_01"] = calibrate_01(args.results_root)
        m = report["step_01"]["maps"]["t_map"]
        print(f"[calibrate] 01 t_map: r={m['pearson_r']:.8f} max|Δ|={m['max_abs_delta']:.3e} "
              f"clusters={report['step_01']['n_clusters']} "
              f"peak={report['step_01']['peak_stat']}", flush=True)

    if "cv04" in steps:
        print("[calibrate] step cv_04 — 2x 218-subject half-split fits ...", flush=True)
        report["step_cv04"] = calibrate_cv04(args.results_root)
        for fold in ("A", "B"):
            f = report["step_cv04"][fold]
            print(f"[calibrate] cv_04 fold {fold}: r={f['t_map']['pearson_r']:.8f} "
                  f"max|Δ|={f['t_map']['max_abs_delta']:.3e} "
                  f"fROI={f['n_froi_voxels']} (pub {f['published_voxels']}, "
                  f"Δ{f['voxel_delta']:+d}) dice={f['dice_vs_published_mask']:.6f}", flush=True)

    report["finished"] = _now()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"[calibrate] wrote {args.out}", flush=True)
    return 0


def _now() -> str:
    # Date.now()-free environments still allow datetime.now() outside workflow scripts.
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    raise SystemExit(main())
