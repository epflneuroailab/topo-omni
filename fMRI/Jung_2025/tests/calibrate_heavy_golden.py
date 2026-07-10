#!/usr/bin/env python3
"""Calibrate the heavy Jung group-map golden tolerance (new54 family).

The group cluster-contrast golden (test_group_maps_golden.py) keeps its tight assertions
`@pytest.mark.skip`ped pending a *measured* tolerance. Unlike Pernet's version-pinned
`SecondLevelModel`, the Jung engine is pure numpy/scipy/nibabel (no nilearn), so it is
*expected* to reproduce the published group maps to near machine precision — but the
tolerance is still measured, then frozen (docs/DESIGN.md §6: measure → freeze, don't guess).

This harness re-runs `glm_engine.run_full_analysis` for the requested clusters against the
on-disk fsaverage6 derivatives, writes the group maps to a scratch dir, and compares each
`{tstat,pval,mean}` × `{L,R}` GIFTI to the published `group_level/` map, reporting:
  - n_subjects / df (must be 78 / 77 — the published analysis, docs/DESIGN.md §7),
  - Pearson r and max|Δ| per map,
  - peak group t (LH/RH) and n_sig FDR survivors.

HEAVY: loads 78×13-run fsaverage6 BOLD per cluster → run via SLURM bigmem, NOT the login
node (per-user cgroup cap ≈ 8 GB; README §3). Reads inputs only; writes one JSON.

Usage:
  python calibrate_heavy_golden.py --family new54 --cluster-ids 5 32 49 6 30 31 \
      --derivatives-root <deriv> --bids-root <bids> \
      --scratch <tmpdir> --out calib_<envtag>.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_DATASET = Path(__file__).resolve().parent.parent
_ANALYSIS = _DATASET / "analysis"

_FAMILY_SUBDIR = {
    "new54": "cluster_contrasts_new54clusters",
}
_FAMILY_CSV = {
    "new54": "cluster_assignments_new54clusters.csv",
}


def _load(name, filename):
    sys.path.insert(0, str(_ANALYSIS))
    spec = importlib.util.spec_from_file_location(name, _ANALYSIS / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _env_info():
    import nibabel, numpy, scipy
    info = {"python": sys.version.split()[0], "numpy": numpy.__version__,
            "scipy": scipy.__version__, "nibabel": nibabel.__version__,
            "executable": sys.executable}
    try:
        import nilearn
        info["nilearn"] = nilearn.__version__
    except ImportError:
        info["nilearn"] = None
    return info


def _load_gii(path):
    import nibabel as nib
    return np.asarray(nib.load(str(path)).darrays[0].data, dtype=np.float64)


def _compare(got, ref):
    m = np.isfinite(got) & np.isfinite(ref)
    n = int(m.sum())
    r = float(np.corrcoef(got[m], ref[m])[0, 1]) if n > 1 else float("nan")
    max_abs = float(np.max(np.abs(got[m] - ref[m]))) if n else float("nan")
    return {"n_finite": n, "pearson_r": r, "max_abs_delta": max_abs}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=tuple(_FAMILY_SUBDIR), required=True)
    ap.add_argument("--cluster-ids", nargs="+", type=int, required=True)
    ap.add_argument("--derivatives-root", required=True)
    ap.add_argument("--bids-root", required=True)
    ap.add_argument("--scratch", required=True, help="Scratch output dir for regenerated group maps.")
    ap.add_argument("--out", required=True, help="JSON report path.")
    ap.add_argument("--subjects", nargs="*", default=None, help="Default: all 83 canonical (→ 78 kept).")
    args = ap.parse_args()

    engine = _load("jung_glm_engine", "glm_engine.py")

    published_dir = Path(args.derivatives_root) / _FAMILY_SUBDIR[args.family] / "group_level"
    cluster_csv = _DATASET / "data" / "cluster_assignments" / _FAMILY_CSV[args.family]
    scratch = Path(args.scratch)

    report = {"env": _env_info(), "family": args.family, "clusters": {}}

    for cid in args.cluster_ids:
        out_dir = scratch / f"{args.family}_cluster-{cid:02d}"
        ok = engine.run_full_analysis(
            target_id=cid,
            output_dir=out_dir,
            bold_dir=args.derivatives_root,
            confounds_dir=args.derivatives_root,
            bids_dir=args.bids_root,
            cluster_file=str(cluster_csv),
            subjects=args.subjects,
        )
        entry = {"success": bool(ok)}

        got_group = out_dir / "group_level"
        summary_path = got_group / "summary.json"
        if summary_path.exists():
            entry["summary"] = json.loads(summary_path.read_text())

        prefix = f"group_cluster-{cid:02d}_space-fsaverage6"
        maps = {}
        for maptype in ("tstat", "pval", "mean"):
            for hemi in ("L", "R"):
                name = f"{prefix}_hemi-{hemi}_{maptype}.func.gii"
                got_f, ref_f = got_group / name, published_dir / name
                if got_f.exists() and ref_f.exists():
                    maps[f"{maptype}_{hemi}"] = _compare(_load_gii(got_f), _load_gii(ref_f))
                else:
                    maps[f"{maptype}_{hemi}"] = {"error": "missing", "got": got_f.exists(), "ref": ref_f.exists()}
        entry["maps"] = maps
        # worst delta / worst r across all maps
        deltas = [m["max_abs_delta"] for m in maps.values() if "max_abs_delta" in m]
        rs = [m["pearson_r"] for m in maps.values() if "pearson_r" in m]
        entry["worst_max_abs_delta"] = max(deltas) if deltas else None
        entry["worst_pearson_r"] = min(rs) if rs else None
        report["clusters"][str(cid)] = entry
        print(f"[cluster {cid}] worst max|Δ|={entry['worst_max_abs_delta']} worst r={entry['worst_pearson_r']}")
        sys.stdout.flush()

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"✓ wrote {args.out}")


if __name__ == "__main__":
    main()
