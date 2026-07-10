#!/usr/bin/env python
"""Assemble the precomputed figure cut, zip it, and write data/precomputed_manifest.json.

This is the single source of truth for WHAT the hosted cut contains: the minimal set of
per-figure plot inputs (selectivity stats, response-profile pickles, ablation/Moran JSONs,
per-cluster t-maps, discovery scores) — deliberately excluding the large intermediates
(feature caches, retinotopy/tonotopy cortical_sheets, NSD HDF5) which are recomputed via the
raw path instead. Authors run this once against the research results tree, then upload the zip
to OSF and drop the returned osf_guid into the manifest.

    python scripts/build_precomputed_cut.py \
        --source-root /path/to/results \
        --discovery-scores /path/to/cluster_selectivity_scores_v1.json \
        --dest ~/topo-omni-cut --zip-out ~/topo_omni_figures_cut.zip

The manifest's ``osf_guid`` stays null until you upload; ``download_precomputed.py --from-local``
works against ``--dest`` for local verification in the meantime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

MODEL_TITLE = "topo-omni"
CATEGORIES = ["faces", "bodies", "scenes", "objects", "vwfa", "speech", "vocals",
              "language_text", "theory_of_mind_text", "multiple_demand_text"]
VISUAL_RP = ["faces", "scenes", "objects", "vwfa", "speech"]
COGNITIVE_RP = ["language_text", "multiple_demand_text", "theory_of_mind_text"]
STIM_PCTS = [5, 10, 15, 20, 25, 30]
CLUSTERS = ["05", "32"]
MORANS_JSON = ("island_morans_I_results_rating_contrast_vs_ratingv0_"
               "significant_topk=1_audio=True_smooth=True_fwhm=8.json")


def cut_entries():
    """Yield cut-relative paths, each relative to <source-root> (same layout) unless noted."""
    mt = MODEL_TITLE
    # Fig 2b/3/4/5 — unified selectivity map inputs (per-category stats)
    for c in CATEGORIES:
        yield f"{mt}/{c}/{c}_all_selectivity_stats.pkl"
    # Fig 3a-d / 4a — visual + speech response profiles (top-1, even/odd split-half)
    for c in VISUAL_RP:
        for oe in ("even", "odd"):
            yield f"{mt}/response_profiles/{c}_response_profiles_top1_{oe}.pkl"
    # Fig 4b — voice / PerNet folds
    for fold in ("A", "B"):
        yield f"{mt}/response_profiles/pernet_fold_{fold}_response_profiles_top1_None_fwhm_mm=4.0_anat=True.pkl"
    # Fig 5 — cognitive response profiles (top-10)
    for c in COGNITIVE_RP:
        for oe in ("even", "odd"):
            yield f"{mt}/response_profiles/{c}_response_profiles_top10_{oe}.pkl"
    # Fig 6a — driving perception (stimulation)
    for p in STIM_PCTS:
        yield f"{mt}/ablation/similarity_ablation_results_top{p}_stimulate=True_v3.json"
    # Fig 6b-c — suppression accuracy
    yield f"{mt}/ablation/similarity_ablation_results_top10_stimulate=False_v4.json"
    yield f"{mt}/ablation/similarity_no_ablation_results_v4.json"
    # Fig 2c / 7 — discovered-cluster t-maps + model-vs-fMRI Moran's I
    for cid in CLUSTERS:
        yield f"{mt}/spacetop_clusters_figures/{cid}/cluster_{cid}_t_map.npy"
    yield f"{mt}/spacetop_clusters_figures/{MORANS_JSON}"


def sha256_file(path, _b=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_b), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-root", required=True, help="Results dir containing <MODEL_TITLE>/...")
    ap.add_argument("--discovery-scores", default=None,
                    help="Path to cluster_selectivity_scores_v1.json (Fig 2c discovery strip plot).")
    ap.add_argument("--dest", required=True, help="Staging dir (== the extracted cut root).")
    ap.add_argument("--zip-out", required=True, help="Output zip path.")
    ap.add_argument("--manifest", default=str(Path(__file__).resolve().parents[1] / "data" / "precomputed_manifest.json"))
    ap.add_argument("--zip-name", default="topo_omni_figures_cut.zip", help="osf_zip name recorded in the manifest.")
    args = ap.parse_args(argv)

    src = Path(args.source_root)
    dest = Path(args.dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    # (cut_relpath, source_abspath)
    plan = [(rel, src / rel) for rel in cut_entries()]
    if args.discovery_scores:
        plan.append(("spacetop_discovery/cluster_selectivity_scores_v1.json", Path(args.discovery_scores)))

    missing = [str(s) for _, s in plan if not s.is_file()]
    if missing:
        raise SystemExit("Missing source files:\n  " + "\n  ".join(missing))

    files_meta = []
    for rel, s in plan:
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, out)
        files_meta.append({"path": rel, "bytes": out.stat().st_size, "sha256": sha256_file(out)})
    print(f"copied {len(files_meta)} files -> {dest}")

    # zip the cut (members relative to dest, matching manifest paths)
    zip_out = Path(args.zip_out)
    with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel, _ in plan:
            zf.write(dest / rel, rel)
    zip_sha = sha256_file(zip_out)
    print(f"wrote zip {zip_out} ({zip_out.stat().st_size / 1e6:.1f} MB)")

    manifest = {
        "name": "topo-omni-figures",
        "description": "Precomputed model-side figure inputs for AlKhamissi & Mehrer et al., 2026.",
        "osf_guid": None,          # <-- fill in after uploading the zip to OSF
        "osf_zip": args.zip_name,
        "zip_sha256": zip_sha,
        "files": files_meta,
    }
    Path(args.manifest).write_text(json.dumps(manifest, indent=2))
    print(f"wrote manifest {args.manifest} ({len(files_meta)} files, "
          f"total {sum(f['bytes'] for f in files_meta) / 1e6:.1f} MB)")
    print("\nNext: upload the zip to OSF, set 'osf_guid' in the manifest, and commit it.")


if __name__ == "__main__":
    main()
