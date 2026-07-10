#!/usr/bin/env python
"""Download the hosted precomputed (Tier-1) cut so a fresh clone can regenerate the figures.

docs/DESIGN.md §9 step 3 / docs/OSF_DATA.md. For each dataset it reads the in-repo manifest
(`<Dataset>/data/precomputed_manifest.json` — carries the OSF `osf_guid`, the `osf_zip` remote
path, an optional `zip_sha256`, and per-file sha256), fetches that component's SINGLE fMRI-data
zip, extracts it into `<DEST>/<Dataset>/`, and verifies every file's sha256 against the manifest.
One request per dataset instead of thousands (the zip is transport; the per-file hashes are the
integrity guarantee). Re-running re-fetches + re-verifies.

Published on OSF as the "Topo-Omni" fMRI data release, DOI 10.17605/OSF.IO/EHRT6 (docs/OSF_DATA.md).

    python download_precomputed.py --dest <DIR>                    # all three datasets
    python download_precomputed.py --dest <DIR> --datasets Jung_2025
    OSF_TOKEN=... python download_precomputed.py --dest <DIR>      # only if a component is private

Then reproduce (same layout as make_all_figures — <DEST>/<Dataset>/):
    python make_all_figures.py --derivatives-root <DIR>

Only the INPUT cut is downloaded (what `make_figures` reads). Rendered figures are regenerated,
not fetched. The OSF component GUID lives in the manifest (single source of truth); this script
does not import each dataset's config (avoids the cross-dataset `import config` collision).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make `core` importable standalone
from core import precomputed  # noqa: E402

_HERE = Path(__file__).resolve().parent
# Dataset -> in-repo manifest (also names the OSF component + files). Order follows docs/DESIGN.md §9.
DATASETS = {
    "Pernet_2015": "Pernet_2015/data/precomputed_manifest.json",
    "Marvi_2025": "Marvi_2025/data/precomputed_manifest.json",
    "Jung_2025": "Jung_2025/data/precomputed_manifest.json",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", required=True,
                   help="Base dir to download into; each dataset lands in <dest>/<Dataset>/ "
                        "(the layout make_all_figures --derivatives-root expects).")
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS),
                   help="Subset of datasets (default: all three).")
    p.add_argument("--token", default=os.environ.get("OSF_TOKEN"),
                   help="OSF personal access token (default: $OSF_TOKEN). Needed only while the "
                        "component is private; public components download without one.")
    p.add_argument("--no-verify", action="store_true", help="Skip sha256 verification (not recommended).")
    p.add_argument("--from-local", default=None,
                   help="Offline mirror: fetch the zip from <FROM_LOCAL>/<Dataset>/<osf_zip> instead "
                        "of OSF (for testing the zip extract/verify path without network).")
    p.add_argument("--keep-zip", action="store_true",
                   help="Keep the downloaded zip under <dest>/_zip_cache/ instead of deleting it.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dest_base = Path(args.dest)

    workdir = dest_base / "_zip_cache"
    failures = []
    for name in args.datasets:
        manifest = precomputed.load_manifest(_HERE / DATASETS[name])
        dest = dest_base / name
        guid = manifest.get("osf_guid")
        zip_rel = manifest.get("osf_zip")
        print(f"\n=== {name} (osf:{guid}) -> {dest} ===")
        if not zip_rel:
            print(f"    ✗ {name}: manifest has no 'osf_zip' yet (not uploaded).")
            failures.append(name)
            continue
        if args.from_local:
            fetch = precomputed.local_zip_fetcher(Path(args.from_local) / name)
        else:
            if not guid:
                print(f"    ✗ {name}: manifest has no osf_guid yet (not uploaded).")
                failures.append(name)
                continue
            fetch = precomputed.osf_zip_fetcher(guid, token=args.token)
        try:
            precomputed.download_cut_zip(manifest, dest, fetch, workdir=workdir,
                                         verify=not args.no_verify, keep_zip=args.keep_zip)
        except Exception as e:  # noqa: BLE001 — report per-dataset, keep going
            print(f"    ✗ {name} failed: {e}")
            failures.append(name)

    # tidy the transient zip cache (each dataset already deleted its zip; drop the empty dir)
    try:
        if workdir.is_dir() and not any(workdir.iterdir()):
            workdir.rmdir()
    except OSError:
        pass

    if failures:
        print("\nFAILED: " + ", ".join(failures))
        return 1
    print("\nAll requested cuts downloaded + verified. Reproduce with:\n"
          f"    python make_all_figures.py --derivatives-root {dest_base}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
