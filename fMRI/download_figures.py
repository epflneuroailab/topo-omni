#!/usr/bin/env python
"""Download ONLY the rendered figures for each dataset from OSF, for review in any location.

Separate from download_precomputed.py (which fetches the input cut + verifies per-file sha256).
For each dataset this pulls `figures/<Dataset>_fMRI_figures.zip` from that OSF component and
extracts it into `<dest>/<Dataset>_fMRI_figures/`. Figures are regenerable outputs (no hashed
manifest); the zip is the artifact. A `<Dataset>_fMRI_figures_contents.txt` on OSF lists the tree.

    export OSF_TOKEN=...                                   # optional — only if a component is private
    python download_figures.py --dest /some/review/dir              # all three datasets
    python download_figures.py --dest /some/review/dir --datasets Marvi_2025
    python download_figures.py --dest /some/review/dir --keep-zip    # keep the .zip too
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import precomputed  # osf_zip_fetcher + _safe_extract  # noqa: E402

# Dataset -> OSF component GUID (umbrella ehrt6). Matches the shipped per-dataset manifests.
COMPONENTS = {"Pernet_2015": "6uwzr", "Marvi_2025": "tb9cr", "Jung_2025": "dpeys"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", required=True, help="Dir to extract figures into (<dest>/<Dataset>_fMRI_figures/).")
    p.add_argument("--datasets", nargs="+", choices=list(COMPONENTS), default=list(COMPONENTS))
    p.add_argument("--token", default=os.environ.get("OSF_TOKEN"),
                   help="OSF personal access token (default: $OSF_TOKEN). Only if a component is "
                        "private; the public components download without one.")
    p.add_argument("--keep-zip", action="store_true", help="Keep the downloaded .zip under <dest>/_zip_cache/.")
    args = p.parse_args(argv)

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    workdir = dest / "_zip_cache"
    workdir.mkdir(exist_ok=True)

    rc = 0
    for name in args.datasets:
        guid = COMPONENTS[name]
        remote = f"figures/{name}_fMRI_figures.zip"
        top = f"{name}_fMRI_figures"
        print(f"\n=== {name} figures (osf:{guid}) ===")
        try:
            fetch = precomputed.osf_zip_fetcher(guid, token=args.token)
            zpath = workdir / f"{top}.zip"
            print(f"  fetching {remote} …")
            fetch(remote, zpath)
            precomputed._safe_extract(zpath, dest, top)
            if not args.keep_zip:
                zpath.unlink(missing_ok=True)
            n = sum(1 for f in (dest / top).rglob("*") if f.is_file())
            print(f"  -> {dest / top}/  ({n} figure files)")
        except Exception as e:  # noqa: BLE001 — report per-dataset, keep going
            print(f"  ✗ {name} failed: {e}")
            rc = 1

    try:  # drop the empty cache dir
        if workdir.is_dir() and not any(workdir.iterdir()):
            workdir.rmdir()
    except OSError:
        pass

    if rc == 0:
        print(f"\nAll requested figures extracted under {dest}/")
    return rc


if __name__ == "__main__":
    sys.exit(main())
