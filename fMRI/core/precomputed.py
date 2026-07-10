"""Fetch a dataset's precomputed (Tier-1) cut from OSF and verify it against the manifest.

Shared helper behind `download_precomputed.py` (docs/DESIGN.md §9 step 3 / docs/OSF_DATA.md).
The manifest (shipped in-repo at `<Dataset>/data/precomputed_manifest.json`) is the source of
truth: it carries the OSF component `osf_guid` plus, per file, its `path` (relative to the
dataset's cut root) + `bytes` + `sha256`. Downloading = for each manifest file, pull it from the
component's osfstorage into `<dest>/<path>` and check sha256; **idempotent** (a file already
present with the right hash is skipped). Figures are NOT in the manifest — they are regenerated
by `make_figures`, not downloaded.

Pure-stdlib except the actual OSF transport (`osf_fetcher`, lazy `osfclient`). `download_cut`
takes a `fetch_fn(relpath, out_path)` so it is testable with a local-directory fetcher and has
no hard OSF dependency (keeps the core import-safe invariant, docs/DESIGN.md §4).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def load_manifest(manifest_path) -> dict:
    """Load a shipped precomputed_manifest.json."""
    return json.loads(Path(manifest_path).read_text())


def sha256_file(path, _bufsize=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def _ok(path: Path, entry) -> bool:
    """True if a local file already satisfies its manifest entry (size, then sha256 if known)."""
    if not path.is_file():
        return False
    if path.stat().st_size != entry["bytes"]:
        return False
    return entry.get("sha256") is None or sha256_file(path) == entry["sha256"]


def download_cut(manifest: dict, dest, fetch_fn, verify: bool = True, log=print) -> dict:
    """DEPRECATED per-file downloader — superseded by ``download_cut_zip`` (one zip per component).

    Kept only for its unit tests and as the transport-agnostic reference; ``download_precomputed.py``
    no longer calls it (per-file upload was killed by OSF/WaterButler throttling). Do NOT wire this
    up for new hosting — use ``download_cut_zip`` + ``osf_zip_fetcher``.

    Materialize a cut into ``dest`` using ``fetch_fn(relpath, out_path)``. Skips files already present
    with the right size+hash (idempotent). Verifies sha256 after each fetch (unless the manifest entry
    has none — then size-only). Raises on a hash mismatch or a missing remote file (via fetch_fn).
    """
    dest = Path(dest)
    files = manifest["files"]
    downloaded = skipped = 0
    total = len(files)
    for i, entry in enumerate(files, 1):
        rel = entry["path"]
        out = dest / rel
        if _ok(out, entry):
            skipped += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        fetch_fn(rel, out)
        if verify and entry.get("sha256") is not None:
            got = sha256_file(out)
            if got != entry["sha256"]:
                raise ValueError(
                    f"sha256 mismatch for {rel}: manifest {entry['sha256']} != downloaded {got}")
        elif out.stat().st_size != entry["bytes"]:
            raise ValueError(f"size mismatch for {rel}: expected {entry['bytes']}, got {out.stat().st_size}")
        downloaded += 1
        if downloaded % 100 == 0 or i == total:
            log(f"  … {i}/{total} ({downloaded} fetched, {skipped} already present)")
    summary = {"dataset": manifest.get("dataset"), "n_files": total,
               "downloaded": downloaded, "skipped": skipped, "dest": str(dest)}
    log(f"[{summary['dataset']}] {downloaded} fetched + {skipped} present = {total} files -> {dest}")
    return summary


def verify_cut(manifest: dict, dest, verify: bool = True, log=print) -> dict:
    """Verify every manifest file already exists under ``dest`` with the right size (+sha256).

    Used after extracting the component's single zip: the archive IS the transport, so we don't
    fetch per file — we just check the extracted tree against the per-file hashes (unchanged
    integrity guarantee). Raises on the first missing file or hash/size mismatch.
    """
    dest = Path(dest)
    files = manifest["files"]
    total = len(files)
    for i, entry in enumerate(files, 1):
        out = dest / entry["path"]
        if not out.is_file():
            raise FileNotFoundError(
                f"{entry['path']} missing from extracted cut at {dest} (zip incomplete?)")
        if verify and entry.get("sha256") is not None:
            got = sha256_file(out)
            if got != entry["sha256"]:
                raise ValueError(
                    f"sha256 mismatch for {entry['path']}: manifest {entry['sha256']} != extracted {got}")
        elif out.stat().st_size != entry["bytes"]:
            raise ValueError(
                f"size mismatch for {entry['path']}: expected {entry['bytes']}, got {out.stat().st_size}")
        if i % 200 == 0 or i == total:
            log(f"  … verified {i}/{total}")
    summary = {"dataset": manifest.get("dataset"), "n_files": total, "dest": str(dest)}
    log(f"[{summary['dataset']}] {total} files present + verified -> {dest}")
    return summary


def _safe_extract(zip_path, base: Path, expected_top: str, log=print) -> None:
    """Extract ``zip_path`` into ``base``, refusing any member that escapes ``base`` or whose top
    path component isn't ``expected_top`` (the wrapper folder == dataset dir). Guards path
    traversal (Zip-Slip) and a mismatched/tampered archive before writing anything."""
    import zipfile
    base = Path(base).resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            top = name.split("/", 1)[0]
            if top != expected_top:
                raise ValueError(
                    f"zip member {name!r} is outside the expected '{expected_top}/' root")
            target = (base / name).resolve()
            if base not in target.parents and target != base:
                raise ValueError(f"zip member {name!r} escapes the extraction dir")
        zf.extractall(base)
    log(f"  extracted {zip_path} -> {base}/{expected_top}/")


def download_cut_zip(manifest: dict, dest, zip_fetch_fn, workdir, verify: bool = True,
                     keep_zip: bool = False, log=print) -> dict:
    """Fetch the component's single zip, extract it, and verify the extracted cut.

    ``dest`` is ``<base>/<Dataset>``; the zip's members are ``<Dataset>/<relpath>`` so extracting
    into ``dest.parent`` lands the cut exactly where ``make_all_figures`` and ``verify_cut``
    expect. ``zip_fetch_fn(remote_relpath, out_path)`` pulls the zip (OSF or a local mirror).
    ``manifest['osf_zip']`` is the remote relpath (e.g. ``fMRI_data/<Dataset>_fMRI_data.zip``);
    ``manifest['zip_sha256']`` (if present) is a fast transport pre-check before extraction.
    """
    dest = Path(dest)
    base = dest.parent
    remote_rel = manifest["osf_zip"]

    # Resumable: if the cut is already extracted and matches the manifest, skip the whole
    # fetch+extract (a completed re-run costs only a hash pass, not a multi-GB re-download).
    try:
        summary = verify_cut(manifest, dest, verify=verify, log=lambda *_: None)
        log(f"[{manifest.get('dataset')}] already present + verified -> {dest} (skipped download)")
        return summary
    except (FileNotFoundError, ValueError):
        pass  # missing / mismatched / partial -> (re)fetch the zip below

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    zpath = workdir / Path(remote_rel).name
    log(f"  fetching {remote_rel} …")
    zip_fetch_fn(remote_rel, zpath)

    want = manifest.get("zip_sha256")
    if want:
        got = sha256_file(zpath)
        if got != want:
            raise ValueError(f"zip sha256 mismatch: manifest {want} != downloaded {got} "
                             f"({zpath.stat().st_size} bytes) — transfer corrupt, re-run")
        log(f"  zip sha256 OK ({zpath.stat().st_size/1e6:.1f} MB)")

    _safe_extract(zpath, base, dest.name, log=log)
    if not keep_zip:
        zpath.unlink(missing_ok=True)
    return verify_cut(manifest, dest, verify=verify, log=log)


def osf_zip_fetcher(osf_guid: str, token: str | None = None, storage: str = "osfstorage"):
    """Return ``fetch_fn(remote_relpath, out_path)`` pulling ONE named file from a component.

    Unlike ``osf_fetcher`` (per-file, thousands of requests), the component now holds a handful of
    artifacts, so building the index is cheap. ``token`` needed only while the component is private.
    """
    try:
        from osfclient import OSF
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "osfclient not installed — `pip install osfclient` (and set OSF_TOKEN for a "
            "private component). See docs/OSF_DATA.md.") from e

    osf = OSF(token=token) if token else OSF()
    store = osf.project(osf_guid).storage(storage)
    index = {f.path.lstrip("/"): f for f in store.files}  # {relpath: remote File}, built once

    def fetch(remote_rel, out_path):
        remote = index.get(remote_rel)
        if remote is None:
            raise FileNotFoundError(
                f"{remote_rel} not found in OSF component {osf_guid} (osfstorage). "
                f"Available: {sorted(index)[:8]}{' …' if len(index) > 8 else ''}")
        with open(out_path, "wb") as fh:
            remote.write_to(fh)

    return fetch


def local_zip_fetcher(source_root):
    """Return a ``fetch_fn(remote_relpath, out_path)`` copying from a local mirror (tests/offline)."""
    import shutil
    source_root = Path(source_root)

    def fetch(remote_rel, out_path):
        src = source_root / remote_rel
        if not src.is_file():
            raise FileNotFoundError(f"{src} not present in local mirror {source_root}")
        shutil.copy2(src, out_path)

    return fetch


def osf_fetcher(osf_guid: str, token: str | None = None, storage: str = "osfstorage"):
    """DEPRECATED per-file OSF fetcher — superseded by ``osf_zip_fetcher`` (see ``download_cut``).

    Return a ``fetch_fn(relpath, out_path)`` that pulls from an OSF component's osfstorage.
    Requires `osfclient` (lazy). ``token`` (OSF PAT) is only needed while the component is
    private; public components download without one. Remote paths are assumed to mirror the
    manifest relpaths. The remote file index is built once, on first use.
    """
    try:
        from osfclient import OSF
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "osfclient not installed — `pip install osfclient` (and set OSF_TOKEN for a "
            "private component). See docs/OSF_DATA.md.") from e

    osf = OSF(token=token) if token else OSF()
    project = osf.project(osf_guid)
    store = project.storage(storage)
    index = {f.path.lstrip("/"): f for f in store.files}  # {relpath: remote File}, built once

    def fetch(relpath, out_path):
        remote = index.get(relpath)
        if remote is None:
            raise FileNotFoundError(
                f"{relpath} not found in OSF component {osf_guid} (osfstorage). "
                f"Upload incomplete, or the remote layout differs from the manifest.")
        with open(out_path, "wb") as fh:
            remote.write_to(fh)

    return fetch


def local_dir_fetcher(source_root):
    """DEPRECATED per-file mirror fetcher (pairs with ``download_cut``) — the zip path uses
    ``local_zip_fetcher``. Return a ``fetch_fn`` that copies from a local directory."""
    import shutil
    source_root = Path(source_root)

    def fetch(relpath, out_path):
        src = source_root / relpath
        if not src.is_file():
            raise FileNotFoundError(f"{src} not present in local mirror {source_root}")
        shutil.copy2(src, out_path)

    return fetch
