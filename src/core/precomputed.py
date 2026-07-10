"""Fetch the model-side precomputed figure cut from OSF and verify it against a manifest.

Mirrors the mechanics of the brain-side ``fMRI/core/precomputed.py`` (one zip per component,
per-file sha256 as the integrity guarantee), but the model side ships a SINGLE cut for all
figures, so there is one manifest and one zip. The manifest (``data/precomputed_manifest.json``)
carries the OSF component ``osf_guid``, the ``osf_zip`` remote filename, an optional
``zip_sha256`` transport check, and a ``files`` list of ``{path, bytes, sha256}`` where ``path``
is relative to the cut root. Downloading = fetch the zip, extract into the cut root, verify every
file's sha256. Idempotent: an already-extracted, matching cut is re-verified and not re-fetched.

Pure-stdlib except the OSF transport (``osf_zip_fetcher``, lazy ``osfclient``); ``download_cut_zip``
takes a ``zip_fetch_fn`` so it is testable with a local mirror (``local_zip_fetcher``) and offline.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import subprocess


def load_manifest(manifest_path) -> dict:
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


def verify_cut(manifest: dict, dest, verify: bool = True, log=print) -> dict:
    """Check every manifest file exists under ``dest`` with the right size (and sha256 if given).

    Raises FileNotFoundError on the first missing file and ValueError on a size/hash mismatch.
    """
    dest = Path(dest)
    files = manifest["files"]
    total = len(files)
    for i, entry in enumerate(files, 1):
        out = dest / entry["path"]
        if not out.is_file():
            raise FileNotFoundError(f"{entry['path']} missing from cut at {dest} (zip incomplete?)")
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
    log(f"[{manifest.get('name')}] {total} files present + verified -> {dest}")
    return {"name": manifest.get("name"), "n_files": total, "dest": str(dest)}


def _safe_extract(zip_path, dest: Path, log=print) -> None:
    """Extract ``zip_path`` into ``dest``, refusing any member that escapes ``dest`` (Zip-Slip)."""
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            target = (dest / name).resolve()
            if dest != target and dest not in target.parents:
                raise ValueError(f"zip member {name!r} escapes the extraction dir {dest}")
        zf.extractall(dest)
    log(f"  extracted {zip_path} -> {dest}/")


def download_cut_zip(manifest: dict, dest, zip_fetch_fn, workdir, verify: bool = True,
                     keep_zip: bool = False, log=print) -> dict:
    """Fetch the cut's single zip, extract it into ``dest``, and verify the extracted files.

    ``dest`` is the cut root; zip members are relative to it. ``zip_fetch_fn(remote_name, out_path)``
    pulls the zip (OSF or local mirror). ``manifest['osf_zip']`` is the remote filename;
    ``manifest['zip_sha256']`` (if present) is a fast transport pre-check before extraction.
    """
    dest = Path(dest)
    remote_name = manifest["osf_zip"]

    # Resumable: an already-extracted, matching cut costs only a hash pass, not a re-download.
    try:
        summary = verify_cut(manifest, dest, verify=verify, log=lambda *_: None)
        log(f"[{manifest.get('name')}] already present + verified -> {dest} (skipped download)")
        return summary
    except (FileNotFoundError, ValueError):
        pass  # missing / mismatched / partial -> (re)fetch below

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    zpath = workdir / Path(remote_name).name
    log(f"  fetching {remote_name} …")
    zip_fetch_fn(remote_name, zpath)

    want = manifest.get("zip_sha256")
    if want:
        got = sha256_file(zpath)
        if got != want:
            raise ValueError(f"zip sha256 mismatch: manifest {want} != downloaded {got} "
                             f"({zpath.stat().st_size} bytes) — transfer corrupt, re-run")
        log(f"  zip sha256 OK ({zpath.stat().st_size / 1e6:.1f} MB)")

    _safe_extract(zpath, dest, log=log)
    if not keep_zip:
        zpath.unlink(missing_ok=True)

    return verify_cut(manifest, dest, verify=verify, log=log)


def osf_zip_fetcher(osf_guid: str, token: str | None = None, storage: str = "osfstorage"):
    """Return ``fetch_fn(remote_name, out_path)`` pulling one named file from an OSF component."""
    try:
        from osfclient import OSF
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "osfclient not installed — `pip install osfclient` (set OSF_TOKEN for a private "
            "component).") from e

    osf = OSF(token=token) if token else OSF()
    store = osf.project(osf_guid).storage(storage)
    index = {f.path.lstrip("/"): f for f in store.files}  # {relpath: remote File}, built once

    def fetch(remote_name, out_path):
        remote = index.get(remote_name) or index.get("/" + remote_name)
        if remote is None:
            raise FileNotFoundError(
                f"{remote_name} not found in OSF component {osf_guid} (osfstorage). "
                f"Available: {sorted(index)[:8]}{' …' if len(index) > 8 else ''}")
        with open(out_path, "wb") as fh:
            remote.write_to(fh)

    return fetch


def local_zip_fetcher(source_root):
    """Return ``fetch_fn(remote_name, out_path)`` copying from a local mirror (tests/offline)."""
    import shutil
    source_root = Path(source_root)

    def fetch(remote_name, out_path):
        src = source_root / remote_name
        if not src.is_file():
            raise FileNotFoundError(f"{src} not present in local mirror {source_root}")
        shutil.copy2(src, out_path)

    return fetch
