"""Unit tests for core.precomputed — the manifest-driven cut downloader (download/verify/idempotent).

Uses a synthetic manifest + a local-directory fetcher, so no OSF/network. The real OSF transport
(`osf_fetcher`) is a thin adapter over the same `download_cut` contract exercised here.
"""
import hashlib
import json

import pytest

from core import precomputed


def _make_source(root, files):
    """Write {relpath: bytes} under root; return a manifest dict with real sha256/bytes."""
    entries = []
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        entries.append({"path": rel, "group": "g", "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest()})
    return {"dataset": "Test", "osf_guid": "xxxxx", "n_files": len(entries),
            "total_bytes": sum(len(d) for d in files.values()), "files": entries}


def test_download_verifies_and_is_idempotent(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    manifest = _make_source(src, {
        "a.nii.gz": b"alpha-data",
        "sub/b.func.gii": b"beta-data-longer",
        "sub/c.tsv": b"c",
    })
    fetch = precomputed.local_dir_fetcher(src)

    s1 = precomputed.download_cut(manifest, dest, fetch, log=lambda *_: None)
    assert s1 == {"dataset": "Test", "n_files": 3, "downloaded": 3, "skipped": 0, "dest": str(dest)}
    for e in manifest["files"]:
        assert (dest / e["path"]).read_bytes() == (src / e["path"]).read_bytes()

    # second run: everything present + hash-matching -> all skipped (resumable/idempotent)
    s2 = precomputed.download_cut(manifest, dest, fetch, log=lambda *_: None)
    assert s2["downloaded"] == 0 and s2["skipped"] == 3


def test_corrupted_local_file_is_refetched(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    manifest = _make_source(src, {"a.bin": b"good-bytes"})
    fetch = precomputed.local_dir_fetcher(src)
    precomputed.download_cut(manifest, dest, fetch, log=lambda *_: None)

    (dest / "a.bin").write_bytes(b"CORRUPTED")  # wrong size+hash -> not _ok -> re-fetched
    s = precomputed.download_cut(manifest, dest, fetch, log=lambda *_: None)
    assert s["downloaded"] == 1 and (dest / "a.bin").read_bytes() == b"good-bytes"


def test_sha256_mismatch_raises(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    manifest = _make_source(src, {"a.bin": b"real-bytes"})
    manifest["files"][0]["sha256"] = "0" * 64  # manifest claims a different hash than the source
    fetch = precomputed.local_dir_fetcher(src)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        precomputed.download_cut(manifest, dest, fetch, log=lambda *_: None)


def test_missing_remote_file_raises(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    manifest = _make_source(src, {"a.bin": b"x"})
    manifest["files"].append({"path": "ghost.bin", "group": "g", "bytes": 3, "sha256": "a" * 64})
    fetch = precomputed.local_dir_fetcher(src)
    with pytest.raises(FileNotFoundError):
        precomputed.download_cut(manifest, dest, fetch, log=lambda *_: None)


def _make_zip(zip_path, member_bytes):
    """Write a zip of {arcname: bytes} (arcnames already include the <Dataset>/ wrapper)."""
    import zipfile
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for arc, data in member_bytes.items():
            zf.writestr(arc, data)


def test_zip_roundtrip_extract_and_verify(tmp_path):
    import hashlib
    files = {"a.nii.gz": b"alpha", "sub/b.func.gii": b"beta-longer"}
    manifest = {"dataset": "Test", "osf_zip": "fMRI_data/Test_fMRI_data.zip",
                "files": [{"path": r, "bytes": len(d), "sha256": hashlib.sha256(d).hexdigest()}
                          for r, d in files.items()]}
    zsrc = tmp_path / "mirror" / "Test" / "fMRI_data"
    zsrc.mkdir(parents=True)
    zpath = zsrc / "Test_fMRI_data.zip"
    _make_zip(zpath, {f"Test/{r}": d for r, d in files.items()})  # wrapper == dataset dir
    manifest["zip_sha256"] = precomputed.sha256_file(zpath)

    dest = tmp_path / "out" / "Test"
    fetch = precomputed.local_zip_fetcher(tmp_path / "mirror" / "Test")
    precomputed.download_cut_zip(manifest, dest, fetch, workdir=tmp_path / "wd", log=lambda *_: None)
    for r, d in files.items():
        assert (dest / r).read_bytes() == d


def test_zip_sha256_mismatch_raises(tmp_path):
    import hashlib
    d = b"payload"
    manifest = {"dataset": "T", "osf_zip": "fMRI_data/T.zip", "zip_sha256": "0" * 64,
                "files": [{"path": "a.bin", "bytes": len(d), "sha256": hashlib.sha256(d).hexdigest()}]}
    zsrc = tmp_path / "m" / "T" / "fMRI_data"
    zsrc.mkdir(parents=True)
    _make_zip(zsrc / "T.zip", {"T/a.bin": d})
    fetch = precomputed.local_zip_fetcher(tmp_path / "m" / "T")
    with pytest.raises(ValueError, match="zip sha256 mismatch"):
        precomputed.download_cut_zip(manifest, tmp_path / "out" / "T", fetch,
                                     workdir=tmp_path / "wd", log=lambda *_: None)


def test_zip_slip_member_is_rejected(tmp_path):
    """A zip member escaping the extraction root (../evil) must be refused before writing."""
    # non-empty manifest so the resumable verify-first check fails (file absent) and we reach extract
    manifest = {"dataset": "T", "osf_zip": "fMRI_data/T.zip",
                "files": [{"path": "a.bin", "bytes": 1, "sha256": "a" * 64}]}
    zsrc = tmp_path / "m" / "T" / "fMRI_data"
    zsrc.mkdir(parents=True)
    _make_zip(zsrc / "T.zip", {"../evil.txt": b"pwned"})
    fetch = precomputed.local_zip_fetcher(tmp_path / "m" / "T")
    with pytest.raises(ValueError, match="outside the expected|escapes"):
        precomputed.download_cut_zip(manifest, tmp_path / "out" / "T", fetch,
                                     workdir=tmp_path / "wd", log=lambda *_: None)


def test_zip_download_is_resumable(tmp_path):
    """A second download with the cut already extracted skips fetch+extract (verify-first)."""
    import hashlib
    d = b"already-here"
    manifest = {"dataset": "T", "osf_zip": "fMRI_data/T.zip",
                "files": [{"path": "a.bin", "bytes": len(d), "sha256": hashlib.sha256(d).hexdigest()}]}
    zsrc = tmp_path / "m" / "T" / "fMRI_data"
    zsrc.mkdir(parents=True)
    _make_zip(zsrc / "T.zip", {"T/a.bin": d})
    manifest["zip_sha256"] = precomputed.sha256_file(zsrc / "T.zip")
    dest = tmp_path / "out" / "T"
    fetch = precomputed.local_zip_fetcher(tmp_path / "m" / "T")
    precomputed.download_cut_zip(manifest, dest, fetch, workdir=tmp_path / "wd", log=lambda *_: None)

    # remove the source zip so any re-fetch would fail; a resumable re-run must still succeed
    (zsrc / "T.zip").unlink()
    precomputed.download_cut_zip(manifest, dest, fetch, workdir=tmp_path / "wd", log=lambda *_: None)
    assert (dest / "a.bin").read_bytes() == d


def test_shipped_manifests_are_wired(tmp_path):
    """Each dataset's in-repo manifest exists, is non-empty, and carries its OSF GUID."""
    import pathlib
    repo = pathlib.Path(__file__).resolve().parents[2]
    expect = {"Pernet_2015": "6uwzr", "Marvi_2025": "tb9cr", "Jung_2025": "dpeys"}
    for ds, guid in expect.items():
        m = json.loads((repo / ds / "data" / "precomputed_manifest.json").read_text())
        assert m["osf_guid"] == guid
        assert m["n_files"] == len(m["files"]) > 0
        assert all(e.get("sha256") and e["path"] for e in m["files"])
        # zip hosting fields (recorded in the manifest when the cut is packaged for OSF)
        assert m["osf_zip"] == f"fMRI_data/{ds}_fMRI_data.zip"
        assert len(m["zip_sha256"]) == 64 and m["zip_bytes"] > 0
