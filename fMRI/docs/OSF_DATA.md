# OSF hosting & retrieval

The precomputed data tier for the three fMRI datasets is hosted on OSF and retrieved by the
shipped `download_precomputed.py`. This is the reference for how that data is laid out and how
retrieval verifies it.

**This document covers:** the OSF **hosting layout** (the umbrella project, the per-dataset
components, and the three artifacts each one ships); the per-file sha256 **integrity model**;
**retrieval** with `download_precomputed.py` (flags, the optional token for private components); and how to
**verify from a clean clone**.

## Hosting layout

An umbrella OSF project, **Topo-Omni** (`ehrt6`, DOI
[`10.17605/OSF.IO/EHRT6`](https://doi.org/10.17605/OSF.IO/EHRT6)), holds one component per dataset:

| Dataset | Component GUID |
|---|---|
| Pernet_2015 | `6uwzr` |
| Marvi_2025 | `tb9cr` |
| Jung_2025 | `dpeys` |

Each component holds, under an **`fMRI_data/`** folder, three artifacts named
`<Dataset>_fMRI_data.*`:

| Artifact | What |
|---|---|
| `<Dataset>_fMRI_data.zip` | the precomputed cut; zip members are `<Dataset>/<relpath>`, so unzipping yields the reproduction layout directly |
| `<Dataset>_fMRI_data_contents.txt` | a human-readable file tree (paths + sizes), previewable on OSF before downloading multi-GB |
| `<Dataset>_fMRI_data_manifest.json` | the per-file sha256 manifest (integrity source of truth) |

Each zip is under OSF's 5 GB single-file cap. The `fMRI_data/` naming keeps the fMRI data
unmistakable and leaves parallel folders (e.g. `model/`) free for other cuts.

## Integrity model

Every shipped file carries a sha256 in the manifest, and that per-file hash is the real
guarantee. `download_precomputed.py` checks the zip's `zip_sha256` on fetch, extracts it
(Zip-Slip-guarded), then verifies every extracted file's sha256 against the manifest.

`zip_sha256` is only a transport check. It is **not** guaranteed reproducible across build
hosts — zips embed file mtimes and zlib output can vary by version/host, so identical content
can yield a different `zip_sha256`. The per-file hashes, not the zip hash, are what protect the
reviewer.

The manifest is shipped in-repo at `<Dataset>/data/precomputed_manifest.json`. Path + sha256 is
the single source of truth; no OSF file-id is needed, because the downloader matches each
manifest path to its extracted file and verifies the hash.

## Retrieval

```bash
python download_precomputed.py --dest .
```

This pulls each component's zip into `<dest>/<Dataset>/`, extracts, and verifies. Key flags:

- `--dest` — destination root (each dataset lands under `<dest>/<Dataset>/`).
- `--datasets` — restrict to a subset of datasets.
- `--token` / `$OSF_TOKEN` — OSF read token (see below).
- `--no-verify` — skip sha256 verification.
- `--keep-zip` — retain the downloaded zip after extraction.
- `--from-local <mirror>` — extract from a local mirror instead of fetching from OSF.

**The OSF components are public — no token is required** to download. The `--token` /
`$OSF_TOKEN` flag remains for anyone who hosts a private mirror (a read-scope token from
https://osf.io/settings/tokens is enough); against the public components it is simply ignored.

The retrieval engine lives in `core/precomputed.py`.

Licensing (set in each component's OSF metadata): **CC-BY 4.0** for Pernet_2015; **CC0** for
Marvi_2025 and Jung_2025.

## Verify from a clean clone

From a fresh clone, per dataset:

```bash
python download_precomputed.py --dest <dest>
python <Dataset>/make_figures.py --input-source precomputed --derivatives-root <dest>
```

The figures regenerate from the downloaded cut, and the data-gated golden tests then run
instead of skipping. Pernet_2015 uses `--results-root` in place of `--derivatives-root`.
