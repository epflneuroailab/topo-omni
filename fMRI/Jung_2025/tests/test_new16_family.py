"""new16 family — the rebuttal partition cut coarser (Sep 2026, min cluster size 60).

Badr AlKhamissi sent a second partition from the same new-checkpoint discovery run as new27,
cut higher in the dendrogram (`clusters_tvals_disjoint_min60.json`, vendored as
`16_cluster.json`). These tests pin:

  * the family registration (config + the contrast driver agree),
  * the vendored JSON bytes (sha256 = the file Badr sent),
  * the partition's shape (16 clusters, contiguous 0-based IDs, disjoint clips, sizes >= 60),
  * that it is an exact coarsening of new27 (config.NEW16_FROM_NEW27), which is what makes
    the new16-vs-new27 map comparison (analysis/compare_families.py) interpretable.

Pure-Python; no fMRI data required.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

_DATASET = Path(__file__).resolve().parent.parent


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, _DATASET / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


config = _load("jung_config_new16", "config.py")


def _family():
    return config.CLUSTERS["new16"]


def _clusters(family="new16"):
    return json.loads((_DATASET / config.CLUSTERS[family]["model_json"]).read_text())


def _clip_sets(family):
    return {c["cluster_id"]: set(c["video_ids"]) for c in _clusters(family)}


# --- registration -----------------------------------------------------------------

def test_new16_family_is_registered_after_new27():
    assert list(config.CLUSTERS) == ["new54", "new27", "new16"]
    fam = _family()
    assert fam["model_json"] == "data/cluster_assignments/16_cluster.json"
    assert fam["assignments_csv"] == "data/cluster_assignments/cluster_assignments_new16clusters.csv"
    assert fam["derivatives_subdir"] == "cluster_contrasts_new16clusters"
    assert fam["n_clusters"] == 16


def test_contrast_driver_knows_new16():
    sys.path.insert(0, str(_DATASET / "analysis"))
    contrasts = _load("jung_cluster_contrasts_new16", "analysis/cluster_contrasts.py")
    assert contrasts.FAMILIES["new16"]["derivatives_subdir"] == _family()["derivatives_subdir"]


# --- the vendored model JSON ------------------------------------------------------

def test_vendored_json_is_the_file_badr_sent():
    got = hashlib.sha256((_DATASET / _family()["model_json"]).read_bytes()).hexdigest()
    assert got == _family()["model_json_sha256"]
    assert got == "d1b5bd5489f2396469d4c9020ff9fce3c3db5564a0e58eef066137e4f8427078"


def test_partition_shape():
    clusters = _clusters()
    assert [c["cluster_id"] for c in clusters] == list(range(_family()["n_clusters"]))
    for c in clusters:
        assert c["size"] == len(c["video_ids"]) >= 60, c["cluster_id"]
    clips = [v for c in clusters for v in c["video_ids"]]
    assert len(clips) == len(set(clips)) == 2573


def test_same_clips_as_new27():
    new16, new27 = _clip_sets("new16"), _clip_sets("new27")
    assert set().union(*new16.values()) == set().union(*new27.values())


def test_is_an_exact_coarsening_of_new27():
    new16, new27 = _clip_sets("new16"), _clip_sets("new27")
    assert sorted(config.NEW16_FROM_NEW27) == sorted(new16)
    assert sorted(i for parts in config.NEW16_FROM_NEW27.values() for i in parts) == sorted(new27)
    for k, parts in config.NEW16_FROM_NEW27.items():
        assert new16[k] == set().union(*(new27[j] for j in parts)), k


def test_every_clip_name_parses():
    parser = _load("jung_parse_clusters_new16", "analysis/parse_clusters.py")
    for c in _clusters():
        for v in c["video_ids"]:
            session, run, video, chunk = parser.parse_video_id(v)
            assert session is not None, v
