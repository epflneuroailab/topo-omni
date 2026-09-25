"""new27 family — the rebuttal rerun on the new-model cluster partition (Sep 2026).

Badr AlKhamissi re-ran the model-side SpaceTop discovery (spacetop_discovery step 4,
`clusters_tvals.json`, raw dendrogram leaves, no hand merge) with the new Topo-Omni
checkpoint and asked for the same fMRI cluster-vs-others validation that produced the
published new54 maps. These tests pin:

  * the family registration (config + the contrast driver agree),
  * the vendored JSON bytes (sha256 = the file Badr sent),
  * the partition's shape (27 clusters, contiguous 0-based IDs, disjoint clips),
  * that the published new54 family and figures are untouched.

Pure-Python except where noted; no fMRI data required.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_DATASET = Path(__file__).resolve().parent.parent


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, _DATASET / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


config = _load("jung_config_new27", "config.py")


def _family():
    return config.CLUSTERS["new27"]


def _json_path():
    return _DATASET / _family()["model_json"]


def _clusters():
    return json.loads(_json_path().read_text())


# --- registration -----------------------------------------------------------------

def test_new27_family_is_registered_next_to_new54():
    assert list(config.CLUSTERS)[:2] == ["new54", "new27"]
    fam = _family()
    assert fam["model_json"] == "data/cluster_assignments/27_cluster.json"
    assert fam["assignments_csv"] == "data/cluster_assignments/cluster_assignments_new27clusters.csv"
    assert fam["derivatives_subdir"] == "cluster_contrasts_new27clusters"
    assert fam["n_clusters"] == 27


def test_new27_is_not_a_paper_figure():
    # The published figures still draw only from new54; new27 is a rebuttal analysis.
    assert {k: v["ids"] for k, v in config.FIGURES.items()} == {
        "fig6_d4": [5, 32, 49], "figD5": [6, 30, 31]}
    assert config.CLUSTERS["new54"]["model_json"].endswith("54_cluster.json")


def test_contrast_driver_families_match_config():
    sys.path.insert(0, str(_DATASET / "analysis"))
    contrasts = _load("jung_cluster_contrasts_new27", "analysis/cluster_contrasts.py")
    assert set(contrasts.FAMILIES) == set(config.CLUSTERS)
    for key, fam in config.CLUSTERS.items():
        assert contrasts.FAMILIES[key]["derivatives_subdir"] == fam["derivatives_subdir"]


# --- the vendored model JSON ------------------------------------------------------

def test_vendored_json_is_the_file_badr_sent():
    got = hashlib.sha256(_json_path().read_bytes()).hexdigest()
    assert got == _family()["model_json_sha256"]
    assert got == "5f2cd1b5959ee27c3f0bfb2b4ce2fa30d18affa6471d8115f9a49a1577437ead"


def test_partition_has_contiguous_zero_based_ids():
    ids = [c["cluster_id"] for c in _clusters()]
    assert ids == list(range(_family()["n_clusters"]))


def test_partition_sizes_match_clip_lists():
    for c in _clusters():
        assert c["size"] == len(c["video_ids"]) > 0, c["cluster_id"]


def test_partition_clips_are_disjoint():
    seen = {}
    for c in _clusters():
        for v in c["video_ids"]:
            assert v not in seen, f"{v} in clusters {seen[v]} and {c['cluster_id']}"
            seen[v] = c["cluster_id"]
    assert len(seen) == 2573


def test_every_clip_name_parses():
    parser = _load("jung_parse_clusters_new27", "analysis/parse_clusters.py")
    for c in _clusters():
        for v in c["video_ids"]:
            session, run, video, chunk = parser.parse_video_id(v)
            assert session is not None, v
