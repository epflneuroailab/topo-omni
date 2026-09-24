"""Golden master — cluster-assignment CSV parse (docs/DESIGN.md §6 Tier 1, new54 family).

Re-parses the vendored new54 model JSON (data/cluster_assignments/54_cluster.json) through
the ported `parse_clusters` and asserts the result reproduces the vendored on-disk CSV
**bitwise**. This pins the model-input → design-matrix seam: the CSV drives every
downstream regressor. (The 14-cluster branch is not in the paper — README §4.)

The parse is pure pandas + regex (no nilearn) and reproduces the dev CSV exactly
(verified 2026-07-07: `diff` empty). Frozen at bitwise equality.

DATA-GATED (needs sub-0001 `task-alignvideo` events under the raw BIDS root for
run-relative video onsets). Skips cleanly when the raw dataset is absent. Point with
JUNG_RAW_ROOT.
"""
import importlib.util
import os
from pathlib import Path

import pytest

_DATASET = Path(__file__).resolve().parent.parent
_DATA = _DATASET / "data" / "cluster_assignments"
_CANDIDATE_RAW = [
    os.environ.get("JUNG_RAW_ROOT"),
    "/work/upschrimpf1/mehrer/datasets/fMRI_movie_watching/spacetop/ds005256",
]

CASES = [
    ("54_cluster.json", "cluster_assignments_new54clusters.csv"),
    # Sep 2026 rebuttal rerun on the new-model partition (tests/test_new27_family.py).
    ("27_cluster.json", "cluster_assignments_new27clusters.csv"),
]


def _has_deps():
    try:
        import pandas  # noqa: F401
        return True
    except ImportError:
        return False


def _resolve_raw():
    for root in _CANDIDATE_RAW:
        if root and (Path(root) / "sub-0001").exists():
            return Path(root)
    return None


def _load_parser():
    path = _DATASET / "analysis" / "parse_clusters.py"
    spec = importlib.util.spec_from_file_location("jung_parse_clusters", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pytestmark = pytest.mark.skipif(not _has_deps(), reason="pandas not installed")


@pytest.mark.parametrize("json_name,csv_name", CASES)
def test_parse_reproduces_vendored_csv_bitwise(tmp_path, json_name, csv_name):
    raw = _resolve_raw()
    if raw is None:
        pytest.skip("raw ds005256 (sub-0001 events) not found (set JUNG_RAW_ROOT)")

    parser = _load_parser()
    got = parser.parse(str(_DATA / json_name), str(raw))

    out = tmp_path / csv_name
    got.to_csv(out, index=False)

    produced = out.read_text()
    vendored = (_DATA / csv_name).read_text()
    assert produced == vendored, f"{csv_name}: parsed CSV differs from vendored golden"


@pytest.mark.parametrize("json_name,csv_name", CASES)
def test_parse_maps_every_clip_exactly_once(json_name, csv_name):
    """No clip may be silently dropped between the model JSON and the design CSV.

    `parse` only *prints* its failures (filename, missing event, clip past video end), so
    a dropped clip would shrink a cluster's target regressor without any error. Pin that
    the CSV rows are exactly the JSON clips, each once, with the JSON cluster ID.
    """
    import json

    raw = _resolve_raw()
    if raw is None:
        pytest.skip("raw ds005256 (sub-0001 events) not found (set JUNG_RAW_ROOT)")

    clusters = json.loads((_DATA / json_name).read_text())
    expected = {v: c["cluster_id"] for c in clusters for v in c["video_ids"]}

    got = _load_parser().parse(str(_DATA / json_name), str(raw))

    assert len(got) == len(expected)
    assert dict(zip(got["clip_filename"], got["cluster_id"])) == expected
    assert (got["clip_duration_actual"] > 0).all()
    assert (got["clip_duration_actual"] <= 2).all()
