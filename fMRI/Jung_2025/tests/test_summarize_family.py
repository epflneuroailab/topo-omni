"""summarize_family — one table row per cluster from the group maps (Sep 2026).

Turns a family's group_level/ maps + cluster-assignment CSV into the table the authors
read to decide which clusters "validate": max t, the joint-hemisphere one-tailed BH-FDR
q<.05 threshold and survivor count (the paper's stated procedure and exactly what the
figures threshold on — visualize_clusters.calculate_fdr_threshold), n subjects entering
the group test, cluster content, clips left UNMODELED by the design, and the best-matching
published new54 cluster.

Unmodeled clips: the published design marks as "others" every video event whose video
is NOT among the target cluster's videos (regressors.create_cluster_regressors). Clips
of a target video that were assigned to OTHER clusters are therefore in neither regressor
(they fall into the implicit baseline with the rating epochs). Harmless for the published
single-video clusters, but large for mixed clusters, so it is reported per cluster.

Unit tests use synthetic inputs; the golden test characterizes the on-disk published new54
group maps (data-gated).
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_DATASET = Path(__file__).resolve().parent.parent
_ANALYSIS = _DATASET / "analysis"
N_VERT = 40962


def _load(name, filename):
    sys.path.insert(0, str(_ANALYSIS))
    spec = importlib.util.spec_from_file_location(name, _ANALYSIS / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _csv(rows):
    # (cluster_id, video_name, clip_filename) -> minimal assignments CSV frame
    df = pd.DataFrame(rows, columns=["cluster_id", "video_name", "clip_filename"])
    df["cluster_label"] = df.groupby("cluster_id")["video_name"].transform("first")
    return df


ROWS = [
    (0, "A", "a1"), (0, "A", "a2"),
    (1, "A", "a3"), (1, "B", "b1"), (1, "B", "b2"),
    (2, "C", "c1"),
]


def _write_group(engine, group, cid, lh, rh, summary=None):
    group.mkdir(parents=True, exist_ok=True)
    prefix = f"group_cluster-{cid:02d}_space-fsaverage6"
    engine.save_gifti(lh, group / f"{prefix}_hemi-L_tstat.func.gii", "L")
    engine.save_gifti(rh, group / f"{prefix}_hemi-R_tstat.func.gii", "R")
    if summary is not None:
        (group / f"{prefix}_summary.json").write_text(json.dumps(summary))


@pytest.fixture
def setup(tmp_path):
    engine = _load("jung_engine_sumfam", "glm_engine.py")
    rng = np.random.default_rng(0)
    group = tmp_path / "group_level"
    strong = (rng.normal(3.0, 1.0, N_VERT), rng.normal(3.0, 1.0, N_VERT))
    _write_group(engine, group, 0, *strong, summary={"n_subjects": 78, "subjects": ["x"] * 78})
    _write_group(engine, group, 1, -np.abs(rng.normal(0, 1, N_VERT)), -np.abs(rng.normal(0, 1, N_VERT)))
    csv = tmp_path / "assign.csv"
    _csv(ROWS).to_csv(csv, index=False)
    return _load("jung_summarize_family", "summarize_family.py"), csv, group, strong


def test_one_row_per_cluster_with_status(setup):
    sf, csv, group, _ = setup
    t = sf.summarize(csv, group)
    assert list(t["cluster_id"]) == [0, 1, 2]
    assert list(t["status"]) == ["done", "done", "missing"]


def test_content_columns(setup):
    sf, csv, group, _ = setup
    t = sf.summarize(csv, group).set_index("cluster_id")
    assert t.loc[1, "n_clips"] == 3
    assert t.loc[1, "n_videos"] == 2
    assert t.loc[1, "videos"] == "A;B"


def test_unmodeled_clips_are_target_video_clips_in_other_clusters(setup):
    sf, csv, group, _ = setup
    t = sf.summarize(csv, group).set_index("cluster_id")
    assert t.loc[0, "n_unmodeled_clips"] == 1   # a3 (video A, cluster 1)
    assert t.loc[1, "n_unmodeled_clips"] == 2   # a1, a2 (video A, cluster 0)
    assert t.loc[2, "n_unmodeled_clips"] == 0


def test_fdr_survivors_match_the_figure_threshold(setup):
    sf, csv, group, (lh, rh) = setup
    viz = _load("jung_viz_sumfam", "visualize_clusters.py")
    t = sf.summarize(csv, group).set_index("cluster_id")
    lh32, rh32 = lh.astype(np.float32), rh.astype(np.float32)   # GIFTIs store float32
    thr, _, _ = viz.calculate_fdr_threshold(lh32, rh32, q=0.05, df=77)
    assert t.loc[0, "n_fdr_survivors"] == int((np.concatenate([lh32, rh32]) >= thr).sum()) > 0
    assert t.loc[0, "fdr_t"] == pytest.approx(thr)
    assert t.loc[0, "max_t"] == pytest.approx(max(lh32.max(), rh32.max()))
    assert t.loc[1, "n_fdr_survivors"] == 0
    assert not np.isfinite(t.loc[1, "fdr_t"])


def test_subject_count_comes_from_per_cluster_summary_else_assumed(setup):
    sf, csv, group, _ = setup
    t = sf.summarize(csv, group, n_subjects_expected=78).set_index("cluster_id")
    assert t.loc[0, "n_subjects"] == 78 and t.loc[0, "n_subjects_source"] == "summary"
    assert t.loc[1, "n_subjects"] == 78 and t.loc[1, "n_subjects_source"] == "assumed"


def test_unexpected_subject_count_is_flagged(tmp_path):
    engine = _load("jung_engine_sumfam2", "glm_engine.py")
    group = tmp_path / "group_level"
    z = np.zeros(N_VERT)
    _write_group(engine, group, 0, z, z, summary={"n_subjects": 77, "subjects": ["x"] * 77})
    csv = tmp_path / "assign.csv"
    _csv([(0, "A", "a1")]).to_csv(csv, index=False)
    sf = _load("jung_summarize_family2", "summarize_family.py")
    row = sf.summarize(csv, group, n_subjects_expected=78).iloc[0]
    assert row["n_subjects"] == 77
    assert row["status"] == "n_subjects!=78"


def test_best_reference_match_by_clip_jaccard(setup, tmp_path):
    sf, csv, group, _ = setup
    ref = tmp_path / "ref.csv"
    _csv([(5, "A", "a1"), (5, "A", "a2"), (5, "A", "a3"), (9, "C", "c1")]).to_csv(ref, index=False)
    t = sf.summarize(csv, group, reference_csv=ref, reference_ids=[5, 9]).set_index("cluster_id")
    assert t.loc[0, "best_ref_id"] == 5
    assert t.loc[0, "best_ref_jaccard"] == pytest.approx(2 / 3)
    assert t.loc[2, "best_ref_id"] == 9
    assert t.loc[2, "best_ref_jaccard"] == pytest.approx(1.0)


def test_markdown_table_has_a_row_per_cluster(setup):
    sf, csv, group, _ = setup
    md = sf.to_markdown(sf.summarize(csv, group))
    lines = [l for l in md.splitlines() if l.startswith("|")]
    assert len(lines) == 2 + 3          # header + separator + 3 clusters
    assert "cluster_id" in lines[0]


def test_excluded_runs_are_listed_per_cluster(tmp_path):
    engine = _load("jung_engine_sumfam3", "glm_engine.py")
    group = tmp_path / "group_level"
    z = np.zeros(N_VERT)
    _write_group(engine, group, 0, z, z, summary={
        "n_subjects": 78, "subjects": ["x"] * 78,
        "excluded_runs": [["sub-0021", "ses-01", "run-03"], ["sub-0065", "ses-04", "run-02"]]})
    _write_group(engine, group, 1, z, z, summary={"n_subjects": 78, "subjects": ["x"] * 78,
                                                  "excluded_runs": []})
    csv = tmp_path / "assign.csv"
    _csv([(0, "A", "a1"), (1, "B", "b1")]).to_csv(csv, index=False)
    t = _load("jung_summarize_family3", "summarize_family.py").summarize(csv, group).set_index("cluster_id")
    assert t.loc[0, "excluded_runs"] == "sub-0021:ses-01/run-03;sub-0065:ses-04/run-02"
    assert t.loc[1, "excluded_runs"] == ""


def test_cli_reads_the_exclbrokenruns_directory(tmp_path):
    engine = _load("jung_engine_sumfam4", "glm_engine.py")
    group = tmp_path / "cluster_contrasts_new27clusters_exclbrokenruns" / "group_level"
    rng = np.random.default_rng(1)
    _write_group(engine, group, 3, rng.normal(3, 1, N_VERT), rng.normal(3, 1, N_VERT),
                 summary={"n_subjects": 78, "subjects": ["x"] * 78, "excluded_runs": []})
    sf = _load("jung_summarize_family4", "summarize_family.py")
    assert sf.main(["--family", "new27", "--results-root", str(tmp_path), "--exclude-broken-runs"]) == 0
    t = pd.read_csv(group.parent / "summary_table.csv").set_index("cluster_id")
    assert len(t) == 27
    assert t.loc[3, "status"] == "done"
    assert (t.drop(index=3)["status"] == "missing").all()


# --- golden: characterize the published new54 group maps (data-gated) --------------

_DERIV = [
    os.environ.get("JUNG_DERIVATIVES_ROOT"),
    "/work/upschrimpf1/mehrer/datasets/fMRI_movie_watching/spacetop/ds005256/derivatives",
]

# Measured 2026-09-23 on the on-disk published group maps (float32 GIFTIs, df=77): max t
# matches data/cluster_assignments/CLUSTER_INDEX.md to 2 dp; survivor counts agree with
# the index to <0.4% (the index numbers came from an unrecorded July count variant).
PUBLISHED = {5: (4.73, 78171), 32: (4.45, 67924), 49: (4.38, 64724),
             6: (4.17, 55899), 30: (4.32, 65684), 31: (4.32, 71113), 7: (3.68, 0)}


def test_golden_published_new54_table():
    root = next((Path(d) for d in _DERIV if d and (Path(d) / "cluster_contrasts_new54clusters").is_dir()), None)
    if root is None:
        pytest.skip("published new54 group maps not found (set JUNG_DERIVATIVES_ROOT)")
    sf = _load("jung_summarize_family_golden", "summarize_family.py")
    csv = _DATASET / "data" / "cluster_assignments" / "cluster_assignments_new54clusters.csv"
    t = sf.summarize(csv, root / "cluster_contrasts_new54clusters" / "group_level").set_index("cluster_id")
    assert len(t) == 54 and set(t["status"]) == {"done"}
    for cid, (max_t, n_surv) in PUBLISHED.items():
        assert round(t.loc[cid, "max_t"], 2) == max_t, cid
        assert t.loc[cid, "n_fdr_survivors"] == n_surv, cid
