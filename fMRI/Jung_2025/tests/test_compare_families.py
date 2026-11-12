"""compare_families — one family's clusters and group maps against a reference family's.

Written for the new16-vs-new27 comparison (Sep 2026): new16 is new27 cut coarser, so each
new16 cluster either has the identical clips of one new27 cluster (same GLM design, so the
map must reproduce: map_r ~ 1) or merges several (compared with each part). Unit tests
use synthetic inputs; no fMRI data required.
"""
import importlib.util
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


def _csv(path, rows):
    df = pd.DataFrame(rows, columns=["cluster_id", "video_name", "clip_filename"])
    df["cluster_label"] = df.groupby("cluster_id")["video_name"].transform("first")
    df.to_csv(path, index=False)
    return path


def _write(engine, group, cid, t):
    group.mkdir(parents=True, exist_ok=True)
    prefix = f"group_cluster-{cid:02d}_space-fsaverage6"
    engine.save_gifti(t[:N_VERT], group / f"{prefix}_hemi-L_tstat.func.gii", "L")
    engine.save_gifti(t[N_VERT:], group / f"{prefix}_hemi-R_tstat.func.gii", "R")


@pytest.fixture
def setup(tmp_path):
    """A = {0: identical to ref 0, 1: merge of ref 1+2}; ref 3 shares no clips."""
    engine = _load("jung_engine_cmp", "glm_engine.py")
    rng = np.random.default_rng(0)
    signal = rng.normal(0, 1, 2 * N_VERT)
    signal[:5000] += 6.0                                   # a strong patch -> FDR survivors
    other = rng.normal(0, 1, 2 * N_VERT)
    other[-5000:] += 6.0
    a_grp, b_grp = tmp_path / "a" / "group_level", tmp_path / "b" / "group_level"
    _write(engine, a_grp, 0, signal)
    _write(engine, a_grp, 1, other)
    _write(engine, b_grp, 0, signal)                       # identical clips -> identical map
    _write(engine, b_grp, 1, other + rng.normal(0, 0.5, 2 * N_VERT))
    _write(engine, b_grp, 2, rng.normal(0, 1, 2 * N_VERT))
    _write(engine, b_grp, 3, signal)                       # no shared clips, best map match for 0
    a_csv = _csv(tmp_path / "a.csv", [(0, "X", "x1"), (0, "X", "x2"),
                                      (1, "Y", "y1"), (1, "Z", "z1"), (1, "Z", "z2")])
    b_csv = _csv(tmp_path / "b.csv", [(0, "X", "x1"), (0, "X", "x2"), (1, "Y", "y1"),
                                      (2, "Z", "z1"), (2, "Z", "z2"), (3, "W", "w1")])
    return _load("jung_compare_families", "compare_families.py"), a_csv, a_grp, b_csv, b_grp


def test_pairs_only_for_shared_clips_with_relation(setup):
    cf, a_csv, a_grp, b_csv, b_grp = setup
    pairs, _, _ = cf.compare(a_csv, a_grp, b_csv, b_grp)
    got = {(int(a), int(b)): rel for a, b, rel in zip(pairs.cluster_id, pairs.ref_id, pairs.relation)}
    assert got == {(0, 0): "identical", (1, 1): "a_contains_b", (1, 2): "a_contains_b"}
    p = pairs.set_index(["cluster_id", "ref_id"])
    assert p.loc[(1, 2), "jaccard"] == pytest.approx(2 / 3)
    assert p.loc[(1, 2), "frac_a_in_b"] == pytest.approx(2 / 3)
    assert p.loc[(1, 2), "frac_b_in_a"] == 1.0


def test_identical_maps_score_perfectly(setup):
    cf, *paths = setup
    pairs, _, _ = cf.compare(*paths)
    row = pairs[(pairs.cluster_id == 0) & (pairs.ref_id == 0)].iloc[0]
    assert row.map_r == pytest.approx(1.0)
    assert row.dice_fdr == pytest.approx(1.0) and row.dice_top10 == pytest.approx(1.0)
    assert row.n_fdr == row.ref_n_fdr > 0


def test_per_cluster_best_matches_and_parts(setup):
    cf, *paths = setup
    _, per, r_mat = cf.compare(*paths)
    per = per.set_index("cluster_id")
    assert r_mat.shape == (2, 4)
    assert per.loc[0, "best_clip_ref_id"] == 0 and per.loc[0, "best_clip_relation"] == "identical"
    assert per.loc[0, "best_map_ref_id"] in (0, 3)            # both carry the same map
    assert per.loc[1, "best_map_ref_id"] == 1
    assert per.loc[1, "n_parts"] == 2 and per.loc[1, "parts"] == "1;2"
    assert per.loc[1, "mean_map_r_to_parts"] == pytest.approx(
        np.mean([r_mat.loc[1, 1], r_mat.loc[1, 2]]))


def test_reference_ids_restrict_the_reference(setup):
    cf, *paths = setup
    pairs, per, r_mat = cf.compare(*paths, ref_ids=[2])
    assert set(pairs.ref_id) == {2} and list(r_mat.columns) == [2]
    assert per.set_index("cluster_id").loc[0, "best_clip_ref_id"] == -1


def test_all_pairs_and_ids_give_a_targeted_table(setup):
    cf, *paths = setup
    pairs, per, r_mat = cf.compare(*paths, ids=[0], all_pairs=True)
    assert set(pairs.cluster_id) == {0} and list(per.cluster_id) == [0]
    assert list(pairs.ref_id) == [0, 1, 2, 3] and r_mat.shape == (1, 4)
    p = pairs.set_index("ref_id")
    assert p.loc[3, "relation"] == "disjoint" and p.loc[3, "n_shared"] == 0
    assert p.loc[3, "map_r"] == pytest.approx(1.0)             # no shared clips, same map
    assert p.loc[3, "dice_top10_all"] == pytest.approx(1.0)
    assert per.loc[0, "best_clip_ref_id"] == 0                 # disjoint pairs never win on clips


def test_top10_all_is_a_tenth_of_all_vertices(setup):
    cf, _, a_grp, _, _ = setup
    m = cf.load_map(a_grp, 0)
    assert m["top_all"].sum() == pytest.approx(0.1 * 2 * N_VERT, abs=2)
    assert 0 < m["top"].sum() < m["fdr"].sum()


def test_missing_maps_leave_nan_not_crash(setup, tmp_path):
    cf, a_csv, a_grp, b_csv, _ = setup
    pairs, per, _ = cf.compare(a_csv, a_grp, b_csv, tmp_path / "nowhere")
    assert pairs["map_r"].isna().all() and (pairs["ref_n_fdr"] == -1).all()
    assert (per["best_map_ref_id"] == -1).all()


def test_cli_writes_tables_and_heatmap(tmp_path):
    pytest.importorskip("matplotlib")
    engine = _load("jung_engine_cmp_cli", "glm_engine.py")
    rng = np.random.default_rng(2)
    t = rng.normal(0, 1, 2 * N_VERT)
    t[:3000] += 6
    new16 = tmp_path / "cluster_contrasts_new16clusters_exclbrokenruns" / "group_level"
    new27 = tmp_path / "cluster_contrasts_new27clusters_exclbrokenruns" / "group_level"
    _write(engine, new16, 2, t)                            # new16 2 == new27 4 (config.NEW16_FROM_NEW27)
    _write(engine, new27, 4, t)
    cf = _load("jung_compare_families_cli", "compare_families.py")
    assert cf.main(["--family", "new16", "--reference-family", "new27",
                    "--results-root", str(tmp_path), "--exclude-broken-runs"]) == 0
    out = new16.parent
    per = pd.read_csv(out / "compare_new27_per_cluster.csv").set_index("cluster_id")
    assert len(per) == 16
    assert per.loc[2, "best_clip_relation"] == "identical" and per.loc[2, "best_clip_ref_id"] == 4
    assert per.loc[2, "best_clip_map_r"] == pytest.approx(1.0)
    assert (out / "compare_new27.md").exists() and (out / "compare_new27_map_r.png").exists()
