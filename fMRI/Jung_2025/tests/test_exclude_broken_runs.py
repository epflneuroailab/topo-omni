"""Excluding the four runs whose events file is a copy of another run's (Sep 2026).

In OpenNeuro ds005256 v1.1.0 four kept subjects have ONE run whose `events.tsv` holds only
video rows, identical to another run's, and no rating rows (see config.BROKEN_EVENT_RUNS).
The scan itself is the listed run, so the true video timing of that run is unknown. The published pipeline
modeled those runs with the wrong videos; for some clusters this dropped the subject
(empty target -> singular design, or an overrunning copied video -> target/others overlap).

The decided remedy (authors, 2026-09-23) is to drop exactly those four runs: the subject
contributes its other 12 runs. It is opt-in (`exclude_runs` / `--exclude-broken-runs`), so
the default path still reproduces the published maps bitwise, and outputs go to a
separate `*_exclbrokenruns` directory.

Synthetic mini-dataset tests need no fMRI data; the characterization tests at the bottom
are data-gated on the local ds005256 copy.
"""
import glob
import importlib.util
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_DATASET = Path(__file__).resolve().parent.parent
_ANALYSIS = _DATASET / "analysis"
TR = 0.46
N_VERT = 20
SUB = "sub-0001"


def _load(name, relpath):
    sys.path.insert(0, str(_ANALYSIS))
    spec = importlib.util.spec_from_file_location(name, _DATASET / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


config = _load("jung_config_excl", "config.py")
engine = _load("jung_engine_excl", "analysis/glm_engine.py")


# --- synthetic mini-dataset ---------------------------------------------------------
# 3 runs. run-01: video "va" 10 s + 150 s. run-02: a SHORT scan (200 TRs) whose events file
# is a copy of run-01's, so the copied "va" overruns it and spills 147 TRs into run-03.
# run-03: target video "vc" at 2 s for 60 s -> collides with the spill (published overlap
# drop). This mirrors sub-0021 ses-01 run-03 in the real data.
RUNS = {"run-01": 400, "run-02": 200, "run-03": 400}
EVENTS = {
    "run-01": [(10.0, 150.0, "va")],
    "run-02": [(10.0, 150.0, "va")],   # the copy
    "run-03": [(2.0, 60.0, "vc")],
}


def _make_dataset(root):
    rng = np.random.default_rng(0)
    for run, n in RUNS.items():
        func = root / SUB / "ses-01" / "func"
        func.mkdir(parents=True, exist_ok=True)
        stem = f"{SUB}_ses-01_task-alignvideo_acq-mb8_{run}"
        for hemi in ("L", "R"):
            import nibabel as nib
            darrays = [nib.gifti.GiftiDataArray(rng.normal(size=N_VERT).astype(np.float32))
                       for _ in range(n)]
            nib.save(nib.gifti.GiftiImage(darrays=darrays),
                     func / f"{stem}_hemi-{hemi}_space-fsaverage6_bold.func.gii")
        conf = pd.DataFrame(rng.normal(size=(n, 24)), columns=config.STANDARD_CONFOUNDS)
        conf.to_csv(func / f"{stem}_desc-confounds_timeseries.tsv", sep="\t", index=False)
        ev = pd.DataFrame([(o, d, "video", f"task-alignvideo/ses-01_{run}_order-01_content-{v}.mp4")
                           for o, d, v in EVENTS[run]],
                          columns=["onset", "duration", "trial_type", "stim_file"])
        ev.to_csv(func / f"{stem}_events.tsv", sep="\t", index=False)
    return root


def _clusters():
    # cluster 0 = the first 60 s of "vc" (run-03); cluster 1 = the first 60 s of "va" (run-01)
    rows = [(0, "vc", "ses-01", "run-03", 2 * k, 2 * k + 2) for k in range(30)]
    rows += [(1, "va", "ses-01", "run-01", 2 * k, 2 * k + 2) for k in range(30)]
    return pd.DataFrame(rows, columns=["cluster_id", "video_name", "session", "run",
                                       "clip_start", "clip_end"])


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    return _make_dataset(tmp_path_factory.mktemp("mini_ds"))


EXCL = {("ses-01", "run-02")}


def test_default_loaders_use_every_run(ds):
    assert engine.load_confounds(SUB, ds).shape == (1000, 24)
    assert engine.load_bold(SUB, "L", ds).shape == (1000, N_VERT)
    assert len(engine.load_subject_events(SUB, ds)) == 3


def test_excluded_run_is_removed_from_bold_and_confounds(ds):
    assert engine.load_confounds(SUB, ds, exclude_runs=EXCL).shape == (800, 24)
    assert engine.load_bold(SUB, "L", ds, exclude_runs=EXCL).shape == (800, N_VERT)


def test_excluded_run_is_removed_from_run_info_and_renumbered(ds):
    regressors = _load("jung_regressors_excl", "analysis/regressors.py")
    ri = regressors.load_run_info(SUB, ds, exclude_runs=EXCL)
    assert ri == {("ses-01", "run-01"): (0, 400), ("ses-01", "run-03"): (400, 800)}


def test_excluded_run_is_removed_from_events(ds):
    ev = engine.load_subject_events(SUB, ds, exclude_runs=EXCL)
    assert set(zip(ev["session"], ev["run"])) == {("ses-01", "run-01"), ("ses-01", "run-03")}


def test_published_behavior_drops_the_subject_on_overlap(ds):
    t_lh, t_rh = engine.run_subject_analysis(SUB, 0, ds, ds, _clusters(), ds)
    assert t_lh is None and t_rh is None


def test_excluding_the_broken_run_keeps_the_subject(ds):
    t_lh, t_rh = engine.run_subject_analysis(SUB, 0, ds, ds, _clusters(), ds, exclude_runs=EXCL)
    assert t_lh is not None and t_lh.shape == (N_VERT,) and np.isfinite(t_lh).all()


def test_empty_target_drops_with_an_explicit_reason(ds, capsys):
    # Target video only in the excluded run: the subject has no data for this cluster.
    clusters = _clusters()
    clusters.loc[clusters.cluster_id == 1, "run"] = "run-02"
    t_lh, _ = engine.run_subject_analysis(SUB, 1, ds, ds, clusters, ds, exclude_runs=EXCL)
    assert t_lh is None
    assert "no target timepoints" in capsys.readouterr().out


# --- configuration --------------------------------------------------------------------

def test_the_four_broken_runs_are_recorded_with_their_source():
    assert set(config.BROKEN_EVENT_RUNS) == {
        ("sub-0021", "ses-01", "run-03"), ("sub-0065", "ses-04", "run-02"),
        ("sub-0094", "ses-04", "run-01"), ("sub-0105", "ses-02", "run-02")}
    assert config.BROKEN_EVENT_RUNS[("sub-0021", "ses-01", "run-03")] == "run-04"
    assert config.BROKEN_EVENT_RUNS[("sub-0065", "ses-04", "run-02")] == "run-01"
    assert config.BROKEN_EVENT_RUNS[("sub-0094", "ses-04", "run-01")] == "run-02"
    assert config.BROKEN_EVENT_RUNS[("sub-0105", "ses-02", "run-02")] == "run-01"


def test_engine_copy_matches_config():
    assert engine.BROKEN_EVENT_RUNS == config.BROKEN_EVENT_RUNS
    assert engine.EXCLUDED_RUNS_SUFFIX == config.EXCLUDED_RUNS_SUFFIX == "_exclbrokenruns"


def test_broken_runs_belong_to_kept_subjects():
    dropped = {f"sub-{s}" for s in config.CONFOUND_DROPPED_SUBJECTS}
    for sub, _, _ in config.BROKEN_EVENT_RUNS:
        assert sub in config.CANONICAL_SUBJECTS and sub not in dropped


def test_runs_to_exclude_groups_by_subject():
    by_sub = engine.broken_runs_by_subject()
    assert by_sub["sub-0105"] == {("ses-02", "run-02")}
    assert len(by_sub) == 4


# --- contrast driver + per-cluster summary --------------------------------------------

def test_driver_flag_passes_exclusions_and_suffixes_default_output(monkeypatch, tmp_path):
    contrasts = _load("jung_contrasts_excl", "analysis/cluster_contrasts.py")
    csv = tmp_path / "c.csv"
    pd.DataFrame({"cluster_id": [0, 1], "cluster_label": ["a", "b"]}).to_csv(csv, index=False)
    seen = {}
    monkeypatch.setattr(contrasts, "run_full_analysis", lambda **kw: seen.update(kw) or True)
    assert contrasts.main(["--family", "new27", "--cluster-id", "0", "--cluster-file", str(csv),
                           "--derivatives-root", str(tmp_path), "--bids-root", str(tmp_path),
                           "--exclude-broken-runs"]) == 0
    assert seen["excluded_runs"] == engine.broken_runs_by_subject()
    assert Path(seen["output_dir"]).name == "cluster_contrasts_new27clusters_exclbrokenruns"


def test_driver_default_is_the_published_behavior(monkeypatch, tmp_path):
    contrasts = _load("jung_contrasts_excl2", "analysis/cluster_contrasts.py")
    csv = tmp_path / "c.csv"
    pd.DataFrame({"cluster_id": [0], "cluster_label": ["a"]}).to_csv(csv, index=False)
    seen = {}
    monkeypatch.setattr(contrasts, "run_full_analysis", lambda **kw: seen.update(kw) or True)
    contrasts.main(["--family", "new54", "--cluster-id", "0", "--cluster-file", str(csv),
                    "--derivatives-root", str(tmp_path), "--bids-root", str(tmp_path)])
    assert seen["excluded_runs"] is None
    assert Path(seen["output_dir"]).name == "cluster_contrasts_new54clusters"


# Identical subject maps -> zero group variance; scipy warns, irrelevant to what is tested.
@pytest.mark.filterwarnings("ignore:Precision loss occurred in moment calculation")
def test_summary_records_the_excluded_runs(ds, tmp_path):
    out = tmp_path / "res"
    csv = tmp_path / "clusters.csv"
    _clusters().to_csv(csv, index=False)
    # The same synthetic subject three times: the group test needs n >= 3.
    ok = engine.run_full_analysis(target_id=0, output_dir=out, bold_dir=ds, confounds_dir=ds,
                                  bids_dir=ds, cluster_file=str(csv), subjects=[SUB] * 3,
                                  excluded_runs={SUB: EXCL})
    assert ok
    import json
    s = json.loads((out / "group_level" / "group_cluster-00_space-fsaverage6_summary.json").read_text())
    assert s["excluded_runs"] == [[SUB, "ses-01", "run-02"]]


# --- characterization on the real dataset (data-gated) ---------------------------------

_BIDS = [os.environ.get("JUNG_RAW_ROOT"),
         "/work/upschrimpf1/mehrer/datasets/fMRI_movie_watching/spacetop/ds005256"]


def _bids():
    return next((Path(b) for b in _BIDS if b and (Path(b) / "sub-0021").is_dir()), None)


@pytest.mark.parametrize("key", sorted(config.BROKEN_EVENT_RUNS))
def test_broken_events_file_repeats_its_source_videos_without_ratings(key):
    root = _bids()
    if root is None:
        pytest.skip("raw ds005256 not found (set JUNG_RAW_ROOT)")
    sub, ses, run = key
    read = lambda r: pd.read_csv(next((root / sub / ses / "func").glob(
        f"*task-alignvideo*_{r}_events.tsv")), sep="\t")
    broken, source = read(run), read(config.BROKEN_EVENT_RUNS[key])
    videos = lambda ev: ev.loc[ev.trial_type == "video", ["onset", "duration", "stim_file"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(videos(broken), videos(source))
    assert not broken.trial_type.str.startswith("rating").any()   # the malformed file
    assert source.trial_type.str.startswith("rating").any()       # the complete file


def _predicted_n(family, exclude):
    """Per-cluster n from events + confound run lengths only (no BOLD)."""
    import contextlib
    import io
    regressors = _load("jung_regressors_pred", "analysis/regressors.py")
    root = _bids()
    deriv = root / "derivatives"
    dropped = {f"sub-{s}" for s in config.CONFOUND_DROPPED_SUBJECTS}
    kept = [s for s in config.CANONICAL_SUBJECTS if s not in dropped]
    by_sub = engine.broken_runs_by_subject() if exclude else {}
    df = pd.read_csv(_DATASET / config.CLUSTERS[family]["assignments_csv"])
    subj = {}
    for s in kept:
        ri, cum = {}, 0
        for p in sorted(glob.glob(f"{deriv}/{s}/ses-*/func/*alignvideo*desc-confounds_timeseries.tsv")):
            k = re.search(r"(ses-\d+)_task-alignvideo_acq-mb8_(run-\d+)", p).groups()
            if k in by_sub.get(s, set()):
                continue
            n = sum(1 for _ in open(p)) - 1
            ri[k], cum = (cum, cum + n), cum + n
        subj[s] = (engine.load_subject_events(s, root, exclude_runs=by_sub.get(s)), ri)
    n_by_cluster = {}
    for c in sorted(df.cluster_id.unique()):
        n = 0
        for s, (ev, ri) in subj.items():
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    tb, _ = regressors.create_cluster_regressors(c, df, ev, ri, return_binary=True)
            except ValueError:
                continue
            n += tb.sum() > 0
        n_by_cluster[int(c)] = n
    return n_by_cluster


def test_new27_predicted_subject_counts_with_exclusion():
    if _bids() is None or not (_bids() / "derivatives").is_dir():
        pytest.skip("ds005256 + derivatives not found")
    n = _predicted_n("new27", exclude=True)
    # Only the two clusters whose videos a subject saw exclusively in its broken run.
    assert {c: k for c, k in n.items() if k != 78} == {8: 77, 14: 77}


def test_paper_clusters_keep_78_with_exclusion():
    if _bids() is None or not (_bids() / "derivatives").is_dir():
        pytest.skip("ds005256 + derivatives not found")
    n = _predicted_n("new54", exclude=True)
    assert all(n[c] == 78 for c in (5, 32, 49, 6, 30, 31))
