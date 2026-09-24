"""Group-level summary is written PER CLUSTER (bug fix, Sep 2026).

The engine used to write every cluster's group summary to the one shared
`group_level/summary.json`, so each cluster overwrote the previous one (the on-disk
new54 group_level/ holds 54 clusters' maps but a single summary.json, for cluster 4) and
parallel SLURM array tasks raced on the same file. The per-cluster summary also records
WHICH subjects entered the group test: the engine's blanket `except` drops a subject on
ANY error (not only the 5 confound drops), so a transient I/O failure inside one array
task would otherwise silently yield n=77 for that cluster only.

Synthetic subject t-maps + reuse mode: pure numpy/nibabel, no fMRI data, runs in seconds.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_DATASET = Path(__file__).resolve().parent.parent
_ANALYSIS = _DATASET / "analysis"
N_VERT = 40962


def _load_engine():
    sys.path.insert(0, str(_ANALYSIS))
    spec = importlib.util.spec_from_file_location("jung_glm_engine_summary", _ANALYSIS / "glm_engine.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_subject_maps(engine, root, cid, subjects, shift):
    rng = np.random.default_rng(cid)
    for s in subjects:
        d = root / "subject_level" / f"cluster-{cid:02d}" / s
        d.mkdir(parents=True)
        for hemi in ("L", "R"):
            engine.save_gifti(rng.normal(shift, 1.0, N_VERT), d / f"{s}_hemi-{hemi}_tstat.func.gii", hemi)


def _run(engine, root, cid):
    return engine.run_full_analysis(
        target_id=cid, output_dir=root, bold_dir=None, confounds_dir=None, bids_dir=None,
        cluster_file=None, reuse_subject_maps=True, subject_maps_root=str(root))


def test_each_cluster_keeps_its_own_summary(tmp_path):
    engine = _load_engine()
    subjects = engine.CANONICAL_SUBJECTS[:5]
    _write_subject_maps(engine, tmp_path, 0, subjects, shift=0.0)
    _write_subject_maps(engine, tmp_path, 1, subjects[:4], shift=2.0)

    assert _run(engine, tmp_path, 0)
    assert _run(engine, tmp_path, 1)

    group = tmp_path / "group_level"
    s0 = json.loads((group / "group_cluster-00_space-fsaverage6_summary.json").read_text())
    s1 = json.loads((group / "group_cluster-01_space-fsaverage6_summary.json").read_text())

    assert s0["target_id"] == "0" and s1["target_id"] == "1"
    assert s0["n_subjects"] == 5 and s0["df"] == 4
    assert s1["n_subjects"] == 4 and s1["df"] == 3
    assert s0["subjects"] == subjects
    assert s1["subjects"] == subjects[:4]
    assert s1["max_t_lh"] > s0["max_t_lh"]


def test_legacy_summary_json_is_still_written(tmp_path):
    # Back-compat: existing readers (test_render_only_from_subject_maps) use summary.json.
    engine = _load_engine()
    _write_subject_maps(engine, tmp_path, 3, engine.CANONICAL_SUBJECTS[:3], shift=1.0)
    assert _run(engine, tmp_path, 3)
    legacy = json.loads((tmp_path / "group_level" / "summary.json").read_text())
    assert legacy["target_id"] == "3"
