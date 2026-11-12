"""SLURM array launcher for a whole cluster family (Sep 2026, new27 rebuttal rerun).

One array task per cluster ID: the per-subject GLM re-fit from the fsaverage6 BOLD
(78 x 13 runs, ~5 GB/subject) plus the group test — ~2 h and <52 GB per cluster on
bigmem (measured July 2026 calibration jobs). These tests pin the header against the
config, and exercise the script body in DRY_RUN mode (prints the command, runs nothing),
so no SLURM or fMRI data is needed.
"""
import importlib.util
import os
import re
import subprocess
from pathlib import Path

import pytest

_DATASET = Path(__file__).resolve().parent.parent
SCRIPT = _DATASET / "slurm" / "cluster_contrasts_array.sbatch"


def _config():
    spec = importlib.util.spec_from_file_location("jung_config_slurm", _DATASET / "config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _header(key):
    m = re.search(rf"^#SBATCH --{key}=(\S+)", SCRIPT.read_text(), re.M)
    assert m, f"missing #SBATCH --{key}"
    return m.group(1)


def _dry_run(tmp_path, **env):
    base = {"PATH": os.environ["PATH"], "DRY_RUN": "1", "SLURM_SUBMIT_DIR": str(_DATASET),
            "SLURM_ARRAY_TASK_ID": "3", "DERIV": str(tmp_path / "deriv"), "BIDS": str(tmp_path / "bids")}
    base.update(env)
    return subprocess.run(["bash", str(SCRIPT)], env=base, capture_output=True, text=True)


def test_script_is_valid_bash():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


def test_array_covers_every_new27_cluster():
    assert _header("array") == f"0-{_config().CLUSTERS['new27']['n_clusters'] - 1}"


def test_resources_match_the_calibrated_heavy_run():
    assert _header("partition") == "bigmem"
    assert _header("qos") == "bigmem"
    assert _header("account") == "upschrimpf1"
    assert int(_header("mem").rstrip("G")) >= 52
    h, m, s = map(int, _header("time").split(":"))
    assert h >= 4


def test_csv_naming_convention_matches_config_for_every_family():
    # The script derives the CSV as data/cluster_assignments/cluster_assignments_<fam>clusters.csv
    for key, fam in _config().CLUSTERS.items():
        assert fam["assignments_csv"] == f"data/cluster_assignments/cluster_assignments_{key}clusters.csv"
        assert fam["derivatives_subdir"] == f"cluster_contrasts_{key}clusters"


def test_dry_run_builds_the_contrast_command(tmp_path):
    r = _dry_run(tmp_path)
    assert r.returncode == 0, r.stderr
    cmd = r.stdout.strip().splitlines()[-1]
    assert "analysis/cluster_contrasts.py" in cmd
    assert "--family new27" in cmd
    assert "--cluster-id 3" in cmd
    assert f"--cluster-file {_DATASET}/data/cluster_assignments/cluster_assignments_new27clusters.csv" in cmd
    assert f"--derivatives-root {tmp_path}/deriv" in cmd
    assert f"--bids-root {tmp_path}/bids" in cmd
    # Default = the decided analysis (authors, 2026-09-23): the 4 copied-events runs dropped,
    # written to a separate *_exclbrokenruns directory (tests/test_exclude_broken_runs.py).
    assert cmd.split("--output-dir ")[1].split()[0] == \
        f"{tmp_path}/deriv/cluster_contrasts_new27clusters_exclbrokenruns"
    assert "--exclude-broken-runs" in cmd
    assert "--subjects" not in cmd


def test_dry_run_can_select_the_published_analysis(tmp_path):
    r = _dry_run(tmp_path, EXCLUDE_BROKEN_RUNS="0")
    assert r.returncode == 0, r.stderr
    cmd = r.stdout.strip().splitlines()[-1]
    assert "--exclude-broken-runs" not in cmd
    assert cmd.split("--output-dir ")[1].split()[0] == f"{tmp_path}/deriv/cluster_contrasts_new27clusters"


def test_dry_run_honours_results_root_family_and_subjects(tmp_path):
    r = _dry_run(tmp_path, FAMILY="new54", RESULTS_ROOT=str(tmp_path / "res"),
                 SUBJECTS="sub-0001 sub-0002")
    assert r.returncode == 0, r.stderr
    cmd = r.stdout.strip().splitlines()[-1]
    assert "--family new54" in cmd
    assert f"--output-dir {tmp_path}/res/cluster_contrasts_new54clusters_exclbrokenruns" in cmd
    assert cmd.endswith("--subjects sub-0001 sub-0002")


def test_suffix_matches_config():
    assert "_exclbrokenruns" == _config().EXCLUDED_RUNS_SUFFIX
    assert 'SUFFIX="_exclbrokenruns"' in SCRIPT.read_text()


@pytest.mark.parametrize("missing", ["DERIV", "BIDS"])
def test_required_paths_are_enforced(tmp_path, missing):
    env = {"PATH": os.environ["PATH"], "DRY_RUN": "1", "SLURM_SUBMIT_DIR": str(_DATASET),
           "SLURM_ARRAY_TASK_ID": "0", "DERIV": "/d", "BIDS": "/b"}
    del env[missing]
    r = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)
    assert r.returncode != 0
    assert missing in r.stderr


def test_wrong_submit_dir_is_rejected(tmp_path):
    r = _dry_run(tmp_path, SLURM_SUBMIT_DIR=str(tmp_path))
    assert r.returncode != 0
    assert "Jung_2025" in r.stderr
