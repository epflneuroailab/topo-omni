"""Shared helpers for the per-figure reproduction scripts (scripts/figures/figure_*.py).

Each figure script declares a list of STEPS and calls :func:`run_figure`. A step is a dict:

    {
      "name":    "unified selectivity map",     # human label
      "module":  "src.visualize.selectivity",   # python -m <module>
      "args":    ["--top-k-pct", "10", ...],     # CLI args (may include OUT / SAVE_DIR placeholders)
      "stage":   "plot" | "compute",             # "compute" runs only in --input-source raw
      "out_dir_arg": "--out-dir",                # optional: flag that points the module at OUT_DIR
      "collect": ["<MODEL_TITLE>/response_profiles/*_top1.png"],  # optional: globs (rel to SAVE_DIR) to copy into OUT_DIR
      "raw_only": False,                          # skip entirely in precomputed mode
    }

Placeholders substituted in ``args`` at run time: ``{SAVE_DIR}``, ``{MODEL_TITLE}``, ``{OUT_DIR}``.

The two paths (mirrors ``fMRI/``):
  * precomputed (default): run only ``stage == "plot"`` steps, reading intermediates from the
    downloaded cut (SAVE_DIR). No GPU, no stimuli.
  * raw: run ``compute`` steps first (download the HF model, run it on stimuli to produce the
    intermediates), then the ``plot`` steps.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from glob import glob
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_TITLE = "topo-omni"


def build_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-source", choices=("precomputed", "raw"), default="precomputed",
                   help="Plot from the hosted precomputed cut (default) or recompute from the HF model + stimuli.")
    p.add_argument("--derivatives-root", default=os.getenv("SAVE_DIR", "results"),
                   help="Cut root (== SAVE_DIR): holds <MODEL_TITLE>/<category>/... intermediates.")
    p.add_argument("--out", default=str(REPO_ROOT / "figures_out"),
                   help="Base output directory; panels land in <out>/figure_<id>/.")
    p.add_argument("--model", default=None,
                   help="Raw mode only: HF id or local checkpoint dir (default: $TOPO_OMNI_MODEL).")
    return p


def _subprocess_env(save_dir: str, model: str | None) -> dict:
    env = os.environ.copy()
    env["SAVE_DIR"] = str(save_dir)
    env["REPO_DIR"] = env.get("REPO_DIR", str(REPO_ROOT))
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    if model:
        env["TOPO_OMNI_MODEL"] = model
    return env


def _subst(arg: str, save_dir: str, out_dir: Path) -> str:
    return (str(arg).replace("{SAVE_DIR}", str(save_dir))
                    .replace("{MODEL_TITLE}", MODEL_TITLE)
                    .replace("{OUT_DIR}", str(out_dir)))


def _collect(save_dir: str, patterns, out_dir: Path) -> int:
    n = 0
    for pat in patterns:
        for f in sorted(glob(os.path.join(save_dir, _subst(pat, save_dir, out_dir)))):
            shutil.copy2(f, out_dir / Path(f).name)
            n += 1
    return n


def run_figure(figure_id: str, steps: list, argv=None) -> int:
    args = build_parser(f"Reproduce Figure {figure_id} panels.").parse_args(argv)
    save_dir = args.derivatives_root
    out_dir = Path(args.out) / f"figure_{figure_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    env = _subprocess_env(save_dir, args.model)

    print(f"=== Figure {figure_id} ({args.input_source}) -> {out_dir} ===")
    ran = 0
    for step in steps:
        stage = step.get("stage", "plot")
        if step.get("raw_only") and args.input_source != "raw":
            print(f"  - skip [{step['name']}] (raw-only)")
            continue
        if stage == "compute" and args.input_source != "raw":
            continue  # precomputed: intermediates already in the cut

        step_args = [_subst(a, save_dir, out_dir) for a in step.get("args", [])]
        if step.get("out_dir_arg"):
            step_args += [step["out_dir_arg"], str(out_dir)]

        # `module` -> `python -m pkg.mod`; `script` -> `python path/to/script.py`
        # (topo-discover/ is hyphenated, so its scripts run by path, not as modules).
        if step.get("module"):
            cmd = [sys.executable, "-m", step["module"], *step_args]
        else:
            cmd = [sys.executable, _subst(step["script"], save_dir, out_dir), *step_args]
        print(f"  - [{step['name']}] {' '.join(cmd)}")

        # precomputed: every plot step must succeed. raw: compute steps are required, but a plot
        # step whose (non-self-generating) intermediate isn't present is best-effort — warn and
        # continue so the self-generating panels (tonotopy/retinotopy) still render. Populate
        # SAVE_DIR first (e.g. via scripts/*_response_profiles.sh with your stimuli) for the rest.
        required = args.input_source == "precomputed" or stage == "compute"
        result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
        if result.returncode != 0:
            if required:
                raise SystemExit(f"  ✗ step [{step['name']}] failed (exit {result.returncode})")
            print(f"      ⚠ skip [{step['name']}] — intermediate not available in raw mode "
                  f"(provide stimuli + run the analysis first). Continuing.")
            continue
        ran += 1

        if step.get("collect"):
            got = _collect(save_dir, step["collect"], out_dir)
            print(f"      collected {got} panel(s) -> {out_dir}")

    print(f"=== Figure {figure_id}: {ran} step(s) done. Panels in {out_dir} ===")
    return 0
