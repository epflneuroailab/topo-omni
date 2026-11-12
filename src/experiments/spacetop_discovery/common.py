"""Shared paths and IO for the SpaceTop topographic-discovery pipeline.

Discovery has two halves that used to live in two different trees.

* **Upstream — find the clusters.** ``extract_video_embeddings`` ->
  ``compile_embeddings`` -> ``agglomerative_early_stopping`` ->
  ``plot_dendrogram`` -> (``make_cluster_collages`` to label by eye) ->
  ``merge_clusters`` -> ``create_cluster_manifest``, plus
  ``calculate_selectivity`` / ``plot_model_selectivity`` for the per-cluster
  selectivity scores. This half lived in ``topo-discover/`` at the repo root.
* **Downstream — put the clusters on the sheet.** ``extract`` -> ``contrast``
  -> ``plot_selectivity`` -> ``correlate_t_values`` / ``run_clusters_iou``.

They are now one package. The upstream scripts used to hardcode paths relative
to ``topo-discover/`` (``task-alignvideo/clustering_v2/...``), which only
resolved when run with that directory as the cwd; every path now resolves
through this module instead, so the scripts run from the repo root like every
other experiment (``python -m src.experiments.spacetop_discovery.<script>``).
"""
import json
import os
import pickle as pkl

from dotenv import load_dotenv

load_dotenv()

SAVE_DIR = os.getenv("SAVE_DIR", "results")
CKPT_DIR = os.getenv("CKPT_DIR")
STIMULI_DIR = os.getenv("STIMULI_DIR", "stimuli")

# Root of the SpaceTop movie stimuli. Expected layout:
#   <SPACETOP_DIR>/task-alignvideo/*.mp4           the 52 full movie runs
#   <SPACETOP_DIR>/spacetop_embeddings_v2/*.mp4    the 2s clips cut from them,
#                                                  + *_nemotron_embeddings.npz
# Set SPACETOP_DIR in .env; the clips are large and are not kept in the repo.
SPACETOP_DIR = os.getenv("SPACETOP_DIR", os.path.join(STIMULI_DIR, "spacetop"))

CLIPS_SUBDIR = "spacetop_embeddings_v2"
VIDEOS_SUBDIR = "task-alignvideo"


def discovery_dir(model_name: str, tag: str = None) -> str:
    """Where one discovery run writes everything: ``$SAVE_DIR/<model>/spacetop_discovery``.

    Keyed on the model because the clustering is *guided by that model's cortical
    sheets* -- two checkpoints do not share a cluster tree even though they share
    the clip embeddings that seed it. ``tag`` suffixes the directory so two runs of
    the same checkpoint over different clip sets (say prefix vs disjoint clips) do
    not overwrite each other's sheets, tree and scores.
    """
    name = "spacetop_discovery" if not tag else f"spacetop_discovery_{tag}"
    return os.path.join(SAVE_DIR, model_name, name)


def clips_dir(root: str = None) -> str:
    return os.path.join(root or SPACETOP_DIR, CLIPS_SUBDIR)


def videos_dir(root: str = None) -> str:
    return os.path.join(root or SPACETOP_DIR, VIDEOS_SUBDIR)


def resolve_clip(clip_id: str, root: str = None) -> str:
    """Absolute path of a clip as named in ``clips_manifest.json``.

    The manifest stores ``spacetop_embeddings_v2/<clip>.mp4`` -- relative to the
    old working directory. Absolute entries are passed through untouched.
    """
    if os.path.isabs(clip_id):
        return clip_id
    return os.path.join(root or SPACETOP_DIR, clip_id)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data, path, indent=4):
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def read_pickle(path):
    with open(path, "rb") as f:
        return pkl.load(f)


def write_pickle(data, path):
    ensure_parent(path)
    with open(path, "wb") as f:
        pkl.dump(data, f)


def ensure_parent(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def add_run_args(parser, default_model=None):
    """``--model_name`` / ``--spacetop_dir`` / ``--output_dir``, shared by every step.

    ``--output_dir`` defaults to ``discovery_dir(model_name)``; resolve it with
    :func:`resolve_run_args` after parsing.
    """
    from src.core.model_loading import MODEL_TITLE
    parser.add_argument("--model_name", type=str, default=default_model or MODEL_TITLE,
                        help="run title under $SAVE_DIR. Names the output directory; "
                             "pass the same value as --checkpoint's run so a run's "
                             "sheets, tree and scores stay together.")
    parser.add_argument("--spacetop_dir", type=str, default=None,
                        help="root holding task-alignvideo/ and spacetop_embeddings_v2/. "
                             "Default: $SPACETOP_DIR.")
    parser.add_argument("--run_tag", type=str, default=None,
                        help="Suffixes the output directory, e.g. 'disjoint2s'. Use it to "
                             "keep runs of one checkpoint over different clip sets apart.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Default: $SAVE_DIR/<model_name>/spacetop_discovery[_<run_tag>].")
    return parser


def resolve_run_args(args):
    """Fill in the defaults ``add_run_args`` deferred, and make the output dir."""
    args.spacetop_dir = args.spacetop_dir or SPACETOP_DIR
    args.output_dir = args.output_dir or discovery_dir(args.model_name,
                                                       getattr(args, "run_tag", None))
    os.makedirs(args.output_dir, exist_ok=True)
    return args
