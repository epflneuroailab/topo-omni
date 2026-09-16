"""Turn a directory of images into a manifest `load_and_format_dataset` can read.

Emitted rows match the video manifest's schema so the two concatenate cleanly:

    {"id": ..., "type": "image", "path": "/abs/path.jpg",
     "prompt": ..., "model_response": ...}

`type: "image"` is what routes a row to the image batches; see src/training/batching.py for why
image and video rows can never share a batch.

WHY THE TARGET TEXT IS THIN. Only the cortical adaptors train and alpha is 20, so the
spatial loss dominates and the caption mostly sets input statistics. A class name is
enough.

But a thin target makes the task loss HARDER, not easier -- cross-entropy is a per-token
mean, and these targets are ~7-8 tokens of which "A photograph of a" is a free template
and the class name is near-pure 1000-way surprise. A Koala caption is ~37 tokens of
mostly predictable prose. Measured on the live strip-4096 run:

    image steps   task_loss 2.64 - 2.86   (median 2.75)
    video steps   task_loss 0.38 - 0.77   (median 0.53)

with zero overlap between the two populations. Two consequences:

  1. The task_loss trace is BIMODAL. Smoothing or averaging it across steps is
     meaningless; read it split by modality (sheet_T == image-batch-size marks an
     image step).
  2. The task term is a bigger share of the loss on image steps -- ~27% against the
     post-alpha spatial loss of ~10, versus ~5% on video steps. So image steps push the
     adaptors with roughly 5x more task gradient. If that asymmetry matters for a given
     run, give the images richer captions rather than reweighting: a longer target
     lowers the per-token mean directly.

STAGING THE DATASETS (neither is downloaded by this script):

  DTD -- 5,640 texture images, ~600 MB. A third of the 47 categories are explicitly
  oriented (banded, striped, lined, grid, zigzagged), and a texture is globally coherent,
  so its orientation content survives the whole-image pooling an image goes through.
      wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
      tar xf dtd-r1.0.1.tar.gz
      python -m src.tools.build_image_manifest --images-dir dtd/images \
          --labels-from-dirname --out data/dtd_images.json

  ImageNet-1k -- 1.28M images, the TDANN precedent (their V1 orientation maps come from
  SimCLR on ImageNet). Gated on the Hub; $HF_TOKEN covers it.
      huggingface-cli download ILSVRC/imagenet-1k --repo-type dataset --local-dir <dir>
      python -m src.tools.build_image_manifest --images-dir <dir>/train \
          --labels-from-dirname --synset-map <map.json> --out data/imagenet.json
"""
import argparse
import collections
import json
import os
from pathlib import Path

EXTS = (".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="What is shown in this image?")
    ap.add_argument("--captions", default=None,
                    help="JSON mapping basename (with or without extension) -> caption. "
                         "Wins over --labels-from-dirname where both give an answer.")
    ap.add_argument("--labels-from-dirname", action="store_true",
                    help="use the parent directory as the class name (ImageNet / DTD "
                         "layout) and answer 'A photograph of a {class}.'")
    ap.add_argument("--synset-map", default=None,
                    help="JSON mapping a raw directory name (e.g. an n0######## synset) "
                         "to a readable class name")
    ap.add_argument("--limit", type=int, default=None,
                    help="keep only N images, for a quick smoke run")
    ap.add_argument("--per-class", type=int, default=None,
                    help="cap images per class directory. Prefer this over --limit for "
                         "subsetting a class-structured set: --limit takes a flat slice, "
                         "which leaves the class counts uneven.")
    ap.add_argument("--shuffle-seed", type=int, default=None,
                    help="shuffle before --limit. Without it --limit takes the first N of "
                         "a SORTED list, which under a class-per-directory layout is one "
                         "class only -- e.g. 256 DTD images that are all 'banded'.")
    ap.add_argument("--default-response", default="A photograph.")
    ap.add_argument("--response-template", default="A photograph of a {cls}.",
                    help="used with --labels-from-dirname. DTD categories are adjectives, "
                         "so pass 'A photograph of a {cls} texture.' there rather than "
                         "emitting 'A photograph of a banded.'")
    args = ap.parse_args()

    caps = json.load(open(args.captions)) if args.captions else {}
    # accept caption keys with or without the extension
    caps = {**caps, **{Path(k).stem: v for k, v in caps.items()}}
    syn = json.load(open(args.synset_map)) if args.synset_map else {}

    root = Path(args.images_dir)
    paths = sorted(p for p in root.rglob("*") if p.suffix in EXTS)
    if args.shuffle_seed is not None:
        import random
        random.Random(args.shuffle_seed).shuffle(paths)
    if args.per_class:
        seen, kept = {}, []
        for q in paths:
            c = seen.get(q.parent.name, 0)
            if c < args.per_class:
                seen[q.parent.name] = c + 1
                kept.append(q)
        paths = kept
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise SystemExit(f"no images under {root}")

    rows = []
    for p in paths:
        if p.stem in caps:
            response = caps[p.stem]
        elif args.labels_from_dirname:
            cls = syn.get(p.parent.name, p.parent.name).replace("_", " ")
            response = args.response_template.format(cls=cls)
        else:
            response = args.default_response
        rows.append({
            "id": p.stem,
            "type": "image",
            "path": str(p.resolve()),
            "prompt": args.prompt,
            "model_response": response,
        })

    os.makedirs(Path(args.out).parent, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rows, f)
    n_cls = len({r["model_response"] for r in rows})
    per = collections.Counter(Path(r["path"]).parent.name for r in rows)
    lo, hi = (min(per.values()), max(per.values())) if per else (0, 0)
    print(f"> wrote {len(rows):,} image rows to {args.out} ({n_cls:,} distinct responses, "
          f"{len(per):,} class dirs, {lo}-{hi} per class)")


if __name__ == "__main__":
    main()
