"""Stage an image dataset as square JPEGs, ready for `build_image_manifest`.

WHY SQUARE, AND WHY CROP RATHER THAN SQUASH. The training pipeline forces every image
to resized_height == resized_width (see load_and_format_dataset) because the thinker's
no-audio pooling collapses T unless every image yields the same visual token count. That
resize does NOT preserve aspect ratio -- it squashes. Squashing a 500x375 photo to a
square stretches it vertically by 1.33x, which rotates every oriented edge it contains
and skews the orientation statistics of the whole dataset. For a run whose entire readout
is an orientation map, that is a confound, not a detail.

So the images are CENTER-CROPPED to the largest centred square here, then resized. The
pipeline's own resize then finds them already square and is a no-op in aspect terms.

Two sources:
  --hf-repo   an image/label parquet dataset on the Hub (ImageNet-1k)
  --images-dir  a local directory tree (DTD)

Output layout is <out>/<class>/<id>.jpg, which is what
`build_image_manifest --labels-from-dirname` expects.

    python -m src.tools.prepare_images --hf-repo ILSVRC/imagenet-1k \
        --shards 32 --square 224 --out ~/datasets/imagenet-224
    python -m src.tools.prepare_images --images-dir ~/datasets/dtd/dtd/images \
        --square 224 --out ~/datasets/dtd-224
"""
import argparse
import io
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

EXTS = (".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG")


def slug(name: str) -> str:
    """'sports car, sport car' -> 'sports_car'. First synonym only; ImageNet class names
    are comma-separated synonym lists and the full string makes an unwieldy directory."""
    first = name.split(",")[0].strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", first).strip("_") or "class"


def square(img: Image.Image, side: int) -> Image.Image:
    """Largest centred square, then resize. Never changes the aspect ratio of content."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return img.resize((side, side), Image.BICUBIC)


def _write(args):
    payload, dest, side = args
    try:
        img = Image.open(io.BytesIO(payload) if isinstance(payload, bytes) else payload)
        square(img, side).save(dest, "JPEG", quality=92)
        return 1
    except Exception:
        return 0


def from_hf(repo, n_shards, side, out, token, per_class=None):
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=token)
    files = sorted(f for f in api.list_repo_files(repo, repo_type="dataset")
                   if "train" in f and f.endswith(".parquet"))
    if not files:
        raise SystemExit(f"no train parquet shards in {repo}")
    # Spread the picks across the whole shard list. ILSVRC's shards happen to be shuffled
    # across classes, but a class-ORDERED release (as some mirrors are) would hand back
    # only the first ~25 classes if we took a contiguous prefix.
    step = max(1, len(files) // n_shards)
    picks = files[::step][:n_shards]
    print(f"> {repo}: {len(files)} shards, taking {len(picks)} (every {step})")

    names = None
    try:
        from datasets import load_dataset_builder
        names = load_dataset_builder(repo, token=token).info.features["label"].names
    except Exception as e:
        print(f"  (no class names available: {e}); using numeric labels")

    counts, total = {}, 0
    with ProcessPoolExecutor(max_workers=16) as pool:
        for si, shard in enumerate(picks):
            p = hf_hub_download(repo, shard, repo_type="dataset", token=token)
            pf = pq.ParquetFile(p)
            jobs = []
            for rg in range(pf.metadata.num_row_groups):
                t = pf.read_row_groups([rg])
                imgs, labs = t.column("image").to_pylist(), t.column("label").to_pylist()
                for rec, lab in zip(imgs, labs):
                    cls = slug(names[lab]) if names else f"class_{lab:04d}"
                    if per_class and counts.get(cls, 0) >= per_class:
                        continue
                    d = Path(out) / cls
                    d.mkdir(parents=True, exist_ok=True)
                    counts[cls] = counts.get(cls, 0) + 1
                    jobs.append((rec["bytes"], d / f"{cls}_{counts[cls]:05d}.jpg", side))
            total += sum(pool.map(_write, jobs, chunksize=64))
            os.remove(p)   # keep peak disk at one shard
            print(f"  [{si+1}/{len(picks)}] {shard} -> {total:,} images, "
                  f"{len(counts)} classes", flush=True)
    return total, len(counts)


def from_dir(images_dir, side, out):
    root = Path(images_dir)
    paths = sorted(p for p in root.rglob("*") if p.suffix in EXTS)
    print(f"> {root}: {len(paths):,} images")
    jobs = []
    for p in paths:
        d = Path(out) / p.parent.name
        d.mkdir(parents=True, exist_ok=True)
        jobs.append((str(p), d / f"{p.stem}.jpg", side))
    with ProcessPoolExecutor(max_workers=16) as pool:
        total = sum(pool.map(_write, jobs, chunksize=32))
    return total, len({p.parent.name for p in paths})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--hf-repo")
    src.add_argument("--images-dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--square", type=int, default=224,
                    help="output side. 224 gives 256 patches / 64 tokens, exactly what "
                         "the orientation eval's sine gratings produce.")
    ap.add_argument("--shards", type=int, default=32,
                    help="--hf-repo only: how many parquet shards to pull. ILSVRC shards "
                         "hold ~4,358 images each, so 32 gives ~139k.")
    ap.add_argument("--per-class", type=int, default=None,
                    help="--hf-repo only: cap images per class")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.hf_repo:
        n, k = from_hf(args.hf_repo, args.shards, args.square, args.out,
                       os.environ.get("HF_TOKEN"), args.per_class)
    else:
        n, k = from_dir(args.images_dir, args.square, args.out)
    print(f"\n> wrote {n:,} images in {k:,} class directories to {args.out} "
          f"at {args.square}x{args.square}")


if __name__ == "__main__":
    main()
