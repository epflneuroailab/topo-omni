#!/usr/bin/env python3
"""
Connected-components tooling for a binary 2D mask.

Requires:
  pip install numpy scipy

Input/Output:
  Expects a 2D .npy mask (0/1 or bool). Writes a 0/1 uint8 .npy.

Common usage:
  # List islands:
  python islands.py --in mask.npy --list

  # Keep largest (default behavior if you don't specify keep/remove options):
  python islands.py --in mask.npy --out kept.npy

  # Keep a specific island id:
  python islands.py --in mask.npy --out kept.npy --keep-id 3

  # Remove specific island ids (leave others intact):
  python islands.py --in mask.npy --out edited.npy --remove-ids 2 5 7

  # Keep everything except the island containing a seed pixel:
  python islands.py --in mask.npy --out edited.npy --remove-seed 120 55
"""

import argparse
import numpy as np
from scipy.ndimage import label


def _structure(connectivity: int) -> np.ndarray:
    if connectivity == 4:
        return np.array([[0, 1, 0],
                         [1, 1, 1],
                         [0, 1, 0]], dtype=bool)
    if connectivity == 8:
        return np.ones((3, 3), dtype=bool)
    raise ValueError("connectivity must be 4 or 8")


def label_islands(mask: np.ndarray, connectivity: int = 8):
    mask_bool = mask.astype(bool)
    labeled, num = label(mask_bool, structure=_structure(connectivity))
    return labeled, num


def island_stats(labeled: np.ndarray, num: int, t_values: np.ndarray):
    """
    Returns a list of dicts with: id, size, bbox, centroid
    bbox = (rmin, rmax, cmin, cmax) inclusive
    centroid = (r_mean, c_mean)
    """
    stats = []
    for i in range(1, num + 1):
        coords = np.argwhere(labeled == i)
        if coords.size == 0:
            continue
        rmin, cmin = coords.min(axis=0)
        rmax, cmax = coords.max(axis=0)
        r_mean, c_mean = coords.mean(axis=0)

        average_t_values = t_values[labeled==i].mean()

        stats.append({
            "id": i,
            "size": int(coords.shape[0]),
            "bbox": (int(rmin), int(rmax), int(cmin), int(cmax)),
            "centroid": (float(r_mean), float(c_mean)),
            "average_t_values": average_t_values,
        })
    stats.sort(key=lambda d: d["size"], reverse=True)
    return stats


def keep_only_id(labeled: np.ndarray, keep_id: int) -> np.ndarray:
    return (labeled == keep_id).astype(np.uint8)


def remove_ids(mask: np.ndarray, labeled: np.ndarray, ids_to_remove: list[int]) -> np.ndarray:
    """
    Removes selected islands, leaving everything else as-is.
    """
    out = mask.astype(bool).copy()
    for rid in ids_to_remove:
        out[labeled == rid] = False
    return out.astype(np.uint8)


def pick_largest_id(labeled: np.ndarray) -> int:
    counts = np.bincount(labeled.ravel())
    if len(counts) <= 1:
        return 0
    counts[0] = 0
    return int(counts.argmax())


def id_at_seed(labeled: np.ndarray, seed: tuple[int, int]) -> int:
    r, c = seed
    if not (0 <= r < labeled.shape[0] and 0 <= c < labeled.shape[1]):
        raise ValueError(f"Seed {seed} out of bounds for shape {labeled.shape}")
    return int(labeled[r, c])


def print_stats(stats: list[dict], max_rows: int = 50):
    if not stats:
        print("No islands found.")
        return
    print(f"Found {len(stats)} islands. (Sorted by size desc)")
    print("id\tpixels\tbbox(rmin,rmax,cmin,cmax)\tcentroid(r,c)\tavg_t_vals")
    for d in stats[:max_rows]:
        cid = d["id"]
        sz = d["size"]
        bbox = d["bbox"]
        cent = d["centroid"]
        avg_t_values = d["average_t_values"]
        print(f"{cid}\t{sz}\t{bbox}\t({cent[0]:.2f},{cent[1]:.2f})\t{avg_t_values:.4f}")
    if len(stats) > max_rows:
        print(f"... ({len(stats) - max_rows} more not shown; increase --max-list-rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input mask as .npy (2D)")
    ap.add_argument("--out", dest="out_path", default=None, help="Output .npy path")
    ap.add_argument("--connectivity", type=int, default=8, choices=[4, 8])

    ap.add_argument("--list", action="store_true", help="Print island ids and stats, then exit")
    ap.add_argument("--max-list-rows", type=int, default=50)

    # Keep options
    ap.add_argument("--keep-id", type=int, default=None, help="Keep ONLY this island id")
    ap.add_argument("--keep-seed", nargs=2, type=int, default=None, metavar=("ROW", "COL"),
                    help="Keep ONLY the island containing (ROW, COL)")

    # Remove options
    ap.add_argument("--remove-ids", nargs="+", type=int, default=None,
                    help="Remove these island ids (leave other islands intact)")
    ap.add_argument("--remove-seed", nargs=2, type=int, default=None, metavar=("ROW", "COL"),
                    help="Remove the island containing (ROW, COL)")

    args = ap.parse_args()

    mask = np.load(args.in_path)
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")

    labeled, num = label_islands(mask, connectivity=args.connectivity)
    stats = island_stats(labeled, num)

    if args.list:
        print_stats(stats, max_rows=args.max_list_rows)
        return

    if args.out_path is None:
        raise ValueError("Please provide --out unless using --list")

    # Decide operation priority:
    # 1) keep-only (keep-id / keep-seed)
    # 2) remove (remove-ids / remove-seed)
    # 3) default: keep largest only

    if args.keep_id is not None or args.keep_seed is not None:
        if args.keep_seed is not None:
            keep_id = id_at_seed(labeled, tuple(args.keep_seed))
            if keep_id == 0:
                raise ValueError(f"--keep-seed {tuple(args.keep_seed)} is not inside any island.")
        else:
            keep_id = int(args.keep_id)
            if keep_id < 1 or keep_id > num:
                raise ValueError(f"--keep-id must be in [1, {num}] (got {keep_id}).")
        out = keep_only_id(labeled, keep_id)

    elif args.remove_ids is not None or args.remove_seed is not None:
        ids = []
        if args.remove_ids is not None:
            ids.extend([int(x) for x in args.remove_ids])
        if args.remove_seed is not None:
            rid = id_at_seed(labeled, tuple(args.remove_seed))
            if rid == 0:
                raise ValueError(f"--remove-seed {tuple(args.remove_seed)} is not inside any island.")
            ids.append(rid)

        # Validate ids
        bad = [i for i in ids if i < 1 or i > num]
        if bad:
            raise ValueError(f"Invalid island ids {bad}; valid range is [1, {num}]")

        # Remove them
        out = remove_ids(mask, labeled, ids)

    else:
        # default: keep largest only
        keep_id = pick_largest_id(labeled)
        if keep_id == 0:
            out = np.zeros_like(mask, dtype=np.uint8)
        else:
            out = keep_only_id(labeled, keep_id)

    np.save(args.out_path, out.astype(np.uint8))
    kept_pixels = int(out.sum())
    print(f"Saved: {args.out_path} | islands found: {num} | kept pixels: {kept_pixels}")


if __name__ == "__main__":
    main()
