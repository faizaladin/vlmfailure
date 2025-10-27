#!/usr/bin/env python3
"""
Shuffle per-video frame folders and reorder the combined CSV to match.

Default behavior: create a new folder with symlinks to the original frame
folders and write a new CSV where rows are ordered to match the shuffled
folders. Adds columns `order_index` and `frames_path` to the CSV.

Usage example:
  python3 shuffle_frames.py \
    --frames-dir "/path/to/combined/frames" \
    --csv "/path/to/combined/combined.csv" \
    --out-frames "/path/to/combined/shuffled_frames" \
    --out-csv "/path/to/combined/combined_shuffled.csv" \
    --seed 123
"""
import argparse
import csv
import logging
import random
from pathlib import Path
import shutil
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames-dir", type=Path, required=True, help="directory containing per-video frame folders")
    p.add_argument("--csv", type=Path, required=True, help="combined CSV to align")
    p.add_argument("--out-frames", type=Path, required=True, help="output directory for shuffled frame folders (symlinks by default)")
    p.add_argument("--out-csv", type=Path, required=True, help="output CSV path (reordered)")
    p.add_argument("--seed", type=int, default=42, help="random seed for reproducible shuffle")
    p.add_argument("--copy", action="store_true", help="copy folders instead of creating symlinks (can be large)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    frames_dir = args.frames_dir
    csv_path = args.csv
    out_frames = args.out_frames
    out_csv = args.out_csv

    if not frames_dir.exists() or not frames_dir.is_dir():
        raise SystemExit(f"frames dir not found: {frames_dir}")
    if not csv_path.exists():
        raise SystemExit(f"csv file not found: {csv_path}")

    # load CSV rows into dict keyed by folder stem (assumes new_filename exists and matches folder names)
    rows_by_stem = {}
    rows = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
            # prefer new_filename column, fallback to filename
            fname = r.get("new_filename") or r.get("filename")
            if not fname:
                continue
            stem = Path(fname).stem
            rows_by_stem.setdefault(stem, []).append(r)

    # list frame folders (immediate subdirs)
    folders = [d for d in sorted(frames_dir.iterdir()) if d.is_dir()]
    folder_names = [d.name for d in folders]
    logging.info(f"Found {len(folder_names)} frame folders in {frames_dir}")

    # build list of folder names that have CSV rows
    matched = [name for name in folder_names if name in rows_by_stem]
    unmatched = [name for name in folder_names if name not in rows_by_stem]
    if unmatched:
        logging.warning(f"{len(unmatched)} folders have no matching CSV row; they will be placed at the end: {unmatched[:5]}")

    # shuffle matched folders
    random.seed(args.seed)
    random.shuffle(matched)

    # final order: matched shuffled first, then unmatched
    final_order = matched + unmatched

    # prepare output frames dir
    if out_frames.exists():
        logging.info(f"Removing existing out-frames folder: {out_frames}")
        shutil.rmtree(out_frames)
    out_frames.mkdir(parents=True)

    new_rows = []
    order_idx = 0
    for name in final_order:
        src = frames_dir / name
        if not src.exists():
            logging.warning(f"Source folder missing (skipping): {src}")
            continue

        order_idx += 1
        prefix = f"{order_idx:06d}_"
        dest_name = prefix + name
        dest = out_frames / dest_name

        if args.copy:
            shutil.copytree(src, dest)
        else:
            # create symlink
            try:
                os.symlink(src.resolve(), dest)
            except FileExistsError:
                dest.unlink()
                os.symlink(src.resolve(), dest)

        # attach CSV rows matching this folder (there may be multiple rows for same stem)
        stem = name
        matched_rows = rows_by_stem.get(stem, [])
        if matched_rows:
            for r in matched_rows:
                new_r = dict(r)
                new_r["order_index"] = str(order_idx)
                new_r["frames_path"] = str(dest)
                new_rows.append(new_r)
        else:
            # folder didn't match any csv row: create a minimal row
            new_r = {k: "" for k in (rows[0].keys() if rows else ["filename"]) }
            new_r["order_index"] = str(order_idx)
            new_r["frames_path"] = str(dest)
            # set an identifier
            new_r.setdefault("filename", "")
            new_rows.append(new_r)

    # write new CSV with extra columns
    if new_rows:
        # preserve original fieldnames and append new ones
        base_fields = list(new_rows[0].keys())
        # ensure consistent ordering: original csv fieldnames first
        with csv_path.open("r", newline="") as f:
            orig_fields = list(csv.DictReader(f).fieldnames or [])
        # avoid duplicates
        out_fields = orig_fields + [c for c in ["order_index", "frames_path"] if c not in orig_fields]

        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=out_fields)
            writer.writeheader()
            for r in new_rows:
                # ensure all keys present
                row = {k: r.get(k, "") for k in out_fields}
                writer.writerow(row)
    else:
        logging.warning("No rows to write to out CSV")

    logging.info(f"Wrote {len(new_rows)} rows to {out_csv}")
    logging.info(f"Shuffled frames (symlinks or copies) placed in {out_frames}")


if __name__ == "__main__":
    main()
