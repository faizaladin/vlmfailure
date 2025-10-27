#!/usr/bin/env python3
"""
Combine collision videos from all subfolders and randomly sample the same number
of success and lane_violation videos. Copies selected videos into a single
folder and writes a combined CSV with metadata.

Usage: python combine.py --root /path/to/VLM_Data --out combined
"""
import argparse
import csv
import logging
import random
import shutil
from pathlib import Path
from typing import Dict, List


def find_run_logs(root: Path) -> List[Path]:
	return list(root.rglob("run_log.csv"))


def parse_run_log(run_log_path: Path) -> List[Dict]:
	rows = []
	with run_log_path.open("r", newline="") as f:
		reader = csv.DictReader(f)
		for r in reader:
			rows.append(r)
	return rows


def collect_videos(root: Path):
	"""Traverse run_log files and collect available videos for each label.

	Returns dict with keys 'collision', 'success', 'lane violation' mapping to
	lists of metadata dicts.
	"""
	groups = {"collision": [], "success": [], "lane violation": []}

	for rl in find_run_logs(root):
		# parent folder like town2_rainy
		parent = rl.parent
		parent_name = parent.name  # expecting e.g. town2_rainy
		parts = parent_name.split("_")
		if len(parts) >= 2:
			town = parts[0]
			weather = parts[1]
		else:
			town = parent_name
			weather = ""

		rows = parse_run_log(rl)
		for r in rows:
			run_id_raw = r.get("run_id", "").strip()
			if run_id_raw == "":
				continue
			try:
				run_id = int(run_id_raw)
			except Exception:
				# keep as string fallback
				run_id = run_id_raw

			violation = (r.get("violation") or "").strip()
			label = "" if violation == "" else violation

			# determine expected file location for each label
			# video filenames are run_<id>.mp4 in subfolders named collision, success, lane_violation
			for folder_label, dest_label in [("collision", "collision"), ("success", "success"), ("lane_violation", "lane violation")]:
				video_path = parent / folder_label / f"run_{run_id}.mp4"
				if video_path.exists():
					meta = {
						"run_log": str(rl),
						"source_path": str(video_path),
						"filename": video_path.name,
						"label": dest_label,
						"run_id": str(run_id),
						"violation": violation,
						"violation_time": r.get("violation_time", ""),
						"collision_object": r.get("collision_object", ""),
						"town": town,
						"weather": weather,
					}
					groups[dest_label].append(meta)
					break

	return groups


def sample_and_copy(groups, out_dir: Path, seed: int = None):
	random.seed(seed)
	collisions = groups.get("collision", [])
	n = len(collisions)
	logging.info(f"Found {n} collision videos")

	successes = groups.get("success", [])
	lane_violations = groups.get("lane violation", [])

	if n == 0:
		logging.warning("No collision videos found; exiting")
		return []

	if len(successes) < n:
		logging.warning(f"Only {len(successes)} success videos available; sampling without replacement will use all of them")
	if len(lane_violations) < n:
		logging.warning(f"Only {len(lane_violations)} lane violation videos available; sampling without replacement will use all of them")

	sampled_success = random.sample(successes, min(n, len(successes))) if successes else []
	sampled_lane = random.sample(lane_violations, min(n, len(lane_violations))) if lane_violations else []

	out_dir.mkdir(parents=True, exist_ok=True)

	combined_entries = []

	def copy_item(meta):
		src = Path(meta["source_path"])
		# create new filename to avoid collisions: town_weather_run_XX.mp4
		prefix = f"{meta.get('town','')}_{meta.get('weather','')}"
		new_name = f"{prefix}_{src.name}" if prefix.strip("_") else src.name
		dest = out_dir / new_name
		shutil.copy2(src, dest)
		new_meta = dict(meta)
		new_meta["copied_to"] = str(dest)
		new_meta["new_filename"] = new_name
		return new_meta

	# copy collisions (all)
	for c in collisions:
		try:
			combined_entries.append(copy_item(c))
		except Exception as e:
			logging.exception(f"Failed copying {c.get('source_path')}: {e}")

	for s in sampled_success:
		try:
			combined_entries.append(copy_item(s))
		except Exception as e:
			logging.exception(f"Failed copying {s.get('source_path')}: {e}")

	for l in sampled_lane:
		try:
			combined_entries.append(copy_item(l))
		except Exception as e:
			logging.exception(f"Failed copying {l.get('source_path')}: {e}")

	return combined_entries


def extract_frames_for_entries(entries: List[Dict], out_dir: Path, ffmpeg_cmd: str = "ffmpeg") -> None:
	"""For each entry, create a folder named after the video (without extension)
	and extract frames into it using ffmpeg.

	Creates folders: out_dir/<new_filename_without_ext>/frame_000001.jpg ...
	"""
	import subprocess

	out_dir.mkdir(parents=True, exist_ok=True)

	for e in entries:
		video_path = Path(e.get("copied_to") or e.get("source_path"))
		if not video_path.exists():
			logging.warning(f"Video file not found for frame extraction: {video_path}")
			continue

		name_no_ext = Path(e.get("new_filename", video_path.name)).stem
		frames_folder = out_dir / name_no_ext
		frames_folder.mkdir(parents=True, exist_ok=True)

		# ffmpeg pattern
		pattern = str(frames_folder / "frame_%06d.jpg")

		cmd = [ffmpeg_cmd, "-y", "-i", str(video_path), pattern]
		try:
			# run ffmpeg and capture output for logging if it fails
			subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
		except Exception as exc:
			logging.exception(f"Failed extracting frames with command: {' '.join(cmd)} -> {exc}")


def write_csv(entries: List[Dict], out_csv: Path):
	if not entries:
		logging.warning("No entries to write to CSV")
		return
	# determine columns
	columns = [
		"new_filename",
		"filename",
		"label",
		"run_id",
		"town",
		"weather",
		"violation",
		"violation_time",
		"collision_object",
		"run_log",
		"source_path",
		"copied_to",
	]
	with out_csv.open("w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=columns)
		writer.writeheader()
		for e in entries:
			row = {k: e.get(k, "") for k in columns}
			writer.writerow(row)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--root", type=Path, default=Path.cwd(), help="root folder containing town folders")
	parser.add_argument("--out", type=Path, default=Path("combined"), help="output folder for combined videos and csv")
	parser.add_argument("--shuffle", action="store_true", help="shuffle the combined entries before writing CSV and extracting frames")
	parser.add_argument("--no-shuffle", dest="shuffle", action="store_false", help="do not shuffle (keep default ordering)")
	parser.set_defaults(shuffle=True)
	parser.add_argument("--frames", action="store_true", help="extract frames for every copied video into per-video folders using ffmpeg")
	parser.add_argument("--ffmpeg", type=str, default="ffmpeg", help="ffmpeg command/binary to use for frame extraction")
	parser.add_argument("--seed", type=int, default=42, help="random seed for sampling")
	args = parser.parse_args()

	logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

	root = args.root
	out = args.out
	out_videos = out / "videos"
	out_csv = out / "combined.csv"

	logging.info(f"Scanning root: {root}")
	groups = collect_videos(root)
	entries = sample_and_copy(groups, out_videos, seed=args.seed)

	# shuffle entries if requested
	if args.shuffle:
		random.seed(args.seed)
		random.shuffle(entries)

	write_csv(entries, out_csv)

	# optionally extract frames per video
	if args.frames:
		frames_out = out / "frames"
		logging.info(f"Extracting frames to {frames_out} using ffmpeg: {args.ffmpeg}")
		extract_frames_for_entries(entries, frames_out, ffmpeg_cmd=args.ffmpeg)

	logging.info(f"Done. Wrote {len(entries)} rows and copied videos to {out_videos}")


if __name__ == "__main__":
	main()

