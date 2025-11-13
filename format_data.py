# Script to extract first 5 seconds of each trajectory video in Combined Data,
# save to vlm_data/, and create a metadata JSON with labels and CSV info.

import os
import json
import shutil
import subprocess
import csv

COMBINED_DATA_DIR = 'Combined Data'
OUTPUT_DIR = 'vlm_data'
METADATA_JSON = os.path.join(OUTPUT_DIR, 'metadata.json')

def ensure_dir(path):
	if not os.path.exists(path):
		os.makedirs(path)

def extract_first_5_seconds(input_path, output_path):
	# Use ffmpeg to extract first 5 seconds
	cmd = [
		'ffmpeg', '-y', '-i', input_path, '-t', '5', '-c', 'copy', output_path
	]
	subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def get_label_from_folder(folder_name):
	if folder_name == 'success':
		return 'success'
	elif folder_name in ['lane_violation', 'collision']:
		return 'failure'
	else:
		return None

def read_run_log_csv(csv_path):
	# Returns a dict: {trajectory_name: label}
	if not os.path.exists(csv_path):
		return {}
	result = {}
	with open(csv_path, 'r') as f:
		reader = csv.DictReader(f)
		for row in reader:
			# Assume trajectory name is in a column 'trajectory' or similar
			# and label in 'label' or similar; adjust as needed
			traj = row.get('trajectory') or row.get('traj') or row.get('name')
			label = row.get('label') or row.get('result')
			if traj and label:
				result[traj] = label
	return result

def main():
	ensure_dir(OUTPUT_DIR)
	metadata = []

	for town in os.listdir(COMBINED_DATA_DIR):
		town_path = os.path.join(COMBINED_DATA_DIR, town)
		if not os.path.isdir(town_path):
			continue
		for weather in os.listdir(town_path):
			weather_path = os.path.join(town_path, weather)
			if not os.path.isdir(weather_path):
				continue
			# Read run_log.csv for this weather
			run_log_csv = os.path.join(weather_path, 'run_log.csv')
			run_log = read_run_log_csv(run_log_csv)
			for label_folder in ['success', 'lane_violation', 'collision']:
				label_path = os.path.join(weather_path, label_folder)
				if not os.path.isdir(label_path):
					continue
				label = get_label_from_folder(label_folder)
				for video_file in os.listdir(label_path):
					if not video_file.endswith('.mp4'):
						continue
					input_video = os.path.join(label_path, video_file)
					# Output path: vlm_data/town_weather_label_videoname.mp4
					out_name = f"{town}_{weather}_{label_folder}_{video_file}"
					output_video = os.path.join(OUTPUT_DIR, out_name)
					extract_first_5_seconds(input_video, output_video)
					# Get trajectory label from run_log if possible
					traj_name = os.path.splitext(video_file)[0]
					csv_label = run_log.get(traj_name, None)
					metadata.append({
						'video': output_video,
						'label': label,
						'csv_label': csv_label,
						'town': town,
						'weather': weather,
						'source_folder': label_folder,
						'original_video': input_video
					})

	# Save metadata
	with open(METADATA_JSON, 'w') as f:
		json.dump(metadata, f, indent=2)

if __name__ == '__main__':
	main()
