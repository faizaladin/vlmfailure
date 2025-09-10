
import os
import re
import json

frame_dir = "paired_frames"
failure_frames_file = "failure_points_by_trajectory.txt"
success_traj_logs = "success_traj_logs"
failure_traj_logs = "failure_traj_logs"


# Parse failure frames as (folder, frame) pairs, using the actual folder name
failure_set = set()
with open(failure_frames_file, "r") as f:
    for line in f:
        # Example line: y=-0.2000000000000003, heading=25, failure_frames=[59]
        match = re.match(r"y=([^,]+), heading=(\d+), failure_frames=\[([^\]]*)\]", line.strip())
        if match:
            y_val, heading, frames_str = match.groups()
            folder = f"{y_val}_{heading}"
            frame_nums = [s for s in re.findall(r"\d+", frames_str)]
            for frame in frame_nums:
                failure_set.add((folder, str(int(frame))))


# Recursively find all images in all subfolders
image_paths = []
for subdir in os.listdir(frame_dir):
    subdir_path = os.path.join(frame_dir, subdir)
    if os.path.isdir(subdir_path):
        for file in os.listdir(subdir_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                frame_num_match = re.search(r"\d+", os.path.splitext(file)[0])
                frame_num = str(int(frame_num_match.group())) if frame_num_match else os.path.splitext(file)[0]
                # Use the actual folder name (subdir) and frame_num
                image_paths.append((os.path.join(subdir_path, file), subdir, frame_num))

# Sort image_paths by folder, then by frame number (as int if possible)
def try_int(x):
    try:
        return int(x)
    except ValueError:
        return x

image_paths.sort(key=lambda x: (x[1], try_int(x[2])))







# Car bounding box as 4 corner points (relative to car center)
# (x, y, z) in meters: front-left, front-right, rear-right, rear-left
car_bbox_corners = [
    (2.44, 1.04, 0.7),    # front-left (top)
    (2.44, -1.04, 0.7),   # front-right (top)
    (-2.44, -1.04, 0.7),  # rear-right (top)
    (-2.44, 1.04, 0.7)    # rear-left (top)
]

import math

def get_bbox_world(center, heading_deg, z_center):
    # center: (x, y) in world coordinates, heading_deg: float, z_center: float
    # heading is in degrees, 0 means facing +x axis
    cx, cy = float(center[0]), float(center[1])
    cz = float(z_center)
    theta = math.radians(float(heading_deg))
    corners_world = []
    for bx, by, bz in car_bbox_corners:
        # rotate bbox by heading
        wx = cx + bx * math.cos(theta) - by * math.sin(theta)
        wy = cy + bx * math.sin(theta) + by * math.cos(theta)
        wz = cz + bz
        corners_world.append((round(wx, 3), round(wy, 3), round(wz, 3)))
    return corners_world


def get_traj_log_path(folder):
    # folder is like: {position}_{heading}
    # log file is: pos_{position}_head_{heading}.csv
    try:
        # Split on last underscore to separate position and heading
        last_underscore = folder.rfind('_')
        if last_underscore == -1:
            return None
        position = folder[:last_underscore]
        heading = folder[last_underscore+1:]
        fname = f"pos_{position}_head_{heading}.csv"
        fpath_fail = os.path.join(failure_traj_logs, fname)
        fpath_succ = os.path.join(success_traj_logs, fname)
        if os.path.exists(fpath_fail):
            return fpath_fail
        elif os.path.exists(fpath_succ):
            return fpath_succ
        # Try legacy naming (no _head)
        fname_legacy = f"pos_{position}_{heading}.csv"
        fpath_fail_legacy = os.path.join(failure_traj_logs, fname_legacy)
        fpath_succ_legacy = os.path.join(success_traj_logs, fname_legacy)
        if os.path.exists(fpath_fail_legacy):
            return fpath_fail_legacy
        elif os.path.exists(fpath_succ_legacy):
            return fpath_succ_legacy
    except Exception:
        pass
    return None


def get_frame_state(traj_path, frame_num):
    if not traj_path:
        return None, None
    try:
        with open(traj_path, "r") as tf:
            lines = tf.readlines()
        # skip header
        data = [l.strip().split(",") for l in lines[1:]]
        frame_idx = int(frame_num)
        curr = data[frame_idx] if frame_idx < len(data) else None
        next_state = data[frame_idx+1] if frame_idx+1 < len(data) else None
        return curr, next_state
    except Exception:
        return None, None

# Helper to get left/right folder and frame for each image pair

def get_pair_info(img_path):
    # paired_frames/{folder}/{frame}.png
    parts = img_path.split(os.sep)
    if len(parts) < 3:
        return None, None, None, None
    folder = parts[-2]
    frame = parts[-1].split(".")[0]
    # For paired images, right image is always frame+1 in same folder
    try:
        frame_num = int(frame)
        right_frame = str(frame_num + 1).zfill(5)
    except Exception:
        right_frame = None
    return folder, frame, folder, right_frame

jsonl_data = []



for img_path, folder, frame_num in image_paths:
    label = 0 if (folder, frame_num) in failure_set else 1
    # Get left and right image info
    left_folder, left_frame, right_folder, right_frame = get_pair_info(img_path)
    # Get left image state and next state
    left_traj_path = get_traj_log_path(left_folder)
    left_state, left_next = get_frame_state(left_traj_path, left_frame)
    # Get right image state and next state
    right_traj_path = get_traj_log_path(right_folder)
    right_state, right_next = get_frame_state(right_traj_path, right_frame) if right_frame else (None, None)

    # If trajectory data is missing, try to find the closest available frame in the log
    def get_valid_state_pair(traj_path, frame):
        if not traj_path or frame is None:
            return None, None
        try:
            with open(traj_path, "r") as tf:
                lines = tf.readlines()
            data = [l.strip().split(",") for l in lines[1:]]
            idx = int(frame)
            curr = data[idx] if idx < len(data) else (data[-1] if data else None)
            next_state = data[idx+1] if idx+1 < len(data) else None
            return curr, next_state
        except Exception:
            pass
        return None, None

    left_state, left_next = left_state or get_valid_state_pair(left_traj_path, left_frame)[0], left_next or get_valid_state_pair(left_traj_path, left_frame)[1]
    right_state, right_next = right_state or get_valid_state_pair(right_traj_path, right_frame)[0], right_next or get_valid_state_pair(right_traj_path, right_frame)[1]


    # Calculate bounding box for left and right image (if possible)
    if left_state:
        left_str = f" The current position of the car in left image is: x={left_state[1]}, y={left_state[2]}, z={left_state[3]}"
        left_bbox = get_bbox_world((left_state[1], left_state[2]), left_folder.split('_')[-1], left_state[3])
        left_bbox_str = f" The bounding box corners of the car in the left image (world, x, y, z): {left_bbox}"
    else:
        left_str = "Left image position: not found in trajectory log"
        left_bbox_str = "Left image bounding box: unavailable"
    if left_next:
        left_next_str = f" The next position of the car in the left image is: x={left_next[1]}, y={left_next[2]}, z={left_next[3]}"
    else:
        left_next_str = "Left image next position: not found in trajectory log"
    if right_state:
        right_str = f" The current position of the car in the right image is: x={right_state[1]}, y={right_state[2]}, z={right_state[3]}"
        right_bbox = get_bbox_world((right_state[1], right_state[2]), right_folder.split('_')[-1], right_state[3])
        right_bbox_str = f" The bounding box corners of the car in the right image (world, x, y, z): {right_bbox}"
    else:
        right_str = "Right image position: not found in trajectory log"
        right_bbox_str = "Right image bounding box: unavailable"
    if right_next:
        right_next_str = f" The next position of the car in the right image is: x={right_next[1]}, y={right_next[2]}, z={right_next[3]}"
    else:
        right_next_str = "Right image next position: not found in trajectory log"

    prompt = (
        f"This is a paired image of two self-driving cars using a vision based algorithm to steer under different weather conditions, while following the yellow line on the road. "
        f"Left: {left_str}.{left_bbox_str}.{left_next_str}. "
        f"Right: {right_str}.{right_bbox_str}.{right_next_str}. "
        "Given the current position and future trajectory of both cars, is there a cause of failure in the image on the left? Answer in yes or no format."
    )
    entry = {
        "image": img_path,
        "prompt": prompt,
        "label": label
    }
    jsonl_data.append(entry)

with open("llava_finetune.json", "w") as jf:
    json.dump(jsonl_data, jf, indent=2)