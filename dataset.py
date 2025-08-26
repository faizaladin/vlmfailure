
import os
import re
import json

frame_dir = "paired_frames"
failure_frames_file = "failure_points_by_trajectory.txt"


# No folder renaming: use the actual folder name as in paired_frames


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



# Assign labels for JSON only (no CSV), using the correct (folder, frame_num) logic

# Create JSON for LLaVA 1.5 finetuning
finetune_prompt = (
    "This is a paired image of two cars using a vision based algorithm to steer under different weather conditions, while following the yellow line on the road. First, describe the image on the right. What is the setting (e.g., road type, environment)? What is the weather condition? Next, describe the image on the left. What are the key differences compared to the right image, specifically regarding the time of day, weather, and visibility? Identify the key static objects present in both scenes. Static objects are things that don't move, such as the road, guardrail, fence, mountains, and streetlights. Based on the path indicated by the yellow line, is the vehicle heading towards a safe path on the road or is it heading towards one of the static objects you listed earlier (like the guardrail)? This determines if a failure is occurring. Using this information determine if there is a cause of failure in the image on the left, and explain why. If there is no cause of failure, please answer no and explain why there is no cause of failure. If there is, please answer yes followed by reasoning specific to the image and pertains to the weather condition. Create a list of these failures and provide bullet points of reasoning for each one. Lastly, see if there are similar failures that could arise in different weather conditions."
)

jsonl_data = []
for img_path, folder, frame_num in image_paths:
    label = 0 if (folder, frame_num) in failure_set else 1
    entry = {
        "image": img_path,
        "prompt": finetune_prompt,
        "label": label
    }
    jsonl_data.append(entry)

with open("llava_finetune.json", "w") as jf:
    json.dump(jsonl_data, jf, indent=2)