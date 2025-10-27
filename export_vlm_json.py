"""
Export VLM training data to JSON: for each sequence, store initial 16 image paths, label, collision object (if any), and a prompt.
"""
import os
import csv
import json

CSV_PATH = "combined_shuffled/combined_shuffled_reordered.csv"
FRAMES_ROOT = "combined_shuffled/frames"
OUT_JSON = "vlm_sequences.json"
NUM_FRAMES = 16

 # label_map removed, only use label words

prompt_text = (
    "Classify this trajectory as success, lane violation, or collision. "
    "Give reasoning to support your classification. "
    "If there was a collision, what static object will the car collide with?"
)

sequences = []
with open(CSV_PATH, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        folder = os.path.join(FRAMES_ROOT, os.path.splitext(row["new_filename"])[0])
        frames = sorted([f for f in os.listdir(folder) if f.endswith(".jpg")])[:NUM_FRAMES]
        frame_paths = [os.path.join(folder, fname) for fname in frames]
        label = row["label"].lower()
        collision_object = row["collision_object"] if label == "collision" else None
        sequences.append({
            "frames": frame_paths,
            "label": label,
            "collision_object": collision_object,
            "prompt": prompt_text
        })


with open(OUT_JSON, "w") as f:
    json.dump(sequences, f, indent=2)
print(f"Exported {len(sequences)} sequences to {OUT_JSON}")


# Export Llava 1.5 input format
LLAVA_OUT_JSON = "llava_input.json"
llava_entries = []
for seq in sequences:
    llava_entries.append({
        "images": seq["frames"],
        "prompt": seq["prompt"],
        "expected": seq["label"],
        "collision_object": seq["collision_object"]
    })
with open(LLAVA_OUT_JSON, "w") as f:
    json.dump(llava_entries, f, indent=2)
print(f"Exported {len(llava_entries)} entries to {LLAVA_OUT_JSON}")