"""
PyTorch Dataset for VLM training: loads first 16 frames from each sequence folder and label from CSV.
Assumes shuffled CSV and frames folders are aligned.
"""
import os
import csv
from typing import List, Tuple
from PIL import Image
from torch.utils.data import Dataset

class VLMTrajectoryDataset(Dataset):
    def __init__(self, csv_path: str, frames_root: str, num_frames: int = 16, transform=None):
        self.entries = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.entries.append(row)
        self.frames_root = frames_root
        self.num_frames = num_frames
        self.transform = transform
        self.label_map = {"success": 0, "collision": 1, "lane violation": 2}

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx) -> Tuple[List[Image.Image], int, str]:
        entry = self.entries[idx]
        folder = os.path.join(self.frames_root, os.path.splitext(entry["new_filename"])[0])
        # get first 16 frames (sorted)
        frames = sorted([f for f in os.listdir(folder) if f.endswith(".jpg")])[:self.num_frames]
        images = []
        for fname in frames:
            img_path = os.path.join(folder, fname)
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            images.append(img)
        label = self.label_map.get(entry["label"].lower(), -1)
        collision_object = entry["collision_object"] if label == 1 else None
        return images, label, collision_object

# Example usage:

if __name__ == "__main__":
    dataset = VLMTrajectoryDataset("combined_shuffled/combined_shuffled_reordered.csv", "combined_shuffled/frames", num_frames=16)
    images, label, collision_object = dataset[0]
    print(f"Label: {label}")
    print(f"Number of images: {len(images)}")
    print(f"First image size: {images[0].size if images else 'N/A'}")
    print(f"Image types: {[type(img) for img in images]}")
    print(f"Collision object: {collision_object}")
