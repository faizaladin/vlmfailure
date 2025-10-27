
"""
Llava training/inference scaffold: loads initial sequence, prompt, and expected label from llava_input.json.
"""

import json
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
import wandb


class LlavaSequenceDataset(Dataset):
    def __init__(self, json_path, num_frames=16, transform=None):
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.num_frames = num_frames
        self.transform = transform or transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        self.label_map = {"success": 0, "collision": 1, "lane violation": 2}
        # Build collision object map from all objects in dataset
        objects = set()
        for entry in self.data:
            obj = entry.get("collision_object")
            if obj:
                objects.add(obj)
        self.collision_object_map = {obj: i for i, obj in enumerate(sorted(objects))}
        print("Collision objects:", self.collision_object_map)
        print("Number of collision objects:", len(self.collision_object_map))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        images = entry["images"][:self.num_frames]
        imgs = [self.transform(Image.open(f).convert("RGB")) for f in images]
        prompt = entry["prompt"]
        expected = entry["expected"]
        label_id = self.label_map[expected]
        collision_object = entry.get("collision_object", None)
        if collision_object:
            collision_object_id = self.collision_object_map[collision_object]
        else:
            collision_object_id = -1  # -1 for non-collision samples
        return torch.stack(imgs), prompt, label_id, collision_object_id

if __name__ == "__main__":
    # Setup
    num_frames = 16
    dataset = LlavaSequenceDataset("llava_input.json", num_frames=num_frames)
    total_len = len(dataset)
    indices = list(range(total_len))
    split = int(0.8 * total_len)
    train_indices = indices[:split]  # first 80%
    eval_indices = indices[split:]   # last 20%

    train_set = torch.utils.data.Subset(dataset, train_indices)
    eval_set = torch.utils.data.Subset(dataset, eval_indices)

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    eval_loader = DataLoader(eval_set, batch_size=8, shuffle=False)

    # Initialize wandb
    wandb.init(project="vlm_llava_training", name="run_1")

    print("Training set size:", len(train_set))
    print("Eval set size:", len(eval_set))

    # Example training loop with wandb logging
    for batch_idx, batch in enumerate(train_loader):
        images, prompts, label_ids, collision_object_ids = batch
        # ...existing code for model inference and loss...
        # Example (replace with actual model and loss):
        batch_loss = torch.rand(1).item()  # Dummy loss
        wandb.log({"train/batch_loss": batch_loss, "train/batch_idx": batch_idx})
        print(f"Batch {batch_idx} loss: {batch_loss:.4f}")
        if batch_idx > 2:
            break

    # Example eval loop with wandb logging
    for batch_idx, batch in enumerate(eval_loader):
        images, prompts, label_ids, collision_object_ids = batch
        # ...existing code for model inference and eval metrics...
        # Example (replace with actual model and metrics):
        eval_loss = torch.rand(1).item()  # Dummy eval loss
        wandb.log({"eval/batch_loss": eval_loss, "eval/batch_idx": batch_idx})
        print(f"Eval batch {batch_idx} loss: {eval_loss:.4f}")
        if batch_idx > 2:
            break
