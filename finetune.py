import json
import torch
import numpy as np
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
import math
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, TaskType
from torch.nn import CosineSimilarity, MSELoss

class LlavaJsonClassificationDataset(Dataset):
    def __init__(self, json_path, processor, max_length=128):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['image']).convert('RGB')
        text = item['prompt']
        label = item['label']
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text},
                ],
            },
        ]
        processed = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        processed = {k: v.squeeze(0) for k, v in processed.items()}
        processed['labels'] = processed['input_ids'].clone()
        processed['label'] = torch.tensor(label, dtype=torch.float)
        return processed

def collate_fn(batch):
    out = {}
    for key in batch[0]:
        if isinstance(batch[0][key], torch.Tensor):
            out[key] = torch.stack([item[key] for item in batch])
        else:
            out[key] = [item[key] for item in batch]
    return out

# ... (imports and dataset code unchanged) ...

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    dataset = LlavaJsonClassificationDataset("llava_finetune.json", processor)

    model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    model = model.to(device)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.BCEWithLogitsLoss()
    epochs = 1
    accumulation_steps = 4
    scaler = torch.cuda.amp.GradScaler()

    # Split dataset: 80% train, 20% val
    total_len = len(dataset)
    train_len = int(0.8 * total_len)
    val_len = total_len - train_len
    training_dataset, validation_dataset = random_split(dataset, [train_len, val_len])
    val_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    failure_set = set()
    for idx in training_dataset.indices:
        if dataset.data[idx]['label'] == 0:
            failure_set.add(idx)
    print(len(failure_set), "failure frames in training set")

    def nearest_power_of_2(n):
        return 2 ** (n.bit_length() - 1) if n > 0 else 1

    batch_size = nearest_power_of_2(len(failure_set) * 2)
    print(f"Dynamic batch size set to: {batch_size}")


    for epoch in range(epochs):
        failure_indices = [idx for idx in training_dataset.indices if dataset.data[idx]['label'] == 0]
        success_indices = [idx for idx in training_dataset.indices if dataset.data[idx]['label'] == 1]

        half_batch = batch_size // 2
        num_failures = min(half_batch, len(failure_indices))
        num_successes = min(half_batch, len(success_indices))

        selected_failure_indices = np.random.choice(failure_indices, num_failures, replace=False)
        selected_success_indices = np.random.choice(success_indices, num_successes, replace=False)
        batch_indices = np.concatenate([selected_failure_indices, selected_success_indices])
        np.random.shuffle(batch_indices)

        batch = [dataset[idx] for idx in batch_indices]
        batch = collate_fn(batch)
        print(f"Batch size: {len(batch['label'])}, Failures: {int((batch['label']==0).sum())}, Successes: {int((batch['label']==1).sum())}")

        from torch.utils.data import Dataset

        class BatchDictDataset(Dataset):
            def __init__(self, batch):
                self.batch = batch
            def __len__(self):
                return len(self.batch['label'])
            def __getitem__(self, idx):
                return {k: v[idx] for k, v in self.batch.items()}

        batch_loader = DataLoader(BatchDictDataset(batch), batch_size=1, shuffle=False, collate_fn=collate_fn)
        print(f"Number of items in the batch (from batch_loader): {len(batch_loader.dataset)}")

        total_loss = 0
        optimizer.zero_grad()

        for step, batch_data in enumerate(batch_loader):
            input_ids = batch_data['input_ids'].to(device)
            attention_mask = batch_data['attention_mask'].to(device)
            labels = batch_data['labels'].to(device)
            targets = batch_data['label'].to(device)

            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                logits = outputs.logits  # (batch, seq, vocab)

                # Use the first token's logits for binary classification (adjust position if needed)
                first_token_logits = logits[:, 0, :]
                tokenizer = processor.tokenizer
                yes_id = tokenizer("yes", return_tensors="pt").input_ids[0, 1].item()
                no_id = tokenizer("no", return_tensors="pt").input_ids[0, 1].item()
                # Optionally, you can use both yes and no logits, but for BCEWithLogitsLoss, use one
                pred_logits = first_token_logits[:, yes_id]
                loss = criterion(pred_logits, targets) / accumulation_steps

                # For monitoring: convert logits to predicted label (0 or 1)
                pred_label = (torch.sigmoid(pred_logits) > 0.5).long().item()
                print(f"Predicted label: {pred_label}, Ground Truth label: {int(targets.item())}")

            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(batch_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps

        avg_loss = total_loss / len(batch_loader)
        print(f"Epoch {epoch+1} - Training Loss: {avg_loss:.4f}")