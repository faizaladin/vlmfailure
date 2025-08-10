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

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    dataset = LlavaJsonClassificationDataset("llava_finetune.json", processor)

    model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],  # adjust for your model
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
    accumulation_steps = 4  # For gradient accumulation
    scaler = torch.cuda.amp.GradScaler()  # For mixed precision

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

    # Calculate batch size: 2 * len(failure_set), rounded down to nearest power of 2
    def nearest_power_of_2(n):
        return 2 ** (n.bit_length() - 1) if n > 0 else 1

    batch_size = nearest_power_of_2(len(failure_set) * 2)
    print(f"Calculated batch size: {batch_size}")

    for epoch in range(epochs):
        # Get indices for failure (label==0) and success (label==1) in training set
        failure_indices = [idx for idx in training_dataset.indices if dataset.data[idx]['label'] == 0]
        success_indices = [idx for idx in training_dataset.indices if dataset.data[idx]['label'] == 1]

        # Number of each class to sample for the batch
        half_batch = batch_size // 2
        num_failures = min(half_batch, len(failure_indices))
        num_successes = min(half_batch, len(success_indices))

        # Randomly sample indices for the batch
        selected_failure_indices = np.random.choice(failure_indices, num_failures, replace=False)
        selected_success_indices = np.random.choice(success_indices, num_successes, replace=False)
        batch_indices = np.concatenate([selected_failure_indices, selected_success_indices])
        np.random.shuffle(batch_indices)

        # Create the batch
        batch = [dataset[idx] for idx in batch_indices]
        batch = collate_fn(batch)
        print(f"Batch size: {len(batch['label'])}, Failures: {int((batch['label']==0).sum())}, Successes: {int((batch['label']==1).sum())}")

        # Create a DataLoader for this batch
        batch_dataset = torch.utils.data.TensorDataset(
            batch['input_ids'],
            batch['attention_mask'],
            batch['labels'],
            batch['label']
        )
        batch_loader = DataLoader(batch_dataset, batch_size=1, shuffle=False)
        print(f"Number of items in the batch (from batch_loader): {len(batch_loader.dataset)}")

        total_loss = 0
        optimizer.zero_grad()

        cos = CosineSimilarity(dim=1)
        mse = MSELoss()

        for step, (input_ids, attention_mask, labels, targets) in enumerate(batch_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            targets = targets.to(device)

            with torch.cuda.amp.autocast():
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,  # <-- get hidden states
                    labels=labels
                )
                # Get the last hidden state for the answer token
                # (Assume answer is at the last position; adjust as needed)
                hidden_states = outputs.hidden_states[-1]  # (batch, seq, hidden)
                answer_emb = hidden_states[:, -1, :]       # (batch, hidden)

                # Reference embeddings (precompute and move to device)
                ref_yes = "yes there is a failure"
                ref_no = "there is no cause of failure"
                with torch.no_grad():
                    ref_yes_ids = processor.tokenizer(ref_yes, return_tensors="pt").input_ids.to(device)
                    ref_no_ids = processor.tokenizer(ref_no, return_tensors="pt").input_ids.to(device)
                    embed_tokens = model.model.model.language_model.embed_tokens
                    yes_emb = embed_tokens(ref_yes_ids).mean(dim=1)  # (1, hidden)
                    no_emb = embed_tokens(ref_no_ids).mean(dim=1)    # (1, hidden)

                # Cosine similarities
                sim_yes = cos(answer_emb, yes_emb)  # (batch,)
                sim_no = cos(answer_emb, no_emb)    # (batch,)

                # Target: if GT label==0, want sim_yes=1, sim_no=-1; if GT label==1, want sim_no=1, sim_yes=-1
                target_sim_yes = (targets == 0).float() * 1.0 + (targets == 1).float() * -1.0
                target_sim_no = (targets == 1).float() * 1.0 + (targets == 0).float() * -1.0

                # Loss: encourage correct similarity to be high, incorrect to be low
                loss = mse(sim_yes, target_sim_yes) + mse(sim_no, target_sim_no)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(batch_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps

            # For monitoring: which similarity is higher?
            pred_label_cos = 0 if sim_yes.item() > sim_no.item() else 1
            print(f"CosineSim Pred label: {pred_label_cos}, GT label: {int(targets.item())}")

        avg_loss = total_loss / len(batch_loader)
        print(f"Epoch {epoch+1} - Training Loss: {avg_loss:.4f}")