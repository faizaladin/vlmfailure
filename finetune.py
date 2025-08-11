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

class BatchDictDataset(Dataset):
            def __init__(self, batch):
                self.batch = batch
            def __len__(self):
                return len(self.batch)
            def __getitem__(self, idx):
                return self.batch[idx]

class LlavaJsonClassificationDataset(Dataset):
    def __init__(self, json_path, processor, max_length=1024):
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
        # Map 0/1 to text
        if label == 0:
            target_text = "yes"
        elif label == 1:
            target_text = "no"
        else:
            target_text = str(label)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text},
                ],
            },
        ]
        # Get prompt tokens
        prompt_encoding = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        prompt_encoding = {k: v.squeeze(0) for k, v in prompt_encoding.items()}
        input_ids = prompt_encoding['input_ids']
        seq_len = input_ids.shape[0]
        # Tokenize the target_text as the label sequence
        target_tokens = self.processor.tokenizer(
            target_text,
            add_special_tokens=False,
            return_tensors="pt"
        )["input_ids"].squeeze(0)
        # Build labels: -100 for prompt, target tokens for answer, pad/truncate to match input_ids
        labels = torch.full((seq_len,), -100, dtype=torch.long)
        # Place answer tokens immediately after the prompt
        prompt_len = (input_ids != self.processor.tokenizer.pad_token_id).sum().item()
        answer_len = min(target_tokens.shape[0], seq_len - prompt_len)
        if answer_len > 0 and prompt_len + answer_len <= seq_len:
            labels[prompt_len:prompt_len+answer_len] = target_tokens[:answer_len]
        processed = prompt_encoding
        processed['labels'] = labels
        processed['target_text'] = target_text
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
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    print('Any parameters require grad:', any(p.requires_grad for p in model.parameters()))
    print('Parameters that require grad:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print('  ', name)
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
    model = model.to(device)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.BCEWithLogitsLoss()
    epochs = 1
    accumulation_steps = 4

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

    batch_loader = DataLoader(BatchDictDataset(batch), batch_size=1, shuffle=False, collate_fn=collate_fn)
    print(f"Number of items in the batch (from batch_loader): {len(batch_loader.dataset)}")

    for epoch in range(epochs):
        model.train()
        for i, batch in enumerate(batch_loader):
            optimizer.zero_grad()
            # Move all tensor inputs to device
            inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor) and k != "label"}
            output = model(**inputs)
            loss = output.loss  # This is differentiable
            print('input_ids shape:', inputs['input_ids'].shape)
            print('labels shape:', inputs['labels'].shape)
            print('labels:', inputs['labels'])
            print('Loss:', loss)
            print('Loss requires grad:', loss.requires_grad)
            print('Loss grad_fn:', loss.grad_fn)

            # Inspect output (optional)
            loss.backward()
            optimizer.step()
    

