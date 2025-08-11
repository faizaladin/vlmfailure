import json
import torch
import numpy as np
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, TaskType

# Dataset classes as you had them
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

        # Map 0/1 to text label for generation target
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

        target_tokens = self.processor.tokenizer(
            target_text,
            add_special_tokens=False,
            return_tensors="pt"
        )["input_ids"].squeeze(0)

        labels = torch.full((seq_len,), -100, dtype=torch.long)
        prompt_len = (input_ids != self.processor.tokenizer.pad_token_id).sum().item()
        answer_len = min(target_tokens.shape[0], seq_len - prompt_len)
        if answer_len > 0 and prompt_len + answer_len <= seq_len:
            labels[prompt_len:prompt_len+answer_len] = target_tokens[:answer_len]

        processed = prompt_encoding
        processed['labels'] = labels
        processed['target_text'] = target_text
        processed['label'] = torch.tensor(label, dtype=torch.float)  # binary label for BCE
        return processed

# Model wrapper with classification head
class LlavaWithClassificationHead(nn.Module):
    def __init__(self, base_model, hidden_size):
        super().__init__()
        self.base_model = base_model
        self.classification_head = nn.Linear(hidden_size, 1)  # binary classification

    def forward(self, input_ids, images, labels=None, **kwargs):
        outputs = self.base_model(
            input_ids=input_ids,
            images=images,
            labels=labels,
            output_hidden_states=True,
            **kwargs,
        )
        hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, hidden_size)
        cls_hidden = hidden_states[:, -1, :]  # last token hidden state
        logits_cls = self.classification_head(cls_hidden).squeeze(-1)  # (batch,)
        return outputs, logits_cls

def nearest_power_of_2(n):
    return 2 ** (n.bit_length() - 1) if n > 0 else 1

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load processor and dataset
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    dataset = LlavaJsonClassificationDataset("llava_finetune.json", processor)

    # Load base model and apply LoRA
    base_model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    base_model = get_peft_model(base_model, lora_config)

    # Wrap model with classification head
    model = LlavaWithClassificationHead(base_model, base_model.config.hidden_size)
    model.gradient_checkpointing_enable()
    model = model.to(device)
    model.train()

    print('Trainable parameters:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print('  ', name)

    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion_cls = nn.BCEWithLogitsLoss()

    epochs = 1
    accumulation_steps = 4

    # Split dataset 80/20 train/val
    total_len = len(dataset)
    train_len = int(0.8 * total_len)
    val_len = total_len - train_len
    training_dataset, validation_dataset = random_split(dataset, [train_len, val_len])

    # Example dynamic batching based on label balance
    failure_set = set(idx for idx in training_dataset.indices if dataset.data[idx]['label'] == 0)
    print(len(failure_set), "failure frames in training set")
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
    batch_loader = DataLoader(BatchDictDataset(batch), batch_size=1, shuffle=False)

    alpha = 1.0  # generation loss weight
    beta = 1.0   # classification loss weight

    for epoch in range(epochs):
        model.train()
        for i, batch in enumerate(batch_loader):
            optimizer.zero_grad()

            # Move tensors to device
            inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor) and k not in ['label']}
            labels_cls = batch['label'].to(device)

            outputs, logits_cls = model(**inputs)
            loss_text = outputs.loss
            loss_cls = criterion_cls(logits_cls, labels_cls)

            loss = alpha * loss_text + beta * loss_cls
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch} Iter {i} Loss Text: {loss_text.item():.4f} Loss Cls: {loss_cls.item():.4f} Total: {loss.item():.4f}")
