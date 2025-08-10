import json
import torch
import numpy as np
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split

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
    

# ...existing imports and code...

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    dataset = LlavaJsonClassificationDataset("llava_finetune.json", processor)

    model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.BCEWithLogitsLoss()

    epochs = 10

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

    # model.train()
    # for epoch in range(epochs):
    #     total_loss = 0
    #     for batch in train_dataloader:
    #         input_ids = batch['input_ids'].to(device)
    #         attention_mask = batch['attention_mask'].to(device)
    #         labels = batch['labels'].to(device)
    #         targets = batch['label'].to(device)

    #         optimizer.zero_grad()
    #         outputs = model(
    #             input_ids=input_ids,
    #             attention_mask=attention_mask,
    #             labels=labels
    #         )
    #         logits = outputs.logits
    #         first_token_logits = logits[:, 0, :]
    #         tokenizer = model.config.tokenizer_class.from_pretrained(model.config._name_or_path)
    #         yes_id = tokenizer("yes", return_tensors="pt").input_ids[0, 1].item()
    #         pred_logits = first_token_logits[:, yes_id]
    #         loss = criterion(pred_logits, targets)
    #         loss.backward()
    #         optimizer.step()
    #         total_loss += loss.item()
    #     print(f"Epoch {epoch+1} - Train Loss: {total_loss/len(train_dataloader):.4f}")

    #     # Optional: Validation loop
    #     model.eval()
    #     val_loss = 0
    #     with torch.no_grad():
    #         for batch in val_dataloader:
    #             input_ids = batch['input_ids'].to(device)
    #             attention_mask = batch['attention_mask'].to(device)
    #             labels = batch['labels'].to(device)
    #             targets = batch['label'].to(device)

    #             outputs = model(
    #                 input_ids=input_ids,
    #                 attention_mask=attention_mask,
    #                 labels=labels
    #             )
    #             logits = outputs.logits
    #             first_token_logits = logits[:, 0, :]
    #             pred_logits = first_token_logits[:, yes_id]
    #             loss = criterion(pred_logits, targets)
    #             val_loss += loss.item()
    #     print(f"Epoch {epoch+1} - Val Loss: {val_loss/len(val_dataloader):.4f}")
    #     model.train()