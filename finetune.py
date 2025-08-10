import json
import torch
import numpy as np
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor, Trainer, TrainingArguments
from torch.utils.data import Dataset, random_split, DataLoader, Sampler
import random
from tqdm import tqdm
import wandb

dataset = "llava_finetune.json"

# Custom Dataset for loading JSON data
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
        label = item['label']  # Load the label from the JSON
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
        processed['label'] = label  # Add label for sampler

# LoRA setup for Hugging Face Trainer
from peft import LoraConfig, get_peft_model, TaskType

def prepare_model_with_lora(model):
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    return model

# Training setup
def label_from_text(text):
    import re
    text_lower = text.strip().lower()
    if re.search(r'\byes\b[\.,!?:;]?', text_lower):
        return 0  # failure image
    elif re.search(r'\bno\b[\.,!?:;]?', text_lower):
        return 1  # no failure image
    else:
        return 0  # default to failure if unclear

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
        processed['label'] = item['label']  # Add label for sampler
        return processed

if __name__ == "__main__":
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    dataset_instance = LlavaJsonClassificationDataset("llava_finetune.json", processor)
    first_entry = dataset_instance[0]
    print(first_entry)