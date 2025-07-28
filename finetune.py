import json
import torch
import numpy as np
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor, Trainer, TrainingArguments
from torch.utils.data import Dataset, random_split
import random
from tqdm import tqdm

dataset = "llava_finetune.json"

# Custom Dataset for loading JSON data
class LlavaJsonDataset(Dataset):
    def __init__(self, json_path, processor, max_length=128):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Assumes each item has 'image' (path) and 'text' fields
        image = Image.open(item['image']).convert('RGB')
        text = item['text']
        processed = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        # Remove batch dimension
        processed = {k: v.squeeze(0) for k, v in processed.items()}
        return processed

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
    first_word = text.strip().split()[0].lower()
    if first_word == 'yes':
        return 1
    elif first_word == 'no':
        return 0
    else:
        raise ValueError(f"Unknown label: {first_word}")

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
        text = item['text']
        label = label_from_text(text)
        processed = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        processed = {k: v.squeeze(0) for k, v in processed.items()}
        processed['labels'] = torch.tensor(label, dtype=torch.long)
        return processed

def main():
    model_name = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(model_name)
    model = LlavaForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)
    model = prepare_model_with_lora(model)

    dataset_path = dataset
    full_dataset = LlavaJsonClassificationDataset(dataset_path, processor)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        save_strategy="epoch",
        logging_dir="./logs",
        learning_rate=2e-4,
        fp16=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor,
    )

    trainer.train()

    # Save the finetuned model and processor
    save_dir = "./finetuned_llava"
    trainer.save_model(save_dir)
    processor.save_pretrained(save_dir)

if __name__ == "__main__":
    main()