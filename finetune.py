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

# Balanced batch sampler
class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size=128, pos_per_batch=64, neg_per_batch=64):
        self.dataset = dataset
        self.batch_size = batch_size
        self.pos_per_batch = pos_per_batch
        self.neg_per_batch = neg_per_batch

        # Find indices for each class
        self.pos_indices = [i for i, item in enumerate(dataset.data) if item['label'] == 1]
        self.neg_indices = [i for i, item in enumerate(dataset.data) if item['label'] == 0]
        self.num_batches = min(len(self.pos_indices) // pos_per_batch, len(self.neg_indices) // neg_per_batch)

    def __iter__(self):
        pos_indices = self.pos_indices.copy()
        neg_indices = self.neg_indices.copy()
        random.shuffle(pos_indices)
        random.shuffle(neg_indices)
        for i in range(self.num_batches):
            pos_batch = pos_indices[i*self.pos_per_batch:(i+1)*self.pos_per_batch]
            neg_batch = neg_indices[i*self.neg_per_batch:(i+1)*self.neg_per_batch]
            batch = pos_batch + neg_batch
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches

def collate_fn(batch):
    # Collate a list of dicts into a dict of batched tensors
    out = {}
    for key in batch[0]:
        if isinstance(batch[0][key], torch.Tensor):
            out[key] = torch.stack([item[key] for item in batch])
        else:
            out[key] = [item[key] for item in batch]
    return out

def main():
    model_name = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(model_name)

    dataset_path = dataset
    full_dataset = LlavaJsonClassificationDataset(dataset_path, processor)
    labels = [item['label'] for item in full_dataset.data]
    num_pos = sum(1 for l in labels if l == 1)
    num_neg = sum(1 for l in labels if l == 0)
    if num_neg > 0:
        pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float)
        print(f"Class imbalance: {num_neg} failure (0), {num_pos} no failure (1), pos_weight for BCE: {pos_weight.item():.2f}")
    else:
        pos_weight = torch.tensor([1.0], dtype=torch.float)

    model = LlavaForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)
    model.gradient_checkpointing_enable()
    model = prepare_model_with_lora(model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Total dataset size: {len(full_dataset)}")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    wandb.init(project="llava-finetune", name="llava-1.5-7b-binary")

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=2,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        save_strategy="epoch",
        logging_dir="./logs",
        learning_rate=2e-4,
        fp16=True,
        report_to="wandb",
        run_name="llava-1.5-7b-binary"
    )

    from transformers import TrainerCallback

    class PrintCallback(TrainerCallback):
        def on_epoch_begin(self, args, state, control, **kwargs):
            print(f"\nStarting epoch {int(state.epoch)+1 if state.epoch is not None else '?'}...")
        def on_epoch_end(self, args, state, control, logs=None, **kwargs):
            print(f"Finished epoch {int(state.epoch)+1 if state.epoch is not None else '?'}.")
            import wandb
            if logs:
                if 'loss' in logs:
                    print(f"Average training loss for epoch: {logs['loss']:.4f}")
                    wandb.log({"train_loss": logs['loss']}, step=int(state.epoch)+1 if state.epoch is not None else None)
                if 'eval_loss' in logs:
                    print(f"Validation loss for epoch: {logs['eval_loss']:.4f}")
                    wandb.log({"val_loss": logs['eval_loss']}, step=int(state.epoch)+1 if state.epoch is not None else None)

    print("Starting training...")

    from transformers import Trainer
    from torch.nn import BCEWithLogitsLoss

    class CustomTrainer(Trainer):
        def __init__(self, *args, pos_weight=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.pos_weight = pos_weight

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.logits

            first_token_logits = logits[:, -labels.shape[1], :]
            tokenizer = self.processing_class.tokenizer if hasattr(self, 'processing_class') else self.tokenizer
            yes_id = tokenizer("yes", return_tensors="pt").input_ids[0, 1].item()
            no_id = tokenizer("no", return_tensors="pt").input_ids[0, 1].item()
            binary_logits = torch.stack([first_token_logits[:, yes_id], first_token_logits[:, no_id]], dim=1)
            gt_label = labels[0][0].item()
            target = torch.tensor([gt_label], dtype=torch.float, device=logits.device)
            selected_logit = binary_logits[:, gt_label]
            selected_logit = selected_logit.unsqueeze(1)
            target = target.unsqueeze(1)
            loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight.to(logits.device))
            loss = loss_fct(selected_logit, target)
            if return_outputs:
                return loss, outputs
            return loss

    # Use the balanced batch sampler for the training set
    train_sampler = BalancedBatchSampler(train_dataset.dataset, batch_size=128, pos_per_batch=64, neg_per_batch=64)
    train_loader = DataLoader(train_dataset.dataset, batch_sampler=train_sampler, collate_fn=collate_fn)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor,
        callbacks=[PrintCallback()],
        pos_weight=pos_weight,
        data_collator=collate_fn
    )

    # Monkey-patch the train_dataloader to use our balanced loader
    def train_dataloader_override():
        return train_loader
    trainer.train_dataloader = train_dataloader_override

    trainer.train()
    print("Training complete.")

    save_dir = "./finetuned_llava"
    print(f"Saving model and processor to {save_dir}")
    trainer.save_model(save_dir)
    processor.save_pretrained(save_dir)

if __name__ == "__main__":
    main()