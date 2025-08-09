import json
import torch
import numpy as np
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor, Trainer, TrainingArguments
from torch.utils.data import Dataset, random_split
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
    import re
    text_lower = text.strip().lower()
    # By convention:
    #   0 = failure image ("yes" answer)
    #   1 = no failure image ("no" answer)
    # Match 'yes' or 'no' as a whole word, possibly followed by punctuation
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
        # Format as a conversation for LLaVA
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
        return processed


def main():
    model_name = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(model_name)

    model = LlavaForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)
    model.gradient_checkpointing_enable()
    model = prepare_model_with_lora(model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    dataset_path = dataset
    full_dataset = LlavaJsonClassificationDataset(dataset_path, processor)

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # ...existing code...
    wandb.init(project="llava-finetune", name="llava-1.5-7b-binary")

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        save_strategy="epoch",
        logging_dir="./logs",
        learning_rate=2e-4,
        fp16=True,  # Use float16 mixed precision
        report_to="wandb",
        run_name="llava-1.5-7b-binary"
        # REMOVE project=...
    )

    from transformers import TrainerCallback

    class PrintCallback(TrainerCallback):
        def on_epoch_begin(self, args, state, control, **kwargs):
            print(f"\nStarting epoch {int(state.epoch)+1 if state.epoch is not None else '?'}...")
        def on_epoch_end(self, args, state, control, **kwargs):
            print(f"Finished epoch {int(state.epoch)+1 if state.epoch is not None else '?'}.")

    print("Trainable parameters (requires_grad=True):")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)

    print("Starting training...")

    from transformers import Trainer
    from torch.nn import BCEWithLogitsLoss

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.logits

            # Get the logits for the first generated token after the prompt
            # (Assume batch size 1 for simplicity)
            first_token_logits = logits[:, -labels.shape[1], :]  # shape: (batch, vocab)
            # Get token ids for "yes" and "no"
            yes_id = self.tokenizer("yes", return_tensors="pt").input_ids[0, 1].item()
            no_id = self.tokenizer("no", return_tensors="pt").input_ids[0, 1].item()
            # Stack logits for "yes" and "no"
            binary_logits = torch.stack([first_token_logits[:, yes_id], first_token_logits[:, no_id]], dim=1)
            # Target: 0 if label is "yes", 1 if label is "no"
            gt_label = labels[0][0].item()
            target = torch.tensor([gt_label], dtype=torch.float, device=logits.device)
            # Use only the logit for the correct class
            selected_logit = binary_logits[:, gt_label]
            selected_logit = selected_logit.unsqueeze(1)
            target = target.unsqueeze(1)
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(selected_logit, target)
            if return_outputs:
                return loss, outputs
            return loss

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor,
        callbacks=[PrintCallback()]
    )

    trainer.train()
    print("Training complete.")

    # Save the finetuned model and processor
    save_dir = "./finetuned_llava"
    print(f"Saving model and processor to {save_dir}")
    trainer.save_model(save_dir)
    processor.save_pretrained(save_dir)

if __name__ == "__main__":
    main()