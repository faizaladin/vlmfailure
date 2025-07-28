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

def compute_image_embedding(model, image, processor, device):
    # Get image embedding from the vision encoder
    # This assumes the model has a vision_tower with a forward method
    image_tensor = processor.image_processor(image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        image_emb = model.model.vision_tower(image_tensor).last_hidden_state.mean(dim=1).squeeze(0)
    return image_emb

def precompute_failure_embeddings(model, processor, dataset, device):
    failure_embs = []
    for item in dataset:
        # 0 = failure image, 1 = no failure image
        text = item['prompt'] if 'prompt' in item else item['text']
        label = label_from_text(text)
        if label == 0:  # Only use failure images for the failure set
            image = Image.open(item['image']).convert('RGB')
            emb = compute_image_embedding(model, image, processor, device)
            failure_embs.append(emb.cpu())
    if failure_embs:
        return torch.stack(failure_embs)
    else:
        return torch.empty(0)

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

    # Precompute failure set embeddings (from training set only)
    print("Precomputing failure set embeddings...")
    # Use the raw data dicts for access to image paths and text
    failure_embs = precompute_failure_embeddings(model, processor, full_dataset.data[:train_size], device)
    print(f"Failure set size: {failure_embs.shape[0]}")

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
        report_to="none"
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
    from torch.nn import functional as F

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            # Standard loss
            labels = inputs.get("labels")
            outputs = model(**inputs)
            loss = outputs.loss

            # Get model prediction (first token after generation prompt)
            logits = outputs.logits
            # Get the predicted token for the first position (batch size 1 assumed)
            pred_token = logits[:, -labels.shape[1], :].argmax(dim=-1)
            # Map predicted token to text
            pred_text = self.tokenizer.decode(pred_token)
            # Get ground truth label (0: failure, 1: no failure)
            gt_label = labels[0][0].item() if labels is not None else None

            # If ground truth is 'no failure' (1) but model predicts 'failure' (0)
            # We'll use a simple check: if 'yes' in pred_text.lower() and gt_label == 1
            # (0 = failure, 1 = no failure)
            if gt_label == 1 and 'yes' in pred_text.lower():
                # Compute image embedding for this sample
                image = inputs['pixel_values'][0] if 'pixel_values' in inputs else None
                if image is not None and failure_embs.shape[0] > 0:
                    # Get embedding for current image
                    with torch.no_grad():
                        image_emb = model.model.vision_tower(image.unsqueeze(0)).last_hidden_state.mean(dim=1).squeeze(0)
                    # Compute Euclidean distances to all failure set embeddings
                    dists = torch.norm(failure_embs.to(image_emb.device) - image_emb, dim=1)
                    min_dist = dists.min()
                    loss = min_dist
            if return_outputs:
                return loss, outputs
            return loss

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor,
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