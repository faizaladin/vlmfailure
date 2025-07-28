

import json
import torch
import numpy as np
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor, Trainer, TrainingArguments
from torch.utils.data import Dataset, random_split
import random

dataset = "llava_finetune.json"

# Load model and processor
model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="auto")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
model.eval()


# Load dataset
with open(dataset) as f:
    data = json.load(f)

# Shuffle and split into train/val (80/20)
random.seed(42)
random.shuffle(data)
split_idx = int(0.8 * len(data))
train_data = data[:split_idx]
val_data = data[split_idx:]

# Custom Dataset for LLaVA finetuning
class LlavaFinetuneDataset(Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        entry = self.data[idx]
        image = Image.open(entry["image"]).convert("RGB")
        prompt = entry["prompt"]
        label = entry["label"]
        # Prepare input for LLaVA
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        # The label is a single int (0 or 1)
        return {"input_ids": inputs["input_ids"].squeeze(0), "attention_mask": inputs["attention_mask"].squeeze(0), "labels": torch.tensor(label, dtype=torch.long)}

train_dataset = LlavaFinetuneDataset(train_data, processor)
val_dataset = LlavaFinetuneDataset(val_data, processor)

# Function to get image embedding from LLaVA vision tower
def get_embedding(image_path, prompt):
    image = Image.open(image_path).convert("RGB")
    # Pass a dummy text input along with the image
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device, torch.float16)
    with torch.no_grad():
        vision_outputs = model.vision_tower(**inputs)
        emb = vision_outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return emb

# Function to get LLaVA answer to the prompt
def get_llava_answer(image_path, prompt):
    image = Image.open(image_path).convert("RGB")
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, torch.float16)
    with torch.no_grad():
        generate_ids = model.generate(**inputs, max_new_tokens=1000)
    output = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
    return output


# Step 1: Embed all images and separate by label
failure_embeddings = []
success_embeddings = []
failure_paths = []
success_paths = []
for entry in data:
    img_path = entry["image"]
    label = entry["label"]
    prompt = entry["prompt"]
    emb = get_embedding(img_path, prompt)
    if label == 0:
        failure_embeddings.append(emb)
        failure_paths.append(img_path)
    else:
        success_embeddings.append(emb)
        success_paths.append(img_path)



# TrainingArguments for Hugging Face Trainer
training_args = TrainingArguments(
    output_dir="llava_finetuned_model",
    num_train_epochs=100,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    learning_rate=1e-4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to=[],
)


# Helper to decode model outputs to yes/no and map to 0/1
def decode_and_map(pred_ids, processor):
    outputs = processor.batch_decode(pred_ids, skip_special_tokens=True)
    mapped = []
    for out in outputs:
        out = out.strip().lower()
        if out.startswith("yes"):
            mapped.append(0)
        elif out.startswith("no"):
            mapped.append(1)
        else:
            # fallback: treat as incorrect
            mapped.append(-1)
    return np.array(mapped)

def compute_metrics(eval_pred):
    pred_ids, labels = eval_pred
    preds = decode_and_map(pred_ids, processor)
    # Only compare where mapping succeeded
    valid = preds != -1
    acc = (preds[valid] == labels[valid]).mean() if np.any(valid) else 0.0
    return {"accuracy": acc}


# Custom Trainer to use min distance to failure set as loss for misclassified successes
class CustomTrainer(Trainer):
    def __init__(self, *args, failure_embeddings=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.failure_embeddings = failure_embeddings

    def compute_loss(self, model, inputs, return_outputs=False):
        # Standard forward
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"]
        )
        logits = outputs.logits
        # Decode model output to yes/no
        pred_ids = torch.argmax(logits, dim=-1)
        # Map to 0/1 as before
        preds = pred_ids.detach().cpu().numpy()
        labels = inputs["labels"].detach().cpu().numpy()
        # Get embeddings for current batch
        batch_embs = []
        for i in range(inputs["input_ids"].shape[0]):
            # Re-embed the image for this batch item
            # (Assumes batch size 1 for simplicity)
            img_idx = self.train_dataset.indices[i] if hasattr(self.train_dataset, 'indices') else i
            img_path = self.train_dataset.data[img_idx]["image"]
            prompt = self.train_dataset.data[img_idx]["prompt"]
            emb = get_embedding(img_path, prompt)
            batch_embs.append(emb)
        batch_embs = np.stack(batch_embs)
        # Custom loss: for misclassified successes, use min distance to failure set
        loss = 0.0
        count = 0
        for i, (pred, label, emb) in enumerate(zip(preds, labels, batch_embs)):
            if label == 1 and pred == 0:
                # Misclassified success, penalize by min distance to failure set
                dists = [np.linalg.norm(emb - f_emb) for f_emb in self.failure_embeddings]
                min_dist = min(dists) if dists else 0.0
                loss += min_dist
                count += 1
            else:
                # Standard cross-entropy for other cases
                loss += torch.nn.functional.cross_entropy(logits[i].unsqueeze(0), torch.tensor([label], device=logits.device))
                count += 1
        loss = loss / count if count > 0 else torch.tensor(0.0, device=logits.device)
        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    failure_embeddings=failure_embeddings,
)

# Train and save model
trainer.train()
trainer.save_model("llava_finetuned_model")