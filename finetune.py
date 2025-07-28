

import json
import torch
import numpy as np
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor, Trainer, TrainingArguments
from torch.utils.data import Dataset, random_split
import random
from tqdm import tqdm

# LoRA/PEFT imports
try:
    from peft import get_peft_model, LoraConfig, TaskType
except ImportError:
    print("peft not found. Please install with: pip install peft")
    raise

dataset = "llava_finetune.json"

print("Loading model and processor...")
model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="auto")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
model.eval()

# --- LoRA/PEFT setup ---
print("Configuring LoRA for memory-efficient finetuning...")
lora_config = LoraConfig(
    r=8, # rank
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"], # typical for LLaMA-based models
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
print("LoRA configuration applied.")

# Move model to GPU if available
if torch.cuda.is_available():
    model = model.cuda()
    print("Model moved to CUDA (GPU).")
else:
    print("CUDA not available, using CPU.")
print("Model and processor loaded.")



print(f"Loading dataset from {dataset}...")
with open(dataset) as f:
    data = json.load(f)
print(f"Loaded {len(data)} samples.")


print("Shuffling and splitting dataset into train/val (80/20)...")
random.seed(42)
random.shuffle(data)
split_idx = int(0.8 * len(data))
train_data = data[:split_idx]
val_data = data[split_idx:]
print(f"Train set: {len(train_data)} samples, Val set: {len(val_data)} samples.")

print("Creating custom Dataset objects...")
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
print("Custom Dataset objects created.")

# Function to get image embedding from LLaVA vision tower
def get_embedding(image_path, prompt):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[get_embedding] Warning: Could not open image {image_path}: {e}")
        return None
    # Only get pixel_values for vision_tower
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    # Move to GPU if available
    if torch.cuda.is_available():
        inputs = {k: v.cuda().half() if torch.is_tensor(v) else v for k, v in inputs.items()}
    else:
        inputs = {k: v.float() if torch.is_tensor(v) else v for k, v in inputs.items()}
    pixel_values = inputs["pixel_values"]
    with torch.no_grad():
        vision_outputs = model.vision_tower(pixel_values=pixel_values)
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
    )
    # Move to GPU if available
    if torch.cuda.is_available():
        inputs = {k: v.cuda().half() if torch.is_tensor(v) else v for k, v in inputs.items()}
    else:
        inputs = {k: v.float() if torch.is_tensor(v) else v for k, v in inputs.items()}
    with torch.no_grad():
        generate_ids = model.generate(**inputs, max_new_tokens=1000)
    output = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
    return output


# Step 1: Embed all images and separate by label
failure_embeddings = []
failure_paths = []
num_failures = sum(1 for entry in data if entry["label"] == 0)
print(f"Embedding {num_failures} failure images...")
for idx, entry in enumerate(tqdm([e for e in data if e["label"] == 0], desc="Embedding failure images")):
    img_path = entry["image"]
    prompt = entry["prompt"]
    emb = get_embedding(img_path, prompt)
    if emb is None:
        tqdm.write(f"Warning: Skipping {img_path} due to invalid image.")
        continue
    failure_embeddings.append(emb)
    failure_paths.append(img_path)
print("Finished embedding failure images.")

# --- Custom PyTorch Training Loop ---
from torch.utils.data import DataLoader

num_epochs = 100
batch_size = 1
learning_rate = 1e-4
device = model.device


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
print("Optimizer set to only update LoRA parameters:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"  {name}")

def decode_and_map_logits(logits):
    pred_ids = torch.argmax(logits, dim=-1)
    mapped = []
    for pred in pred_ids:
        # This assumes output is [batch, seq] or [batch]
        # For simplicity, treat as binary classification
        # You may need to adjust this for your model's output
        if pred.item() == 0:
            mapped.append(0)
        elif pred.item() == 1:
            mapped.append(1)
        else:
            mapped.append(-1)
    return np.array(mapped)

def validate(model, val_loader, failure_embeddings):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            preds = decode_and_map_logits(logits)
            labels_np = labels.cpu().numpy()
            valid = preds != -1
            total += np.sum(valid)
            correct += np.sum(preds[valid] == labels_np[valid])
    acc = correct / total if total > 0 else 0.0
    print(f"Validation accuracy: {acc:.4f}")
    return acc

print("Starting custom training loop...")
best_val_acc = 0.0
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    count = 0
    epoch_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
for batch_idx, batch in epoch_bar:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    # Move tensors to GPU if available
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        labels = labels.cuda()
    optimizer.zero_grad()
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        # Custom loss logic
        pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()
        labels_np = labels.cpu().numpy()
        batch_embs = []
        for i in range(input_ids.shape[0]):
            # Re-embed the image for this batch item
            img_idx = batch_idx * batch_size + i
            img_path = train_dataset.data[img_idx]["image"]
            prompt = train_dataset.data[img_idx]["prompt"]
            emb = get_embedding(img_path, prompt)
            batch_embs.append(emb)
        batch_embs = np.stack(batch_embs)
        loss = 0.0
        for i, (pred, label, emb) in enumerate(zip(pred_ids, labels_np, batch_embs)):
            if label == 1 and pred == 0:
                dists = [np.linalg.norm(emb - f_emb) for f_emb in failure_embeddings]
                min_dist = min(dists) if dists else 0.0
                loss = loss + min_dist
            else:
                loss = loss + torch.nn.functional.cross_entropy(logits[i].unsqueeze(0), torch.tensor([label], device=logits.device))
        loss = loss / input_ids.shape[0]
        if isinstance(loss, float):
            loss = torch.tensor(loss, requires_grad=True, device=device)
    loss.backward()
    optimizer.step()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    running_loss += loss.item()
    count += 1
    if (batch_idx + 1) % 10 == 0:
        epoch_bar.set_postfix({"loss": f"{running_loss/count:.4f}"})
    print(f"Epoch {epoch+1} finished. Avg loss: {running_loss/count:.4f}")
    val_acc = validate(model, val_loader, failure_embeddings)
    if val_acc > best_val_acc:
        print("New best model found. Saving...")
        model.save_pretrained("llava_finetuned_model")
        best_val_acc = val_acc
print("Training complete. Best validation accuracy: {:.4f}".format(best_val_acc))
print("Model saved to llava_finetuned_model.")