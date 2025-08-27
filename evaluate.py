
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
from peft import PeftModel
import sys
import json
import os

dataset_path = "llava_finetune.json"


# --- 1. Define Quantization Config ---
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)


# --- 2. Load Base Model and Processor ---
base_model_id = "llava-hf/llava-1.5-7b-hf"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading base model on: {device}")

base_model = LlavaForConditionalGeneration.from_pretrained(
    base_model_id,
    quantization_config=quantization_config,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(base_model_id)

# --- 3. Load and Merge the LoRA Adapter ---
adapter_dir = "llava-finetuned"
print(f"Loading LoRA adapter from: {adapter_dir}")
model = PeftModel.from_pretrained(base_model, adapter_dir)
print("Merging adapter weights for faster inference...")
model = model.merge_and_unload()
print("Merge complete.")



# --- 4. Load Dataset and Identify Last 10 Trajectories ---
with open(dataset_path, "r") as f:
    data = json.load(f)

# Extract trajectory names from image paths
def get_traj_name(image_path):
    # e.g., paired_frames/-0.2000000000000003_25/00000.png -> -0.2000000000000003_25
    return image_path.split("/")[1]

all_traj_names = [get_traj_name(entry["image"]) for entry in data]
unique_traj_names = sorted(list(set(all_traj_names)), key=lambda x: all_traj_names.index(x))
last_10_traj = unique_traj_names[-10:]

# Filter entries for last 10 trajectories
val_entries = [entry for entry in data if get_traj_name(entry["image"]) in last_10_traj]
print(f"Number of images in last 10 trajectories: {len(val_entries)}")

# --- 5. Run Inference on Each Image ---
results = []
for idx, entry in enumerate(val_entries):
    image_path = entry["image"]
    prompt_text = entry["prompt"]
    label = entry["label"]
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        continue
    image = Image.open(image_path)
    full_prompt = f"USER: <image>\n{prompt_text}\nASSISTANT:"
    inputs = processor(text=full_prompt, images=image, return_tensors="pt").to(device)
    generate_ids = model.generate(**inputs, max_new_tokens=1000)
    response_text = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    try:
        answer = response_text.split("ASSISTANT:")[-1].strip()
    except IndexError:
        answer = "Could not parse answer."
    print("-" * 30)
    print(f"[{idx+1}/{len(val_entries)}] {image_path}")
    print(f"Label: {label}")
    print(f"Model's Answer: '{answer}'")
    print("-" * 30)
    results.append({"image": image_path, "label": label, "answer": answer})

# Optionally, save results to a file
with open("val_inference_results.json", "w") as f:
    json.dump(results, f, indent=2)