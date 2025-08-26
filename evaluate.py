import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
from peft import PeftModel
import sys

dataset = "llava_finetune.json"

# --- 1. Define Quantization Config ---
# This MUST match the quantization used during training
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# --- 2. Load Base Model and Processor ---
base_model_id = "llava-hf/llava-1.5-7b-hf"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading base model on: {device}")

# Load the base model with 4-bit quantization
base_model = LlavaForConditionalGeneration.from_pretrained(
    base_model_id,
    quantization_config=quantization_config,
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(base_model_id)

# --- 3. Load and Merge the LoRA Adapter ---
# This should be the directory where the final adapter was saved
adapter_dir = "llava-finetuned" 

print(f"Loading LoRA adapter from: {adapter_dir}")
model = PeftModel.from_pretrained(base_model, adapter_dir)

print("Merging adapter weights for faster inference...")
model = model.merge_and_unload()
print("Merge complete.")


# --- 4. Prepare Inputs and Run Inference ---

# This is the prompt text from your training data
prompt_text = "This is a paired image of two cars using a vision based algorithm to steer under different weather conditions, while following the yellow line on the road. First, describe the image on the right. What is the setting (e.g., road type, environment)? What is the weather condition? Next, describe the image on the left. What are the key differences compared to the right image, specifically regarding the time of day, weather, and visibility? Identify the key static objects present in both scenes. Static objects are things that don't move, such as the road, guardrail, fence, mountains, and streetlights. Based on the path indicated by the yellow line, is the vehicle heading towards a safe path on the road or is it heading towards one of the static objects you listed earlier (like the guardrail)? This determines if a failure is occurring. Using this information determine if there is a cause of failure in the image on the left, and explain why. If there is no cause of failure, please answer no and explain why there is no cause of failure. If there is, please answer yes followed by reasoning specific to the image and pertains to the weather condition. Create a list of these failures and provide bullet points of reasoning for each one. Lastly, see if there are similar failures that could arise in different weather conditions."

# Load your image
image = Image.open("paired_success.png")

# **FIX:** Manually create the prompt in the EXACT same format as training
full_prompt = f"USER: <image>\n{prompt_text}\nASSISTANT:"

# Process the inputs using the same method as in the training script's __getitem__
inputs = processor(text=full_prompt, images=image, return_tensors="pt").to(device)


# --- 5. Generate and Decode Output ---
print("Generating response...")
# Generate a very short response, as we only expect "yes" or "no"
generate_ids = model.generate(**inputs, max_new_tokens=10000)

# Decode the output
response_text = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

# Extract only the assistant's answer
try:
    # The raw output includes the prompt, so we split to get only the generated part
    answer = response_text.split("ASSISTANT:")[-1].strip()
except IndexError:
    answer = "Could not parse answer."

print("-" * 30)
print(f"Model's Answer: '{answer}'")
print("-" * 30)