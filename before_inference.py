import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import sys

# --- 1. Load Base Model and Processor (no quantization, no LoRA) ---
base_model_id = "llava-hf/llava-1.5-7b-hf"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading base model on: {device}")
model = LlavaForConditionalGeneration.from_pretrained(base_model_id).to(device)
processor = AutoProcessor.from_pretrained(base_model_id)

# This is the prompt text from your training data
prompt_text = "This is a paired image of two cars using a vision based algorithm to steer under different weather conditions, while following the yellow line on the road. First, describe the image on the right. What is the setting (e.g., road type, environment)? What is the weather condition? Next, describe the image on the left. What are the key differences compared to the right image, specifically regarding the time of day, weather, and visibility? Identify the key static objects present in both scenes. Static objects are things that don't move, such as the road, guardrail, fence, mountains, and streetlights. Based on the path indicated by the yellow line, is the vehicle heading towards a safe path on the road or is it heading towards one of the static objects you listed earlier (like the guardrail)? This determines if a failure is occurring. Using this information determine if there is a cause of failure in the image on the left, and explain why. If there is no cause of failure, please answer no and explain why there is no cause of failure. If there is, please answer yes followed by reasoning specific to the image and pertains to the weather condition. Create a list of these failures and provide bullet points of reasoning for each one. Lastly, see if there are similar failures that could arise in different weather conditions."

# Load your image
image = Image.open("paired_success.png")


# Keep the original format of user and assistant
full_prompt = f"USER: <image>\n{prompt_text}\nASSISTANT:"


# Process the inputs using the same method as in the training script's __getitem__
inputs = processor(text=full_prompt, images=image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}


# --- 5. Generate and Decode Output ---
print("Generating response...")
# Generate a very short response, as we only expect "yes" or "no"
generate_ids = model.generate(**inputs, max_new_tokens=10000)


# Decode and print the raw output
response_text = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("-" * 30)
print(f"Raw model output: {response_text}")
print("-" * 30)