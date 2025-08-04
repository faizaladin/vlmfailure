import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
from peft import PeftModel

# Path to your saved model directory
model_dir = "./finetuned_llava"

# Load processor
processor = AutoProcessor.from_pretrained(model_dir)

# Load base model
model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")

# # Load LoRA weights
# model = PeftModel.from_pretrained(base_model, model_dir)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = model.to(device)

# Now use `model` and `processor` for inference
image = Image.open("paired_success_4k.png")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "This is a paired image of two cars using a vision based algorithm to steer under different weather conditions, while following the yellow line on the road. Is there a cause of failure in this image, and if so what is the cause of failure? If there is no cause of failure, please answer no and explain why there is no cause of failure. If there is, please answer yes followed by reasoning specific to the image and pertains to the weather condition. Create a list of these failures and provide bullet points of reasoning for each one. Lastly, see if there are similar failures that could arise in different weather conditions."},
        ],
    },
]

# Prepare inputs and move to the same device as model
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
)
for k, v in inputs.items():
    if isinstance(v, torch.Tensor):
        # Only cast float tensors to float16, keep index tensors as Long
        if v.dtype == torch.float:
            inputs[k] = v.to(device, torch.float16)
        else:
            inputs[k] = v.to(device)

# Generate
generate_ids = model.generate(**inputs, max_new_tokens=1000)
output = processor.batch_decode(generate_ids, skip_special_tokens=True)
print(output)