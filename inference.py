import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image

# Load the model in half-precision
model = LlavaForConditionalGeneration.from_pretrained("./finetuned_llava", torch_dtype=torch.float16)
processor = AutoProcessor.from_pretrained("./finetuned_llava", use_fast=True)
image = Image.open("paired_frames/pos_-2.7755575615628914e-16_head_29/frame_00013.png")


conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "This is a paired image of two cars using a vision based algorithm to steer under different weather conditions. Is there a cause of failure in this image, and if so what is the cause of failure? Please give a yes or no answer followed by reasoning specific to the image and pertains to the weather condition. Create a list of these failures and provide bullet points of reasoning for each one. Lastly, see if there are similar failures that could arise in different weather conditions."},
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

# Generate
generate_ids = model.generate(**inputs, max_new_tokens=50)
output = processor.batch_decode(generate_ids, skip_special_tokens=True)
print(output)