import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image

# Load the model in half-precision
model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="auto")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
image1 = Image.open("paired_fail_4k.png")
image2 = Image.open("paired_success_4k.png")


conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image1},
            {"type": "image", "image": image2},
            {"type": "text", "text": "These are two paired images of cars using a vision based algorithm to steer under two different weather conditions. The weather conditions on both left images are exactly the same and the weather conditions on the right images are exactly the same. The first pair ends in failure and the second pair succeeds. Please explain why the car crashes in the first pair of images but not in the second pair of images, despite being under the same weather conditions. Create a list of these failures and provide bullet points of reasoning for each one. Lastly, see if there are similar failures that could arise in different weather conditions."},
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
generate_ids = model.generate(**inputs, max_new_tokens=1000)
output = processor.batch_decode(generate_ids, skip_special_tokens=True)
print(output)