import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from huggingface_hub import login

# Replace with your actual Hugging Face token
login(token="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")

pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

input_image = load_image("frame_0000.png")

image = pipe(
    image=input_image,
    prompt="Make this image more photorealistic",
    guidance_scale=2.5
).images[0]