import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image

# Replace with your actual Hugging Face token
access_token = "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"

pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev",
    torch_dtype=torch.bfloat16,
    use_auth_token=access_token
)
pipe.to("cuda")

input_image = load_image("frame_0000.png")

image = pipe(
    image=input_image,
    prompt="Make this image more photorealistic",
    guidance_scale=2.5
).images[0]