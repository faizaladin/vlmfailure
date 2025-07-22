import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

# prepare image
init_image = load_image("frame_0000.png")

prompt = "Make this image more photorealistic"

# pass prompt and image to pipeline
image = pipeline(prompt, image=init_image, strength=0.3).images[0]

# save the generated image
image.save("output.png")