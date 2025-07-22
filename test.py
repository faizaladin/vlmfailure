import requests
import torch
from PIL import Image
from io import BytesIO

from diffusers import StableDiffusionImg2ImgPipeline

device = "cuda"
model_id_or_path = "stable-diffusion-v1-5/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe = pipe.to(device)

init_image = Image.open("frame_0000.png").convert("RGB")
init_image = init_image.resize((768, 512))

prompt = "A car driving on a road in first person POV, 4k"

images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images
images[0].save("car_driving.png")