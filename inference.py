
import av
import torch
import numpy as np
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor, BitsAndBytesConfig

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`list[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            frames.append(frame)
        if len(frames) == len(indices):
            break
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


# 8-bit quantization config
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
)

# Load the model with 8-bit quantization
model = VideoLlavaForConditionalGeneration.from_pretrained(
    "LanguageBind/Video-LLaVA-7B-hf",
    quantization_config=quantization_config,
    device_map="auto"
)
processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")


# Load the video as an np.array, sampling uniformly 20 frames
video_path = "vlm_data/town 2_town2_rainy_collision_run_8.mp4"  # Use your local path
container = av.open(video_path)
total_frames = container.streams.video[0].frames
indices = np.linspace(0, total_frames - 1, 15).astype(int)
video = read_video_pyav(container, indices)

# For better results, we recommend to prompt the model in the following format
prompt = "USER: <video>\nThis is a video sequence from a car's vision controller. This sequence *is* the trajectory of the car.\n\nPredict: **Success** (stays on road) or **Failure** (off-road or collision).\n\nReasoning: Explain *why* based on how the where the car is heading and what it might collide with. ASSISTANT:"
inputs = processor(text=prompt, videos=video, return_tensors="pt")

# Move all tensors to the same device as the model
device = model.device if hasattr(model, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for k, v in inputs.items():
    if isinstance(v, torch.Tensor):
        inputs[k] = v.to(device)

out = model.generate(**inputs, max_new_tokens=200)
print(processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0])