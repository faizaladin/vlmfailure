# Batch inference over all videos in evaluation_trajectories.json
import json
import os
import av
import torch
import numpy as np
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor, BitsAndBytesConfig

def read_video_pyav(container, indices):
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

from torch import nn
# Load the model with 8-bit quantization
model = VideoLlavaForConditionalGeneration.from_pretrained(
    "LanguageBind/Video-LLaVA-7B-hf",
    quantization_config=quantization_config,
    device_map="auto"
)
processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")

# Classification head definition (same as train.py)
class ClassificationHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 2)
    def forward(self, x):
        return self.fc(x[:, 0, :])

# Attach and load weights
hidden_size = model.config.hidden_size if hasattr(model.config, 'hidden_size') else model.config.text_config.hidden_size
model.classification_head = ClassificationHead(hidden_size).to(model.device)
model.load_state_dict(torch.load("vlm_model_epoch2.pth", map_location=model.device), strict=False)
model.eval()



# Load video paths
with open("evaluation_trajectories.json", "r") as f:
    video_paths = json.load(f)

# Load ground truth labels from metadata.json
with open("vlm_data/metadata.json", "r") as f:
    metadata = json.load(f)
video_to_label = {item["video"]: item["label"] for item in metadata}

def get_ground_truth_label(video_path):
    # Use metadata mapping
    return video_to_label.get(video_path, None)


# Helper to extract predicted label from model output
def get_predicted_label_from_logits(logits):
    pred = torch.argmax(logits, dim=1).item()
    return "success" if pred == 1 else "failure"

results = []
y_true = []
y_pred = []
for video_path in video_paths:
    print(f"Processing: {video_path}")
    if not os.path.exists(video_path):
        print(f"Missing: {video_path}")
        continue
    try:
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        indices = np.linspace(0, total_frames - 1, 8).astype(int)
        video = read_video_pyav(container, indices)
        video = np.transpose(video, (0, 3, 1, 2))  # (num_frames, C, H, W)
        prompt = "USER: <video>\nThis is a video sequence from a car's vision controller. This sequence *is* the trajectory of the car.\n\nPredict: **Success** (stays on road) or **Failure** (off-road or collision).\n\nReasoning: Explain *why* based on how the where the car is heading, weather, and objects the car might collide with. ASSISTANT:"
        inputs = processor(text=prompt, videos=video, return_tensors="pt")
        device = model.device if hasattr(model, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)
        with torch.no_grad():
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                pixel_values_videos=inputs['pixel_values_videos'],
                output_hidden_states=True,
                return_dict=True
            )
            logits = model.classification_head(outputs.hidden_states[-1])
            pred_label = get_predicted_label_from_logits(logits)
        gt_label = get_ground_truth_label(video_path)
        results.append({"video": video_path, "ground_truth": gt_label, "predicted": pred_label})
        if gt_label is not None and pred_label is not None:
            y_true.append(gt_label)
            y_pred.append(pred_label)
        print(f"{video_path}: GT={gt_label}, Pred={pred_label}")
    except Exception as e:
        print(f"Error processing {video_path}: {e}")

# Compute binary metrics
def compute_metrics(y_true, y_pred):
    tp = sum((yt == 'success') and (yp == 'success') for yt, yp in zip(y_true, y_pred))
    fp = sum((yt != 'success') and (yp == 'success') for yt, yp in zip(y_true, y_pred))
    fn = sum((yt == 'success') and (yp != 'success') for yt, yp in zip(y_true, y_pred))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

precision, recall, f1 = compute_metrics(y_true, y_pred)
print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

with open("inference_results.json", "w") as f:
    json.dump(results, f, indent=2)
