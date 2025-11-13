import wandb
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json
import av
import tqdm
from tqdm.auto import tqdm

# --- Dataset ---

class VideoClassificationDataset(Dataset):
	def __init__(self, metadata, processor, num_frames=15):
		# metadata can be a list (already loaded) or a path to a JSON file
		if isinstance(metadata, str):
			with open(metadata, 'r') as f:
				self.metadata = json.load(f)
		else:
			self.metadata = metadata
		self.processor = processor
		self.num_frames = num_frames

	def __len__(self):
		return len(self.metadata)

	def read_video_pyav(self, video_path):
		container = av.open(video_path)
		total_frames = container.streams.video[0].frames
		indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
		frames = []
		container.seek(0)
		for i, frame in enumerate(container.decode(video=0)):
			if i in indices:
				frames.append(frame)
			if len(frames) == len(indices):
				break
		return np.stack([x.to_ndarray(format="rgb24") for x in frames])

	def __getitem__(self, idx):
		item = self.metadata[idx]
		video = self.read_video_pyav(item['video'])  # (num_frames, H, W, C)
		# Rearrange to (num_frames, C, H, W)
		video = np.transpose(video, (0, 3, 1, 2))
		prompt = "USER: <video>\nThis is a video sequence from a car's vision controller. This sequence *is* the trajectory of the car.\n\nPredict: **Success** (stays on road) or **Failure** (off-road or collision).\n\nReasoning: Explain *why* based on how the where the car is heading and what it might collide with. ASSISTANT:"
		inputs = self.processor(text=prompt, videos=video, return_tensors="pt")
		# Remove batch dimension from processor outputs
		inputs = {k: v.squeeze(0) if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == 1 else v for k, v in inputs.items()}
		label = 1 if item['label'] == 'success' else 0
		return {**inputs, 'label': torch.tensor(label, dtype=torch.long)}

# --- Classification Head ---
class ClassificationHead(nn.Module):
	def __init__(self, hidden_size):
		super().__init__()
		self.fc = nn.Linear(hidden_size, 2)

	def forward(self, x):
		# x: (batch, seq_len, hidden_size) -> use [CLS] token or mean pooling
		return self.fc(x[:, 0, :])

# --- Training Setup ---
def main():
	# Initialize wandb
	wandb.init(project="vlm-binary-classification", name="vlm-train-run")
	# Configs
	metadata_json = 'vlm_data/metadata.json'
	batch_size = 2
	num_epochs = 15
	lr = 1e-5
	num_frames = 15

	# Processor
	processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")


	# Load and shuffle metadata
	with open(metadata_json, 'r') as f:
		metadata = json.load(f)
	random.shuffle(metadata)
	split_idx = int(0.8 * len(metadata))
	train_metadata = metadata[:split_idx]
	eval_metadata = metadata[split_idx:]

	# Save eval trajectories to JSON
	with open('vlm_data/eval_trajectories.json', 'w') as f:
		json.dump(eval_metadata, f, indent=2)

	# Datasets

	train_dataset = VideoClassificationDataset(train_metadata, processor, num_frames=num_frames)
	eval_dataset = VideoClassificationDataset(eval_metadata, processor, num_frames=num_frames)

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

	# Quantization config
	quantization_config = BitsAndBytesConfig(
		load_in_8bit=True,
		llm_int8_threshold=6.0,
		llm_int8_has_fp16_weight=False,
	)

	# Model
	model = VideoLlavaForConditionalGeneration.from_pretrained(
		"LanguageBind/Video-LLaVA-7B-hf",
		quantization_config=quantization_config,
		device_map="auto"
	)

	# Prepare for LoRA
	model = prepare_model_for_kbit_training(model)
	lora_config = LoraConfig(
		r=8,
		lora_alpha=16,
		target_modules=["q_proj", "v_proj"],
		lora_dropout=0.05,
		bias="none",
	)
	model = get_peft_model(model, lora_config)

	# Classification head
	hidden_size = model.config.hidden_size if hasattr(model.config, 'hidden_size') else model.config.text_config.hidden_size
	model.classification_head = ClassificationHead(hidden_size).to(model.device)


	# Print all parameters that require gradients
	print("Parameters with requires_grad=True:")
	for name, param in model.named_parameters():
		if param.requires_grad:
			print(name, param.shape)
	for name, param in model.classification_head.named_parameters():
		if param.requires_grad:
			print("classification_head." + name, param.shape)

	# Optimizer
	optimizer = optim.AdamW(list(model.parameters()) + list(model.classification_head.parameters()), lr=lr)

	# Training loop
	model.train()

	from sklearn.metrics import precision_recall_fscore_support, accuracy_score
	train_losses = []
	eval_losses = []
	train_precisions = []
	train_recalls = []
	train_f1s = []
	eval_precisions = []
	eval_recalls = []
	eval_f1s = []

	for epoch in range(num_epochs):
	# Save model at the end of each epoch
		# Training
		model.train()
		epoch_losses = []
		all_preds = []
		all_labels = []
		train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=True, dynamic_ncols=True)
		for i, batch in enumerate(train_iter, 1):
			labels = batch['label'].to(model.device)
			video_tensor = batch['pixel_values_videos']
			outputs = model(
				input_ids=batch['input_ids'].to(model.device),
				attention_mask=batch['attention_mask'].to(model.device),
				pixel_values_videos=video_tensor.to(model.device),
				output_hidden_states=True,
				return_dict=True
			)
			logits = model.classification_head(outputs.hidden_states[-1])
			loss = nn.CrossEntropyLoss()(logits, labels)
			preds = torch.argmax(logits, dim=1)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			epoch_losses.append(loss.item())
			all_preds.extend(preds.cpu().numpy())
			all_labels.extend(labels.cpu().numpy())
			train_iter.set_postfix(batch=i, loss=loss.item())
			wandb.log({"train/batch_loss": loss.item()})
		precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
		acc = accuracy_score(all_labels, all_preds)
		train_precisions.append(precision)
		train_recalls.append(recall)
		train_f1s.append(f1)
		wandb.log({
			"train/epoch_loss": train_losses[-1],
			"train/precision": precision,
			"train/recall": recall,
			"train/f1": f1,
			"train/accuracy": acc,
			"epoch": epoch+1
		})
		torch.save(model.state_dict(), f"vlm_model_epoch{epoch+1}.pth")

		# Evaluation
		model.eval()
		eval_epoch_losses = []
		eval_preds = []
		eval_labels = []
		with torch.no_grad():
			eval_iter = tqdm(eval_loader, desc=f"Epoch {epoch+1} [Eval]", leave=True, dynamic_ncols=True)
			for i, batch in enumerate(eval_iter, 1):
				labels = batch['label'].to(model.device)
				video_tensor = batch['pixel_values_videos']
				outputs = model(
					input_ids=batch['input_ids'].squeeze(1).to(model.device),
					attention_mask=batch['attention_mask'].squeeze(1).to(model.device),
					pixel_values_videos=video_tensor.to(model.device),
					output_hidden_states=True,
					return_dict=True
				)
				logits = model.classification_head(outputs.hidden_states[-1])
				loss = nn.CrossEntropyLoss()(logits, labels)
				preds = torch.argmax(logits, dim=1)
				eval_epoch_losses.append(loss.item())
				eval_preds.extend(preds.cpu().numpy())
				eval_labels.extend(labels.cpu().numpy())
				eval_iter.set_postfix(batch=i, loss=loss.item())
				eval_iter.set_postfix(batch=i, loss=loss.item())
		eval_precision, eval_recall, eval_f1, _ = precision_recall_fscore_support(eval_labels, eval_preds, average='binary', zero_division=0)
		eval_acc = accuracy_score(eval_labels, eval_preds)
		eval_precisions.append(eval_precision)
		eval_recalls.append(eval_recall)
		eval_f1s.append(eval_f1)
		wandb.log({
			"eval/epoch_loss": eval_losses[-1],
			"eval/precision": eval_precision,
			"eval/recall": eval_recall,
			"eval/f1": eval_f1,
			"eval/accuracy": eval_acc,
			"epoch": epoch+1
		})
	print(f"Epoch {epoch+1} Train Loss: {train_losses[-1]:.4f} Eval Loss: {eval_losses[-1]:.4f} Train F1: {f1:.4f} Eval F1: {eval_f1:.4f}")
	# Save model at the end of each epoch
	torch.save(model.state_dict(), f"vlm_model_epoch{epoch+1}.pth")


	# Save the trained model
	torch.save(model.state_dict(), "vlm_model.pth")
	print("Training complete. Model saved as vlm_model.pth.")

if __name__ == "__main__":
	main()
