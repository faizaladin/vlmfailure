


import json
import torch
import numpy as np
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor, TrainingArguments, Trainer, BitsAndBytesConfig
from torch.utils.data import Dataset, random_split, DataLoader, WeightedRandomSampler
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# --- Classification Dataset ---
class LlavaClassificationDataset(Dataset):
    def __init__(self, json_path, processor, collision_object_map=None):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.processor = processor
        self.label_map = {"success": 0, "collision": 1, "lane violation": 2}
        # Build collision object map if not provided
        if collision_object_map is None:
            objects = set()
            for entry in self.data:
                obj = entry.get("collision_object")
                if obj:
                    objects.add(obj)
            self.collision_object_map = {obj: i for i, obj in enumerate(sorted(objects))}
        else:
            self.collision_object_map = collision_object_map

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['image']
        image = Image.open(image_path).convert('RGB')
        prompt = item['prompt']
        main_label = self.label_map[item['label']]
        collision_object = item.get('collision_object', None)
        if collision_object:
            collision_object_id = self.collision_object_map[collision_object]
        else:
            collision_object_id = -1
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'main_label': torch.tensor(main_label, dtype=torch.long),
            'collision_object_id': torch.tensor(collision_object_id, dtype=torch.long),
            'prompt': prompt,
            'collision_object': collision_object,
            'image_path': image_path
        }


# --- Classification Collate Function ---
def classification_collate_fn(batch):
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    main_labels = torch.stack([item['main_label'] for item in batch])
    collision_object_ids = torch.stack([item['collision_object_id'] for item in batch])
    prompts = [item['prompt'] for item in batch]
    collision_objects = [item['collision_object'] for item in batch]
    image_paths = [item['image_path'] for item in batch]
    return {
        'pixel_values': pixel_values,
        'main_labels': main_labels,
        'collision_object_ids': collision_object_ids,
        'prompts': prompts,
        'collision_objects': collision_objects,
        'image_paths': image_paths
    }


# --- Classification Head ---
import torch.nn as nn
class LlavaClassificationHead(nn.Module):
    def __init__(self, base_model, num_main_classes, num_collision_objects):
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size
        self.main_classifier = nn.Linear(hidden_size, num_main_classes)
        self.collision_classifier = nn.Linear(hidden_size, num_collision_objects)

    def forward(self, pixel_values):
        outputs = self.base_model.vision_tower(pixel_values=pixel_values)
        pooled = outputs.last_hidden_state.mean(dim=1)  # [batch, hidden]
        main_logits = self.main_classifier(pooled)
        collision_logits = self.collision_classifier(pooled)
        return main_logits, collision_logits

class CustomTrainer(Trainer):
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_indices = self.train_dataset.indices
        full_dataset = self.train_dataset.dataset
        train_labels = [full_dataset.data[i]['main_label'].item() for i in train_indices]
        class_counts = np.bincount(train_labels)
        class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
        sample_weights = class_weights[train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


if __name__ == "__main__":
    json_path = "llava_finetune.json"
    model_id = "llava-hf/llava-1.5-7b-hf"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
    )
    base_model = prepare_model_for_kbit_training(base_model)

    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = 'right'
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    base_model = get_peft_model(base_model, lora_config)
    base_model.print_trainable_parameters()

    # Build collision object map for dataset and head
    with open(json_path, 'r') as f:
        all_data = json.load(f)
    objects = set()
    for entry in all_data:
        obj = entry.get("collision_object")
        if obj:
            objects.add(obj)
    collision_object_map = {obj: i for i, obj in enumerate(sorted(objects))}

    dataset = LlavaClassificationDataset(json_path, processor, collision_object_map)

    def get_traj(entry):
        return entry['image'].split('/')[1]

    trajs = sorted(set(get_traj(entry) for entry in all_data))
    last_10_trajs = set(trajs[-10:])
    val_indices = [i for i, entry in enumerate(all_data) if get_traj(entry) in last_10_trajs]
    train_indices = [i for i, entry in enumerate(all_data) if get_traj(entry) not in last_10_trajs]

    training_dataset = torch.utils.data.Subset(dataset, train_indices)
    validation_dataset = torch.utils.data.Subset(dataset, val_indices)

    print(f"Training samples: {len(training_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")

    # Classification head
    num_main_classes = 3
    num_collision_objects = len(collision_object_map)
    model = LlavaClassificationHead(base_model, num_main_classes, num_collision_objects)

    training_args = TrainingArguments(
        output_dir="llava-finetuned-model-sampler",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=10,
        learning_rate=1e-5,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=200,
        eval_steps=200,
        eval_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        gradient_checkpointing=True,
        report_to="wandb",
    )

    # Custom training loop for classification
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion_main = nn.CrossEntropyLoss()
    criterion_collision = nn.CrossEntropyLoss()

    train_loader = DataLoader(training_dataset, batch_size=1, shuffle=True, collate_fn=classification_collate_fn)
    eval_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, collate_fn=classification_collate_fn)

    for epoch in range(training_args.num_train_epochs):
        model.train()
        for batch in train_loader:
            pixel_values = batch['pixel_values'].to(device)
            main_labels = batch['main_labels'].to(device)
            collision_object_ids = batch['collision_object_ids'].to(device)
            main_logits, collision_logits = model(pixel_values)
            main_loss = criterion_main(main_logits, main_labels)
            collision_mask = (main_labels == 1)
            if collision_mask.any():
                collision_loss = criterion_collision(collision_logits[collision_mask], collision_object_ids[collision_mask])
                total_loss = main_loss + collision_loss
            else:
                total_loss = main_loss
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch+1} complete.")

    # Save model
    torch.save(model.state_dict(), "llava-finetuned-classification.pt")
    print("Training complete. Model saved to 'llava-finetuned-classification.pt'")

    # Eval: generate text output for each sample
    model.eval()
    from transformers import AutoTokenizer
    tokenizer = AutoProcessor.from_pretrained(model_id).tokenizer
    with torch.no_grad():
        for batch in eval_loader:
            pixel_values = batch['pixel_values'].to(device)
            prompts = batch['prompts']
            image_paths = batch['image_paths']
            # Generate text output using base_model
            for i in range(pixel_values.size(0)):
                inputs = processor(text=prompts[i], images=Image.open(image_paths[i]).convert('RGB'), return_tensors="pt").to(device)
                output_ids = base_model.generate(**inputs, max_new_tokens=32)
                response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                print(f"Prompt: {prompts[i]}")
                print(f"Generated response: {response}")
