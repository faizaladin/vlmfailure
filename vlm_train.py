import json
import torch
import numpy as np
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor, TrainingArguments, BitsAndBytesConfig
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
import wandb
import torch.nn as nn # <-- Moved import to top

# --- Multi-Image Sequence Dataset ---
class LlavaSequenceClassificationDataset(Dataset):
    def __init__(self, json_path, processor, collision_object_map, num_frames=16):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.processor = processor
        self.label_map = {"success": 0, "collision": 1, "lane violation": 2}
        self.num_frames = num_frames
        self.collision_object_map = collision_object_map

    def __len__(self):
        return len(self.data)

    def concatenate_images(self, image_paths, resize=(224, 224)):
        images = [Image.open(p).convert("RGB").resize(resize) for p in image_paths[:self.num_frames]]
        if not images:
            # Return a blank placeholder image if no paths are provided
            return Image.new("RGB", resize, "white") 
        total_width = sum(img.size[0] for img in images)
        max_height = max(img.size[1] for img in images)
        new_img = Image.new("RGB", (total_width, max_height))
        x_offset = 0
        for img in images:
            new_img.paste(img, (x_offset, 0))
            x_offset += img.size[0]
        return new_img

    def __getitem__(self, idx):
        item = self.data[idx]
        image_paths = item['images']
        concat_img = self.concatenate_images(image_paths)
        
        prompt = f"USER: <image>\n{item['prompt']} ASSISTANT:"
        
        main_label = self.label_map[item['expected']]
        collision_object = item.get('collision_object', None)
        
        # <-- CHANGED: Assign "N/A" class for non-collision events
        if main_label == self.label_map["collision"]:
            # This is a collision event, find the object
            collision_object_id = self.collision_object_map[collision_object]
        else:
            # This is "success" or "lane violation", target is "N/A"
            collision_object_id = self.collision_object_map["N/A"]

        # The processor needs both text and images
        inputs = self.processor(
            text=prompt, 
            images=concat_img, 
            return_tensors="pt", 
            padding="max_length", # Pad to a consistent length
            max_length=1024,
            truncation=True
        )
        
        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'main_label': torch.tensor(main_label, dtype=torch.long),
            'collision_object_id': torch.tensor(collision_object_id, dtype=torch.long), # <-- Will now be N/A index for non-collisions
            'prompt': prompt, # Return the formatted prompt
            'collision_object': collision_object,
            'image_paths': image_paths
        }

# --- Classification Collate Function ---
def sequence_classification_collate_fn(batch):
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    main_labels = torch.stack([item['main_label'] for item in batch])
    collision_object_ids = torch.stack([item['collision_object_id'] for item in batch])
    prompts = [item['prompt'] for item in batch]
    collision_objects = [item['collision_object'] for item in batch]
    image_paths = [item['image_paths'] for item in batch]
    
    return {
        'pixel_values': pixel_values,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'main_labels': main_labels,
        'collision_object_ids': collision_object_ids,
        'prompts': prompts,
        'collision_objects': collision_objects,
        'image_paths': image_paths
    }

# --- Classification Head ---
class LlavaClassificationHead(nn.Module):
    def __init__(self, base_model, num_main_classes, num_collision_objects):
        super().__init__()
        self.base_model = base_model
        hidden_size = self.base_model.language_model.config.hidden_size
        self.main_classifier = nn.Linear(hidden_size, num_main_classes)
        # num_collision_objects now includes the "N/A" class
        self.collision_classifier = nn.Linear(hidden_size, num_collision_objects)

    def forward(self, pixel_values, input_ids, attention_mask):
        outputs = self.base_model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        hidden_states = outputs.hidden_states[-1]
        pooled_output = hidden_states.mean(dim=1)
        
        main_logits = self.main_classifier(pooled_output)
        collision_logits = self.collision_classifier(pooled_output)
        
        return main_logits, collision_logits

# Removed unused CustomTrainer class

if __name__ == "__main__":
    wandb.init(project="vlm_llava_training", name="vlm_driving_classification")
    json_path = "llava_input.json"
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
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = 'right'
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

    base_model = prepare_model_for_kbit_training(base_model)

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

    with open(json_path, 'r') as f:
        all_data = json.load(f)
        
    # <-- CHANGED: Build collision object map WITH "N/A"
    objects = {entry.get("collision_object") for entry in all_data if entry.get("collision_object")}
    sorted_objects = sorted(list(objects))
    # Start map with real objects
    collision_object_map = {obj: i for i, obj in enumerate(sorted_objects)}
    # Add "N/A" as the last class
    collision_object_map["N/A"] = len(sorted_objects) 

    # Pass the complete map (including "N/A") to the dataset
    dataset = LlavaSequenceClassificationDataset(json_path, processor, collision_object_map)

    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_indices, val_indices = indices[:split], indices[split:]

    training_dataset = torch.utils.data.Subset(dataset, train_indices)
    validation_dataset = torch.utils.data.Subset(dataset, val_indices)

    print(f"Training samples: {len(training_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")

    num_main_classes = 3
    # <-- CHANGED: This count now includes the "N/A" class
    num_collision_objects = len(collision_object_map) 
    model = LlavaClassificationHead(base_model, num_main_classes, num_collision_objects)

    training_args = TrainingArguments(
        output_dir="llava-finetuned-model-sampler",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        num_train_epochs=100,
        learning_rate=1e-5,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False}, 
        report_to="wandb",
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], 
        lr=training_args.learning_rate
    )
    criterion_main = nn.CrossEntropyLoss()
    # <-- CHANGED: No longer need to ignore index. We train on all samples.
    criterion_collision = nn.CrossEntropyLoss() 

    train_loader = DataLoader(training_dataset, batch_size=training_args.per_device_train_batch_size, shuffle=True, collate_fn=sequence_classification_collate_fn)
    eval_loader = DataLoader(validation_dataset, batch_size=training_args.per_device_eval_batch_size, shuffle=False, collate_fn=sequence_classification_collate_fn)

    # --- Get inverse maps for logging readable labels ---
    inv_label_map = {v: k for k, v in dataset.label_map.items()}
    # This will now correctly include "N/A"
    inv_collision_map = {v: k for k, v in dataset.collision_object_map.items()}

    for epoch in range(int(training_args.num_train_epochs)):
        model.train()
        train_iter = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        total_train_loss = 0
        for batch in train_iter:
            optimizer.zero_grad()
            
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            main_labels = batch['main_labels'].to(device)
            collision_object_ids = batch['collision_object_ids'].to(device)
            
            main_logits, collision_logits = model(pixel_values, input_ids, attention_mask)
            
            main_loss = criterion_main(main_logits, main_labels)
            
            # <-- CHANGED: Train collision loss on ALL samples.
            # Non-collision samples will be trained to predict "N/A".
            collision_loss = criterion_collision(collision_logits, collision_object_ids)
            total_loss = main_loss + collision_loss
            
            total_loss.backward()
            optimizer.step()
            
            total_train_loss += total_loss.item()
            train_iter.set_postfix({"loss": total_loss.item()})
        
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1} complete. Average Training Loss: {avg_train_loss}")
        wandb.log({"epoch": epoch+1, "train/loss": avg_train_loss})

        # --- Evaluation loop (calculates loss and logs image predictions) ---
        model.eval()
        total_eval_loss = 0
        logged_eval_batch = False
        
        with torch.no_grad():
            eval_iter = tqdm(eval_loader, desc=f"Evaluating Epoch {epoch+1}")
            for batch in eval_iter:
                pixel_values = batch['pixel_values'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                main_labels = batch['main_labels'].to(device)
                collision_object_ids = batch['collision_object_ids'].to(device)
                prompts = batch['prompts']
                image_paths_list = batch['image_paths']

                main_logits, collision_logits = model(pixel_values, input_ids, attention_mask)

                main_loss = criterion_main(main_logits, main_labels)
                
                # <-- CHANGED: Calculate collision loss on ALL samples
                collision_loss = criterion_collision(collision_logits, collision_object_ids)
                total_loss = main_loss + collision_loss
                
                total_eval_loss += total_loss.item()

                # --- NEW WANDB IMAGE LOGGING ---
                if not logged_eval_batch:
                    main_preds = torch.argmax(main_logits, dim=1)
                    collision_preds = torch.argmax(collision_logits, dim=1)
                    
                    eval_table = wandb.Table(columns=[
                        "Epoch", "Image", "Prompt", 
                        "Predicted Class", "Target Class",
                        "Predicted Collision", "Target Collision"
                    ])
                    
                    for i in range(len(prompts)):
                        pil_image = dataset.concatenate_images(image_paths_list[i])
                        
                        pred_class = inv_label_map.get(main_preds[i].item(), "N/A")
                        target_class = inv_label_map.get(main_labels[i].item(), "N/A")
                        
                        # This will now correctly show "N/A"
                        pred_coll = inv_collision_map.get(collision_preds[i].item(), "N/A")
                        target_coll = inv_collision_map.get(collision_object_ids[i].item(), "N/A")
                        
                        eval_table.add_data(
                            epoch + 1,
                            wandb.Image(pil_image),
                            prompts[i],
                            pred_class,
                            target_class,
                            pred_coll,
                            target_coll
                        )
                        
                    wandb.log({"eval/predictions": eval_table, "epoch": epoch+1})
                    logged_eval_batch = True
                # --- END OF NEW LOGGING ---

        avg_eval_loss = total_eval_loss / len(eval_loader)
        print(f"Epoch {epoch+1} Evaluation Loss: {avg_eval_loss}")
        wandb.log({"epoch": epoch+1, "eval/loss": avg_eval_loss})
    
    # Save the final model (head and LoRA adapters)
    torch.save(model.state_dict(), "llava-finetuned-classification.pt")
    print("Training complete. Model saved to 'llava-finetuned-classification.pt'")

