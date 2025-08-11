import json
import torch
import numpy as np
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor, TrainingArguments, Trainer
from torch.utils.data import Dataset, random_split
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --- Dataset and Collate Function (No Changes) ---

class LlavaFinetuneDataset(Dataset):
    """
    Custom PyTorch Dataset for loading and processing the LLaVA fine-tuning data.
    Each item in the dataset is a dictionary containing the processed image,
    input_ids, attention_mask, and labels for training.
    """
    def __init__(self, json_path, processor):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['image']
        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            # You can add a base directory for images if your paths are relative
            # e.g., image = Image.open(os.path.join(BASE_IMG_DIR, image_path)).convert('RGB')
            raise

        prompt = item['prompt']
        label = item['label']

        # Map the binary label to a target text for generation.
        # This is the "correct answer" the model will learn to generate.
        target_text = "yes" if label == 1 else "no"

        # Format the input as a conversation for the model.
        # The model is trained to generate the text following "ASSISTANT:".
        full_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
        
        # Process the full input (prompt + image)
        inputs = self.processor(text=full_prompt, images=image, return_tensors="pt")
        
        # Tokenize the target text separately to append it
        labels = self.processor(text=target_text, return_tensors="pt", add_special_tokens=False).input_ids

        # Manually construct the labels tensor for loss calculation.
        # Prompt tokens are ignored by setting them to -100.
        prompt_len = inputs['input_ids'].shape[1]
        
        # Combine prompt and answer tokens to create the full sequence
        inputs['input_ids'] = torch.cat([inputs['input_ids'], labels], dim=1)
        inputs['attention_mask'] = torch.ones(inputs['input_ids'].shape, dtype=torch.long)
        
        # Create the labels tensor for calculating loss
        labels_for_loss = torch.full(inputs['input_ids'].shape, -100, dtype=torch.long)
        labels_for_loss[:, prompt_len:] = labels
        inputs['labels'] = labels_for_loss
        
        # Squeeze the batch dimension; DataLoader will add it back
        return {k: v.squeeze(0) for k, v in inputs.items()}

def collate_fn(batch):
    """
    Custom collate function to pad sequences to the same length in a batch.
    This is essential for batching variable-length text sequences.
    """
    # Handle pixel values (images) by stacking them
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    
    # Pad text-based inputs ('input_ids', 'attention_mask', 'labels')
    text_keys = ['input_ids', 'attention_mask', 'labels']
    padded_text = {}
    for key in text_keys:
        sequences = [item[key] for item in batch]
        # Use pad_token_id for inputs and -100 for labels so they are ignored in loss
        pad_value = processor.tokenizer.pad_token_id if key != 'labels' else -100
        padded_text[key] = torch.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=pad_value
        )
        
    return {**padded_text, 'pixel_values': pixel_values}


# --- Custom Trainer with Weighted Loss ---

class WeightedLossTrainer(Trainer):
    """
    A custom Trainer that overrides the loss computation to use class weights.
    This helps mitigate class imbalance by penalizing errors on the
    minority class more heavily.
    """
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Store the class weights tensor, moving it to the correct device
        if class_weights is not None:
            self.class_weights = class_weights.to(self.args.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Overrides the default loss function to apply class weights.
        """
        labels = inputs.pop("labels")
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Flatten the logits and labels for CrossEntropyLoss
        logits_flat = logits.view(-1, self.model.config.vocab_size)
        labels_flat = labels.view(-1)
        
        # Define the loss function WITH the stored class weights
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        
        loss = loss_fct(logits_flat, labels_flat)
        return (loss, outputs) if return_outputs else loss


if __name__ == "__main__":
    # --- 0. Calculate Class Weights for Imbalance ---
    json_path = "llava_finetune.json"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Count samples for each label (0 and 1)
    class_counts = np.bincount([item['label'] for item in data])
    total_samples = len(data)

    # Calculate weights inversely proportional to class frequency
    # This gives a higher weight to the rarer class
    class_weights = total_samples / (2 * class_counts)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    print(f"Handling class imbalance. Class Counts: {class_counts}")
    print(f"Calculated Loss Weights: {class_weights_tensor}")

    # --- 1. Model and Processor Loading ---
    model_id = "llava-hf/llava-1.5-7b-hf"

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    # CRITICAL: Prepares the model for k-bit training and fixes gradient checkpointing issues
    model = prepare_model_for_kbit_training(model)
    
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = 'right'
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # --- 2. LoRA Configuration ---
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- 3. Dataset and Splitting ---
    dataset = LlavaFinetuneDataset(json_path, processor)
    
    total_len = len(dataset)
    train_len = int(0.9 * total_len)
    val_len = total_len - train_len
    training_dataset, validation_dataset = random_split(dataset, [train_len, val_len])
    
    print(f"Training samples: {len(training_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")

    # --- 4. Training Arguments ---
    training_args = TrainingArguments(
        output_dir="llava-finetuned-model-weighted",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=100,
        learning_rate=1e-5,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=200,
        eval_steps=200,
        evaluation_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        report_to="tensorboard", # or "wandb"
    )

    # --- 5. Instantiate and Run the Custom Trainer ---
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=training_dataset,
        eval_dataset=validation_dataset,
        data_collator=collate_fn,
        class_weights=class_weights_tensor, # Pass the calculated weights
    )
    
    print("Starting training with weighted loss...")
    trainer.train()

    # Save the final model adapter
    trainer.save_model("llava-finetuned-final")
    print("Training complete. Model saved to 'llava-finetuned-final'")

