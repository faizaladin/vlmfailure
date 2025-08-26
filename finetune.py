import json
import torch
import numpy as np
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor, TrainingArguments, Trainer, BitsAndBytesConfig
from torch.utils.data import Dataset, random_split, DataLoader, WeightedRandomSampler
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --- Dataset and Collate Function (No Changes) ---

class LlavaFinetuneDataset(Dataset):
    """
    Custom PyTorch Dataset for loading and processing the LLaVA fine-tuning data.
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
            raise

        prompt = item['prompt']
        label = item['label']
        target_text = "yes" if label == 1 else "no"
        full_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"

        inputs = self.processor(text=full_prompt, images=image, return_tensors="pt")
        labels = self.processor(text=target_text, return_tensors="pt", add_special_tokens=False).input_ids
        prompt_len = inputs['input_ids'].shape[1]

        inputs['input_ids'] = torch.cat([inputs['input_ids'], labels], dim=1)
        inputs['attention_mask'] = torch.ones(inputs['input_ids'].shape, dtype=torch.long)

        labels_for_loss = torch.full(inputs['input_ids'].shape, -100, dtype=torch.long)
        labels_for_loss[:, prompt_len:] = labels
        inputs['labels'] = labels_for_loss

        return {k: v.squeeze(0) for k, v in inputs.items()}

def collate_fn(batch):
    """
    Custom collate function to pad sequences to the same length in a batch.
    """
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    text_keys = ['input_ids', 'attention_mask', 'labels']
    padded_text = {}
    for key in text_keys:
        sequences = [item[key] for item in batch]
        pad_value = processor.tokenizer.pad_token_id if key != 'labels' else -100
        padded_text[key] = torch.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=pad_value
        )
    return {**padded_text, 'pixel_values': pixel_values}


# --- Custom Trainer with WeightedRandomSampler ---

class CustomTrainer(Trainer):
    """
    A custom Trainer that overrides the training dataloader to use a
    WeightedRandomSampler. This ensures each batch has a balanced
    representation of classes, which is the correct way to handle
    imbalance for this type of generative task.
    """
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        # Get the labels for the training subset
        train_indices = self.train_dataset.indices
        full_dataset = self.train_dataset.dataset
        train_labels = [full_dataset.data[i]['label'] for i in train_indices]

        # Calculate weights for each sample
        class_counts = np.bincount(train_labels)
        class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
        sample_weights = class_weights[train_labels]

        # Create the sampler
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        # Create and return the DataLoader with the custom sampler
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

    # --- 1. Model and Processor Loading (with 4-bit Quantization) ---
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = 'right'
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # --- 2. LoRA Configuration ---
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
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
    training_dataset = torch.utils.data.Subset(dataset, list(range(train_len)))
    validation_dataset = torch.utils.data.Subset(dataset, list(range(train_len, total_len)))

    print(f"Training samples: {len(training_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")

    # --- 4. Training Arguments ---
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

# --- 5. Instantiate and Run the Custom Trainer ---
    # NOTE: We are now using the CustomTrainer, not the old WeightedLossTrainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=training_dataset,
        eval_dataset=validation_dataset,
        data_collator=collate_fn,
    )

    print("Starting training with weighted random sampler...")
    trainer.train()

    trainer.save_model("llava-finetuned")
    print("Training complete. Model saved to 'llava-finetuned'")
