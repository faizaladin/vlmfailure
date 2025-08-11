import json
import torch
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor, TrainingArguments, Trainer
from torch.utils.data import Dataset, random_split
from peft import LoraConfig, get_peft_model

# The Dataset class is mostly correct, but we'll simplify it slightly
# by removing the unnecessary binary label for the classification head.
class LlavaFinetuneDataset(Dataset):
    def __init__(self, json_path, processor):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['image']
        # Handle potential relative paths
        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            # If you have a base directory for images, specify it here
            # For example: image = Image.open(os.path.join(BASE_IMG_DIR, image_path)).convert('RGB')
            raise

        prompt = item['prompt']
        label = item['label']

        # The core task: map the binary label to a target text for generation
        # NOTE: Your original code maps 1 to "no" and 0 to "yes". Make sure this is what you intend.
        # Let's assume 1 = failure ("yes, there is a failure") and 0 = success ("no").
        target_text = "yes" if label == 1 else "no"

        # We will format the prompt and response as a conversation
        # The model is trained to generate the assistant's response.
        # The text inside `apply_chat_template` will be tokenized together.
        # The parts corresponding to the assistant's answer will be used to calculate the loss.
        full_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
        
        # Process the full input (prompt + image) and the target text for labels
        inputs = self.processor(text=full_prompt, images=image, return_tensors="pt")
        
        # Tokenize the target text but DON'T add special tokens
        labels = self.processor(text=target_text, return_tensors="pt", add_special_tokens=False).input_ids

        # Combine prompt and answer tokens to create the full `input_ids` and `labels`
        # We need to manually construct the `labels` tensor where prompt tokens are ignored (-100)
        prompt_len = inputs['input_ids'].shape[1]
        
        # Concatenate inputs and labels for the model
        inputs['input_ids'] = torch.cat([inputs['input_ids'], labels], dim=1)
        inputs['attention_mask'] = torch.ones(inputs['input_ids'].shape, dtype=torch.long)
        
        # Create the `labels` tensor for calculating loss
        # We ignore the prompt part by setting its labels to -100
        labels_for_loss = torch.full(inputs['input_ids'].shape, -100, dtype=torch.long)
        labels_for_loss[:, prompt_len:] = labels

        inputs['labels'] = labels_for_loss
        
        # Squeeze batch dimension, as DataLoader will add it back
        return {k: v.squeeze(0) for k, v in inputs.items()}

# A collate function is CRITICAL for batching together items of different sequence lengths
def collate_fn(batch):
    # Separate pixel values from other inputs
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    
    # Pad the text-based inputs
    text_keys = ['input_ids', 'attention_mask', 'labels']
    padded_text = {}
    for key in text_keys:
        sequences = [item[key] for item in batch]
        # `pad_token_id` for input_ids and attention_mask, -100 for labels
        pad_value = processor.tokenizer.pad_token_id if key != 'labels' else -100
        padded_text[key] = torch.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=pad_value
        )
        
    return {**padded_text, 'pixel_values': pixel_values}


if __name__ == "__main__":
    # --- 1. Model and Processor Loading ---
    model_id = "llava-hf/llava-1.5-7b-hf"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model in 4-bit for memory efficiency
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16, # Use float16 for speed and memory
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = 'right' # Important for generation
    processor.tokenizer.pad_token = processor.tokenizer.eos_token # Set pad token

    # --- 2. LoRA Configuration ---
    # Target all possible linear layers for better performance
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

    # --- 3. Dataset and DataLoader ---
    dataset = LlavaFinetuneDataset("llava_finetune.json", processor)
    
    # Split dataset 90/10 train/val
    total_len = len(dataset)
    train_len = int(0.9 * total_len)
    val_len = total_len - train_len
    training_dataset, validation_dataset = random_split(dataset, [train_len, val_len])
    
    print(f"Training samples: {len(training_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")

    # --- 4. Training with Hugging Face Trainer ---
    training_args = TrainingArguments(
    output_dir="llava-finetuned-model",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
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
    gradient_checkpointing=True,
    report_to="wandb",
    run_name="llava-lora-run",
)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_dataset,
        eval_dataset=validation_dataset,
        data_collator=collate_fn, # Use the custom collator
    )
    
    # Start fine-tuning! 🚀
    trainer.train()

    # Save the final model
    trainer.save_model("llava-finetuned-final")