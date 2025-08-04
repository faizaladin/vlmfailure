import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import wandb
from models import ResNetModel
from dataloader import create_dataset

# --- Configuration ---
DATA_DIR = "paired_frames"
LABEL_FILE = "llava_dataset.csv"
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.2
MODEL_SAVE_PATH = "resnet_binary_classifier.pth"

def collate_fn(batch):
    """
    Custom collate function to filter out samples that failed to load.
    """
    # Filter out None entries
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        # Return None if the whole batch is empty
        return None, None
    # Use the default collate function on the filtered batch
    return torch.utils.data.dataloader.default_collate(batch)

def train():
    """
    Main training and validation loop.
    """
    # --- Initialize wandb ---
    wandb.init(
        project="resnet-binary-classifier",
        config={
            "learning_rate": LEARNING_RATE,
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "validation_split": VAL_SPLIT,
            "model_architecture": "ResNet",
        }
    )

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Dataset and DataLoader ---
    full_dataset = create_dataset(data_dir=DATA_DIR, label_file=LABEL_FILE)
    
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Full dataset size: {len(full_dataset)}")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=os.cpu_count()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=os.cpu_count()
    )

    # --- Model, Loss, and Optimizer ---
    model = ResNetModel(num_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Log model architecture to wandb
    wandb.watch(model, log="all", log_freq=100)

    best_val_accuracy = 0.0

    # --- Training Loop ---
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for i, (images, labels) in enumerate(train_loader):
            if images is None:
                continue

            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            correct_predictions += (preds == labels.byte()).sum().item()
            total_samples += labels.size(0)

        epoch_loss = running_loss / total_samples if total_samples > 0 else 0
        epoch_acc = correct_predictions / total_samples if total_samples > 0 else 0
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")

        # --- Validation Loop ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                if images is None:
                    continue
                
                images = images.to(device)
                labels = labels.to(device).float().unsqueeze(1)

                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                preds = torch.sigmoid(outputs) > 0.5
                val_correct += (preds == labels.byte()).sum().item()
                val_total += labels.size(0)

        val_epoch_loss = val_loss / val_total if val_total > 0 else 0
        val_epoch_acc = val_correct / val_total if val_total > 0 else 0
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.4f}")

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "train_accuracy": epoch_acc,
            "val_loss": val_epoch_loss,
            "val_accuracy": val_epoch_acc
        })

        # Save the best model
        if val_epoch_acc > best_val_accuracy:
            best_val_accuracy = val_epoch_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"New best model saved to {MODEL_SAVE_PATH} with accuracy: {best_val_accuracy:.4f}")
            # Save model to wandb
            wandb.save(MODEL_SAVE_PATH)

    print("Finished Training")
    wandb.finish()

if __name__ == '__main__':
    train()
