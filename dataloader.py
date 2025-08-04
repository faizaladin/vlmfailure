import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class PairedImageDataset(Dataset):
    """
    Custom PyTorch Dataset for loading images from subfolders based on a CSV file.
    The CSV file contains relative paths to the images and their corresponding labels.
    """
    def __init__(self, data_dir, label_file, transform=None):
        """
        Args:
            data_dir (str): The root directory where the image folders are located (e.g., 'paired_frames').
            label_file (str): The path to the CSV file containing image paths and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.labels_df = pd.read_csv(label_file)
        self.transform = transform

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.labels_df)

    def __getitem__(self, idx):
        """
        Fetches an image and its label at a given index, then applies transformations.

        Args:
            idx (int): The index of the data point.

        Returns:
            tuple: A tuple containing the transformed image and its label tensor.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_rel_path = self.labels_df.iloc[idx, 0]
        label = self.labels_df.iloc[idx, 1]

        img_full_path = img_rel_path

        try:
            image = Image.open(img_full_path).convert('RGB')
        except (FileNotFoundError, Image.UnidentifiedImageError):
            print(f"Warning: Could not load or identify image file at {img_full_path}. Skipping.")
            return None, None

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

def create_dataset(data_dir, label_file):
    """
    Creates and returns the PairedImageDataset.

    Args:
        data_dir (str): The root directory for the images.
        label_file (str): The path to the CSV label file.

    Returns:
        PairedImageDataset: The configured PyTorch Dataset.
    """
    image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = PairedImageDataset(
        data_dir=data_dir,
        label_file=label_file,
        transform=image_transforms
    )

    return dataset

# Example of how to use the dataset
if __name__ == '__main__':
    data_dir = "paired_frames"
    label_file = "llava_dataset.csv"

    # Get the dataset
    my_dataset = create_dataset(data_dir, label_file)

    # Try to fetch a sample
    if len(my_dataset) > 0:
        print(f"\nSuccessfully created dataset with {len(my_dataset)} samples.")
        
        # Filter out None samples before trying to access an item
        valid_indices = [i for i, (img, lab) in enumerate(my_dataset) if img is not None]
        
        if valid_indices:
            image, label = my_dataset[valid_indices[0]]
            print(f"First valid sample image shape: {image.shape}")
            print(f"First valid sample label: {label.item()}")
        else:
            print("Could not load any valid samples from the dataset.")
            
    else:
        print("\nDataset is empty. Please check your CSV and image paths.")

