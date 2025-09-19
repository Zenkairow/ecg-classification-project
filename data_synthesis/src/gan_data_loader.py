# data_synthesis/src/gan_data_loader.py

import torch
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms

class PairedECGDataset(Dataset):
    """
    A PyTorch Dataset class for loading our paired synthetic ECG images.
    It loads the realistic 'input' image and the clean 'target' image.
    """
    def __init__(self, root_dir, image_size=(256, 512)):
        """
        Args:
            root_dir (str): The root directory of the synthesized dataset (e.g., 'data_synthesis/output/').
            image_size (tuple): The target size (height, width) to resize images to.
        """
        self.root_dir = root_dir
        self.input_image_dir = os.path.join(root_dir, 'images')
        self.target_image_dir = os.path.join(root_dir, 'targets')
        
        # Get a list of all the image filenames from the input directory
        self.filenames = sorted([f for f in os.listdir(self.input_image_dir) if f.endswith('.png')])
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize to [-1, 1] for GANs
        ])

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Fetches a single paired sample of (input_image, target_image).
        """
        img_name = self.filenames[idx]
        
        # --- Load the Input Image (Realistic Photo) ---
        input_img_path = os.path.join(self.input_image_dir, img_name)
        input_image = Image.open(input_img_path).convert("RGB")
        input_tensor = self.transform(input_image)
        
        # --- Load the Target Image (Clean Plot) ---
        target_img_path = os.path.join(self.target_image_dir, img_name)
        target_image = Image.open(target_img_path).convert("RGB")
        target_tensor = self.transform(target_image)
        
        return {"input": input_tensor, "target": target_tensor}

# --- Block for testing the data loader ---
if __name__ == '__main__':
    print("Testing the PairedECGDataset class...")
    # This path is inside the container
    dataset = PairedECGDataset(root_dir='/workspace/ecg-classification-project/data_synthesis/output/')
    
    print(f"Found {len(dataset)} paired samples.")
    
    # Get a single sample
    sample = dataset[0]
    input_tensor = sample['input']
    target_tensor = sample['target']
    
    print("Sample 0 Input Image Tensor Shape:", input_tensor.shape)
    print("Sample 0 Target Image Tensor Shape:", target_tensor.shape)