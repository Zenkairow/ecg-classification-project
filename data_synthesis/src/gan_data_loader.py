# data_synthesis/src/gan_data_loader.py

import torch
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms

class PairedECGDataset(Dataset):
    """
    A PyTorch Dataset class for loading our paired synthetic ECG images and raw signal data.
    """
    def __init__(self, root_dir):
        """
        Args:
            root_dir (str): The root directory of the synthesized dataset (e.g., 'data_synthesis/output/').
        """
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')
        self.signal_dir = os.path.join(root_dir, 'signals')
        
        # Get a list of all the image filenames
        self.image_filenames = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((256, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize to [-1, 1]
        ])

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.image_filenames)

    def __getitem__(self, idx):
        """
        Fetches a single paired sample of (image, signal).
        """
        # --- Load the Image ---
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)
        
        # --- Load the Signal ---
        # The signal file has the same base name but with a .npy extension
        signal_name = img_name.replace('.png', '.npy')
        signal_path = os.path.join(self.signal_dir, signal_name)
        signal_data = np.load(signal_path)
        
        # We need to reshape and normalize the signal data similarly to how we plot it
        # This is a placeholder for now; we'll need a more robust way to create the "target image"
        # For a Pix2Pix model, the target is often another image.
        # Let's represent the 12-lead signal as a 12x5000 single-channel image for now.
        
        # This part will need refinement, but for loading, it's a start.
        # Normalize signal data to be between -1 and 1 for the Tanh activation
        signal_data = (signal_data - np.min(signal_data)) / (np.max(signal_data) - np.min(signal_data))
        signal_data = (signal_data * 2) - 1
        
        signal_tensor = torch.from_numpy(signal_data).float()
        
        # The GAN will learn to generate a signal, not an image of a signal.
        # For now, let's return the raw signal tensor. We'll adapt this for the Pix2Pix architecture.
        # For Pix2Pix, we would convert this signal_tensor into an image representation.
        
        return {"image": image_tensor, "signal": signal_tensor}

# --- Block for testing the data loader ---
if __name__ == '__main__':
    print("Testing the PairedECGDataset class...")
    dataset = PairedECGDataset(root_dir='/workspace/project/data_synthesis/output/')
    
    print(f"Found {len(dataset)} paired samples.")
    
    # Get a single sample
    sample = dataset[0]
    image_tensor = sample['image']
    signal_tensor = sample['signal']
    
    print("Sample 0 Image Tensor Shape:", image_tensor.shape)
    print("Sample 0 Signal Tensor Shape:", signal_tensor.shape)