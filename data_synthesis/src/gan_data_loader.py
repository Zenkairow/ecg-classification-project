# data_synthesis/src/gan_data_loader.py
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as transforms

class PairedECGDataset(Dataset):
    def __init__(self, root_dir, image_size=(256, 512)):
        self.root_dir = root_dir
        self.input_image_dir = os.path.join(root_dir, 'images')
        self.target_image_dir = os.path.join(root_dir, 'targets')
        self.filenames = sorted([f for f in os.listdir(self.input_image_dir) if f.endswith('.png')])
        
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        input_img_path = os.path.join(self.input_image_dir, img_name)
        input_image = Image.open(input_img_path).convert("RGB")
        input_tensor = self.transform(input_image)
        
        target_img_path = os.path.join(self.target_image_dir, img_name)
        target_image = Image.open(target_img_path).convert("RGB")
        target_tensor = self.transform(target_image)
        
        return {"input": input_tensor, "target": target_tensor}

if __name__ == '__main__':
    print("Testing the PairedECGDataset class...")
    # Using the full, correct path for the server environment
    dataset = PairedECGDataset(root_dir='/workspace/ecg-classification-project/data_synthesis/output/')
    
    print(f"Found {len(dataset)} paired samples.")
    sample = dataset[0]
    print("Sample 0 Input Image Tensor Shape:", sample['input'].shape)
    print("Sample 0 Target Image Tensor Shape:", sample['target'].shape)