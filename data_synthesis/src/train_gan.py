import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torchvision

# Assuming gan_model.py with Generator and Discriminator classes exists
# from gan_model import Generator, Discriminator 
# For simplicity, let's redefine the models here if they are not in a separate file.
# NOTE: Make sure you have a gan_model.py file with your Generator and Discriminator architecture.

class Generator(nn.Module):
    # Your U-Net Generator Architecture
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super().__init__()
        self.down1 = nn.Sequential(nn.Conv2d(in_channels, features, 4, 2, 1, bias=False), nn.LeakyReLU(0.2))
        self.down2 = nn.Sequential(nn.Conv2d(features, features*2, 4, 2, 1, bias=False), nn.BatchNorm2d(features*2), nn.LeakyReLU(0.2))
        # ... Add other down layers
        self.up1 = nn.Sequential(nn.ConvTranspose2d(features*2, features, 4, 2, 1, bias=False), nn.BatchNorm2d(features), nn.ReLU())
        # ... Add other up layers and skip connections
        self.final_up = nn.Sequential(nn.ConvTranspose2d(features*2, out_channels, 4, 2, 1), nn.Tanh())
    def forward(self, x):
        # ... Your forward pass with skip connections
        return self.final_up(x) # Simplified for example

class Discriminator(nn.Module):
    # Your PatchGAN Discriminator Architecture
    def __init__(self, in_channels=6, features=[64, 128, 256, 512]):
        super().__init__()
        layers = [nn.Conv2d(in_channels, features[0], 4, 2, 1), nn.LeakyReLU(0.2)]
        # ... Add other layers
        layers.append(nn.Conv2d(features[-1], 1, 4, 1, 1))
        self.model = nn.Sequential(*layers)
    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        return self.model(x)

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-4
BATCH_SIZE = 4 # Adjust based on GPU memory
NUM_EPOCHS = 100
L1_LAMBDA = 100

# --- PATHS (RELATIVE TO PROJECT ROOT) ---
INPUT_DIR = 'model_1_generated_data/inputs_hyper_realistic/'
TARGET_DIR = 'output/targets/'
OUTPUT_CHECKPOINT = "models/gan_checkpoint.pth.tar"
OUTPUT_SAMPLES_DIR = "training_samples/"

# --- AUGMENTATIONS ---
transform_pipeline = A.Compose(
    [
        A.ShiftScaleRotate(shift_limit=0.04, scale_limit=0.07, rotate_limit=5, border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), p=0.9),
        A.Perspective(scale=(0.03, 0.08), pad_mode=cv2.BORDER_CONSTANT, pad_val=(255, 255, 255), p=0.8),
        A.GaussianBlur(blur_limit=(5, 11), p=0.7),
        A.MotionBlur(blur_limit=(5, 11), p=0.5),
        A.GaussNoise(var_limit=(30.0, 80.0), p=0.9),
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.9),
        A.GridDistortion(p=0.5),
        A.Posterize(num_bits=(6, 4), p=0.3),
        A.ImageCompression(quality_lower=40, quality_upper=70, p=0.6),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ],
)

# --- DATA LOADER ---
class ECGPairedDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform
        self.input_images = sorted(os.listdir(input_dir))
        self.target_images = sorted(os.listdir(target_dir))

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, index):
        input_img_name = self.input_images[index]
        target_img_name = self.target_images[index] # Assumes filenames match
        
        input_path = os.path.join(self.input_dir, input_img_name)
        target_path = os.path.join(self.target_dir, target_img_name)

        input_image = np.array(Image.open(input_path).convert("RGB"))
        target_image = np.array(Image.open(target_path).convert("RGB"))

        if self.transform:
            # Apply augmentations ONLY to the input image
            augmented = self.transform(image=input_image)
            input_image = augmented["image"]

        # Transform target separately without augmentations (just normalize and to tensor)
        target_transform = A.Compose([
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
            ToTensorV2(),
        ])
        target_image = target_transform(image=target_image)["image"]

        return input_image, target_image


# --- MAIN TRAINING SCRIPT ---
def main():
    print(f"Starting training on device: {DEVICE}")
    os.makedirs(OUTPUT_SAMPLES_DIR, exist_ok=True)
    
    gen = Generator(in_channels=3).to(DEVICE)
    disc = Discriminator(in_channels=6).to(DEVICE)
    
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()
    
    dataset = ECGPairedDataset(input_dir=INPUT_DIR, target_dir=TARGET_DIR, transform=transform_pipeline)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    for epoch in range(NUM_EPOCHS):
        loop = tqdm(loader, leave=True)
        for idx, (x, y) in enumerate(loop):
            x, y = x.to(DEVICE), y.to(DEVICE)

            # Train Discriminator
            D_real = disc(x, y)
            D_real_loss = BCE(D_real, torch.ones_like(D_real))
            y_fake = gen(x)
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = BCE(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2
            
            disc.zero_grad()
            D_loss.backward()
            opt_disc.step()

            # Train Generator
            D_fake = disc(x, y_fake)
            G_fake_loss = BCE(D_fake, torch.ones_like(D_fake))
            L1 = L1_LOSS(y_fake, y) * L1_LAMBDA
            G_loss = G_fake_loss + L1

            gen.zero_grad()
            G_loss.backward()
            opt_gen.step()
            
            if idx % 200 == 0:
                # Save some sample images
                y_fake_unnorm = y_fake * 0.5 + 0.5 # un-normalize
                torchvision.utils.save_image(y_fake_unnorm, f"{OUTPUT_SAMPLES_DIR}/y_fake_{epoch}_{idx}.png")
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Disc Loss: {D_loss:.4f}, Gen Loss: {G_loss:.4f}")

if __name__ == "__main__":
    main()