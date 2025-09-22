# data_synthesis/src/train_gan.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import torchvision

# Import our custom classes from the correct folders
import sys
sys.path.append('src') # For config
sys.path.append('data_synthesis/src') # For the data loader and gan_model

import config
from gan_data_loader import PairedECGDataset
from gan_model import Generator, Discriminator

# --- Configuration for Local Smoke Test ---
DATA_ROOT = 'data_synthesis/output/'
EPOCHS = 1 # We'll just run one epoch for the smoke test
LEARNING_RATE = 0.0002
BATCH_SIZE = 4 # Use a small batch size for local testing to avoid memory issues
L1_LAMBDA = 100 # Weight for the pixel-wise similarity loss

def train_gan():
    """
    Main function to orchestrate the Pix2Pix GAN training process.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Setup Models and Optimizers ---
    generator = Generator(in_channels=3, out_channels=3).to(device)
    discriminator = Discriminator(in_channels=6).to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    # --- 2. Define Loss Functions ---
    adversarial_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    # --- 3. Load Data ---
    print("Loading paired dataset...")
    dataset = PairedECGDataset(root_dir=DATA_ROOT)
    # Use num_workers=0 for local Windows test
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # --- 4. Training Loop ---
    print("Starting GAN training smoke test...")
    for epoch in range(EPOCHS):
        for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
            real_input = batch["input"].to(device)
            real_target = batch["target"].to(device)

            # --- Train Discriminator ---
            optimizer_D.zero_grad()
            fake_target = generator(real_input)
            pred_real = discriminator(real_input, real_target)
            loss_real = adversarial_loss(pred_real, torch.ones_like(pred_real))
            pred_fake = discriminator(real_input, fake_target.detach())
            loss_fake = adversarial_loss(pred_fake, torch.zeros_like(pred_fake))
            loss_D = (loss_real + loss_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()

            # --- Train Generator ---
            optimizer_G.zero_grad()
            pred_fake = discriminator(real_input, fake_target)
            loss_G_adv = adversarial_loss(pred_fake, torch.ones_like(pred_fake))
            loss_G_l1 = l1_loss(fake_target, real_target) * L1_LAMBDA
            loss_G = loss_G_adv + loss_G_l1
            loss_G.backward()
            optimizer_G.step()

        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")
        
        # Save a sample image to confirm it's working
        sample_dir = "gan_samples"
        os.makedirs(sample_dir, exist_ok=True)
        comparison_grid = torch.cat([real_input.data[:4], fake_target.data[:4], real_target.data[:4]], dim=0)
        torchvision.utils.save_image(comparison_grid, f"{sample_dir}/local_test_epoch_{epoch+1}.png", nrow=4, normalize=True)

    print("Local smoke test complete. GAN training script is working correctly.")


if __name__ == '__main__':
    train_gan()