# data_synthesis/src/train_gan.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import torchvision

# Import our custom classes
import sys
# Add both src directories to the path to ensure all modules are found
sys.path.append('/workspace/ecg-classification-project/src')
sys.path.append('/workspace/ecg-classification-project/data_synthesis/src')

import config
from gan_data_loader import PairedECGDataset
from gan_model import Generator, Discriminator

# --- Configuration ---
DATA_ROOT = '/workspace/ecg-classification-project/data_synthesis/output/'
EPOCHS = 100 # GANs need many epochs to converge
LEARNING_RATE = 0.0002
BATCH_SIZE = 16 # Use a smaller batch size for GANs due to memory constraints
L1_LAMBDA = 100 # Weight for the pixel-wise similarity loss

def train_gan():
    """
    Main function to orchestrate the Pix2Pix GAN training process.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Setup Models and Optimizers ---
    # The Generator takes a 3-channel realistic image and outputs a 3-channel clean plot
    generator = Generator(in_channels=3, out_channels=3).to(device)
    # The Discriminator takes the input image and the target/generated image concatenated (6 channels)
    discriminator = Discriminator(in_channels=6).to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    # --- 2. Define Loss Functions ---
    adversarial_loss = nn.BCEWithLogitsLoss() # For the "real" vs "fake" game
    l1_loss = nn.L1Loss() # For pixel-wise similarity

    # --- 3. Load Data ---
    print("Loading paired dataset...")
    dataset = PairedECGDataset(root_dir=DATA_ROOT)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # --- 4. Training Loop ---
    print("Starting GAN training...")
    for epoch in range(EPOCHS):
        for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
            real_input_image = batch["input"].to(device)
            real_target_image = batch["target"].to(device)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Generate a fake image
            fake_image = generator(real_input_image)

            # Real loss: Discriminator sees the real input + real target
            pred_real = discriminator(torch.cat([real_input_image, real_target_image], 1))
            loss_real = adversarial_loss(pred_real, torch.ones_like(pred_real))

            # Fake loss: Discriminator sees the real input + fake target
            pred_fake = discriminator(torch.cat([real_input_image, fake_image.detach()], 1))
            loss_fake = adversarial_loss(pred_fake, torch.zeros_like(pred_fake))
            
            loss_D = (loss_real + loss_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # The generator needs to fool the discriminator
            pred_fake = discriminator(torch.cat([real_input_image, fake_image], 1))
            loss_G_adv = adversarial_loss(pred_fake, torch.ones_like(pred_fake))
            
            # And it needs to be as close as possible to the real target (L1 loss)
            loss_G_l1 = l1_loss(fake_image, real_target_image) * L1_LAMBDA
            
            loss_G = loss_G_adv + loss_G_l1
            loss_G.backward()
            optimizer_G.step()

        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")
        
        # Save some sample images every 5 epochs to see the progress
        if (epoch + 1) % 5 == 0:
            sample_dir = "/workspace/ecg-classification-project/gan_samples"
            os.makedirs(sample_dir, exist_ok=True)
            comparison_grid = torch.cat([real_input_image.data[:4], fake_image.data[:4], real_target_image.data[:4]], dim=0)
            torchvision.utils.save_image(comparison_grid, f"{sample_dir}/epoch_{epoch+1}.png", nrow=4, normalize=True)

    # Save the trained generator model
    os.makedirs("/workspace/ecg-classification-project/models", exist_ok=True)
    torch.save(generator.state_dict(), "/workspace/ecg-classification-project/models/gan_converter_v1.pth")
    print("GAN Generator model saved.")


if __name__ == '__main__':
    train_gan()