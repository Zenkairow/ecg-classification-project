# data_synthesis/src/train_gan.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import torchvision
import numpy as np
import cv2

# Import our custom classes
import sys
sys.path.append('/workspace/project/src')
sys.path.append('/workspace/project/data_synthesis/src')
import config
from gan_data_loader import PairedECGDataset
from gan_model import Generator, Discriminator
from synthesis_utils import plot_ecg_on_paper # Import our high-quality plotter

# --- Configuration ---
DATA_ROOT = '/workspace/project/data_synthesis/output/'
ECG_PAPER_PATH = '/workspace/project/data_synthesis/background_images/ecg_paper.png'
EPOCHS = 50
LEARNING_RATE_G = 0.0002
LEARNING_RATE_D = 0.0002
BATCH_SIZE = 16
L1_LAMBDA = 100
IMG_HEIGHT = 256
IMG_WIDTH = 512

def train_gan():
    """
    Main function to orchestrate the GAN training process with high-quality targets.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Setup Models and Optimizers ---
    generator = Generator(in_channels=3, out_channels=3).to(device) # Outputting a 3-channel (RGB) image
    discriminator = Discriminator(in_channels=3 + 3).to(device) # Discriminator sees input image + target/generated image
    
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE_G, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE_D, betas=(0.5, 0.999))

    # --- 2. Define Loss Functions ---
    adversarial_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    # --- 3. Load Data ---
    print("Loading paired dataset...")
    dataset = PairedECGDataset(root_dir=DATA_ROOT)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- 4. Training Loop ---
    print("Starting GAN training...")
    for epoch in range(EPOCHS):
        for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
            real_input_image = batch["image"].to(device)
            ground_truth_signal = batch["signal"] # Keep signal on CPU for plotting

            # --- KEY CHANGE: Generate High-Quality Target Image ---
            target_images = []
            for j in range(ground_truth_signal.shape[0]): # Loop through batch
                # Assuming signal is (channels, length) and we need (length, channels)
                signal_np = ground_truth_signal[j].numpy().T
                leads = [str(k) for k in range(signal_np.shape[1])] # Dummy leads
                
                # Use our high-quality plotter
                target_img_np = plot_ecg_on_paper(signal_np, leads, ECG_PAPER_PATH)
                target_img_np = cv2.resize(target_img_np, (IMG_WIDTH, IMG_HEIGHT))
                target_img_tensor = transforms.ToTensor()(target_img_np).to(device)
                target_images.append(target_img_tensor)
            
            real_target_image = torch.stack(target_images)
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            fake_image = generator(real_input_image)
            
            # Real loss
            pred_real = discriminator(torch.cat([real_input_image, real_target_image], 1))
            loss_real = adversarial_loss(pred_real, torch.ones_like(pred_real))

            # Fake loss
            pred_fake = discriminator(torch.cat([real_input_image, fake_image.detach()], 1))
            loss_fake = adversarial_loss(pred_fake, torch.zeros_like(pred_fake))
            
            loss_D = (loss_real + loss_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            pred_fake = discriminator(torch.cat([real_input_image, fake_image], 1))
            loss_G_adv = adversarial_loss(pred_fake, torch.ones_like(pred_fake))
            loss_G_l1 = l1_loss(fake_image, real_target_image) * L1_LAMBDA
            
            loss_G = loss_G_adv + loss_G_l1
            loss_G.backward()
            optimizer_G.step()

        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")
        
        if (epoch + 1) % 5 == 0:
            sample_dir = "/workspace/project/gan_samples" # Use absolute path
            os.makedirs(sample_dir, exist_ok=True)
            # Save the real input, the fake output, and the real target for comparison
            comparison_grid = torch.cat([real_input_image.data[:4], fake_image.data[:4], real_target_image.data[:4]], dim=0)
            torchvision.utils.save_image(comparison_grid, f"{sample_dir}/epoch_{epoch+1}.png", nrow=4, normalize=True)

    os.makedirs("/workspace/project/models", exist_ok=True)
    torch.save(generator.state_dict(), "/workspace/project/models/gan_generator_v1.pth")
    print("GAN Generator model saved.")


if __name__ == '__main__':
    train_gan()