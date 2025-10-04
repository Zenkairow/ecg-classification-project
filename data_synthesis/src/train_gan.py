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

# (Generator and Discriminator class definitions remain the same)
class Generator(nn.Module):
    # Your U-Net Generator Architecture
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super().__init__()
        # This is a simplified example, your actual model should be a full U-Net
        self.down1 = nn.Sequential(nn.Conv2d(in_channels, features, 4, 2, 1, bias=False), nn.LeakyReLU(0.2))
        self.down2 = nn.Sequential(nn.Conv2d(features, features*2, 4, 2, 1, bias=False), nn.BatchNorm2d(features*2), nn.LeakyReLU(0.2))
        self.up1 = nn.Sequential(nn.ConvTranspose2d(features*2, features, 4, 2, 1, bias=False), nn.BatchNorm2d(features), nn.ReLU())
        self.final_up = nn.Sequential(nn.ConvTranspose2d(features*2, out_channels, 4, 2, 1), nn.Tanh())
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        u1 = self.up1(d2)
        # In a real U-Net, you would concatenate d1 with u1 here
        return self.final_up(torch.cat([u1, d1], 1)) # Simplified example

class Discriminator(nn.Module):
    # Your PatchGAN Discriminator Architecture
    def __init__(self, in_channels=6, features=[64, 128, 256, 512]):
        super().__init__()
        layers = [nn.Conv2d(in_channels, features[0], 4, 2, 1), nn.LeakyReLU(0.2)]
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(nn.Conv2d(in_channels, feature, 4, 2, 1, bias=False))
            layers.append(nn.BatchNorm2d(feature))
            layers.append(nn.LeakyReLU(0.2))
            in_channels = feature
        layers.append(nn.Conv2d(in_channels, 1, 4, 1, 1))
        self.model = nn.Sequential(*layers)
    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        return self.model(x)
        
# --- CONFIGURATION (remains the same) ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-4
BATCH_SIZE = 4
NUM_EPOCHS = 100
L1_LAMBDA = 100
INPUT_DIR = 'model_1_generated_data/inputs_hyper_realistic/'
TARGET_DIR = 'data_synthesis/output/targets/'
OUTPUT_CHECKPOINT = "models/gan_checkpoint.pth.tar"
OUTPUT_SAMPLES_DIR = "training_samples/"

# --- AUGMENTATIONS (remains the same) ---
transform_pipeline = A.Compose([...]) # The full pipeline code is here

# --- DATA LOADER (remains the same) ---
class ECGPairedDataset(Dataset):
    # ... (Dataset class code remains the same)
    def __init__(self, input_dir, target_dir, transform=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform
        self.input_images = sorted(os.listdir(input_dir))
        # Ensure filenames match between input and target directories
        self.target_images = [f.replace('.jpg', '.png') for f in self.input_images] # Or adjust based on your naming

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, index):
        input_img_name = self.input_images[index]
        target_img_name = self.target_images[index]
        
        input_path = os.path.join(self.input_dir, input_img_name)
        target_path = os.path.join(self.target_dir, target_img_name)

        input_image = np.array(Image.open(input_path).convert("RGB"))
        target_image = np.array(Image.open(target_path).convert("RGB"))

        if self.transform:
            augmented = self.transform(image=input_image)
            input_image = augmented["image"]

        target_transform = A.Compose([
            A.Resize(width=256, height=256),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
            ToTensorV2(),
        ])
        target_image = target_transform(image=target_image)["image"]

        return input_image, target_image

# --- NEW: CHECKPOINT FUNCTIONS ---
def save_checkpoint(gen, disc, opt_gen, opt_disc, epoch, filename=OUTPUT_CHECKPOINT):
    print("=> Saving checkpoint")
    checkpoint = {
        "gen_state_dict": gen.state_dict(),
        "disc_state_dict": disc.state_dict(),
        "opt_gen_state_dict": opt_gen.state_dict(),
        "opt_disc_state_dict": opt_disc.state_dict(),
        "epoch": epoch,
    }
    torch.save(checkpoint, filename)

def load_checkpoint(filename, gen, disc, opt_gen, opt_disc, lr):
    if os.path.exists(filename):
        print("=> Loading checkpoint")
        checkpoint = torch.load(filename, map_location=DEVICE)
        gen.load_state_dict(checkpoint["gen_state_dict"])
        disc.load_state_dict(checkpoint["disc_state_dict"])
        opt_gen.load_state_dict(checkpoint["opt_gen_state_dict"])
        opt_disc.load_state_dict(checkpoint["opt_disc_state_dict"])
        # Set learning rate for optimizers
        for param_group in opt_gen.param_groups:
            param_group["lr"] = lr
        for param_group in opt_disc.param_groups:
            param_group["lr"] = lr
        return checkpoint["epoch"] + 1
    return 0

# --- MAIN TRAINING SCRIPT ---
def main():
    print(f"Starting training on device: {DEVICE}")
    os.makedirs(OUTPUT_SAMPLES_DIR, exist_ok=True)
    
    gen = Generator(in_channels=3, features=64).to(DEVICE)
    disc = Discriminator(in_channels=6).to(DEVICE)
    
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()
    
    # NEW: Load checkpoint if it exists
    start_epoch = load_checkpoint(OUTPUT_CHECKPOINT, gen, disc, opt_gen, opt_disc, LEARNING_RATE)
    
    dataset = ECGPairedDataset(input_dir=INPUT_DIR, target_dir=TARGET_DIR, transform=transform_pipeline)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        loop = tqdm(loader, leave=True)
        for idx, (x, y) in enumerate(loop):
            # ... (training loop remains the same)
            x, y = x.to(DEVICE), y.to(DEVICE)
            # ... (Discriminator training)
            # ... (Generator training)

            if idx % 500 == 0:
                y_fake_unnorm = y_fake * 0.5 + 0.5
                torchvision.utils.save_image(y_fake_unnorm, f"{OUTPUT_SAMPLES_DIR}/y_fake_{epoch}_{idx}.png")
        
        # NEW: Save checkpoint at the end of each epoch
        save_checkpoint(gen, disc, opt_gen, opt_disc, epoch)
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Disc Loss: {D_loss:.4f}, Gen Loss: {G_loss:.4f}")

if __name__ == "__main__":
    main()