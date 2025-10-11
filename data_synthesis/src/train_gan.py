import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from tqdm import tqdm
import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torchvision

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- NEW, CORRECT MODEL DEFINITIONS ---

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        self.down1 = Block(features, features * 2, down=True, act="leaky", use_dropout=False)
        self.down2 = Block(features * 2, features * 4, down=True, act="leaky", use_dropout=False)
        self.down3 = Block(features * 4, features * 8, down=True, act="leaky", use_dropout=False)
        self.down4 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.down5 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.down6 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU()
        )

        self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)
        self.up2 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.up3 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.up4 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=False)
        self.up5 = Block(features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False)
        self.up6 = Block(features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False)
        self.up7 = Block(features * 2 * 2, features, down=False, act="relu", use_dropout=False)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        return self.final_up(torch.cat([up7, d1], 1))

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels * 2, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(CNNBlock(in_channels, feature, stride=1 if feature == features[-1] else 2))
            in_channels = feature
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"))
        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        return self.model(x)

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-4
BATCH_SIZE = 4
NUM_EPOCHS = 100
L1_LAMBDA = 500 # Drastically increased for stronger reconstruction 
NUM_WORKERS = 4
INPUT_DIR = 'model_1_generated_data/inputs_hyper_realistic/'
TARGET_DIR = 'data_synthesis/output/targets/'
OUTPUT_CHECKPOINT = "models/gan_checkpoint.pth.tar"
OUTPUT_SAMPLES_DIR = "training_samples/"

# --- AUGMENTATIONS (SANITY CHECK VERSION) ---
# This temporary version disables ALL augmentations except for the essentials.
transform_pipeline = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ],
)

# --- DATA LOADER (ROBUST VERSION) ---
class ECGPairedDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform
        input_filenames = {os.path.splitext(f)[0] for f in os.listdir(input_dir)}
        target_filenames = {os.path.splitext(f)[0] for f in os.listdir(target_dir)}
        valid_filenames = sorted(list(input_filenames.intersection(target_filenames)))
        self.image_pairs = [(f + ".jpg", f + ".png") for f in valid_filenames]
        print(f"Found {len(self.image_pairs)} matching image pairs.")
    def __len__(self):
        return len(self.image_pairs)
    def __getitem__(self, index):
        input_img_name, target_img_name = self.image_pairs[index]
        input_path = os.path.join(self.input_dir, input_img_name)
        target_path = os.path.join(self.target_dir, target_img_name)
        try:
            input_image = np.array(Image.open(input_path).convert("RGB"))
            target_image = np.array(Image.open(target_path).convert("RGB"))
        except Exception as e:
            print(f"Error loading images: {input_img_name}, {target_img_name}. Error: {e}")
            return None
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

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch) if batch else (None, None)

# --- CHECKPOINT FUNCTIONS ---
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
        for param_group in opt_gen.param_groups: param_group["lr"] = lr
        for param_group in opt_disc.param_groups: param_group["lr"] = lr
        print(f"=> Resuming from epoch {checkpoint['epoch'] + 1}")
        return checkpoint["epoch"] + 1
    return 0

# --- MAIN TRAINING SCRIPT ---
def main():
    print(f"Starting training on device: {DEVICE}")
    os.makedirs(OUTPUT_SAMPLES_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_CHECKPOINT), exist_ok=True)
    gen = Generator(in_channels=3).to(DEVICE)
    disc = Discriminator(in_channels=3).to(DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE / 10, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()
    start_epoch = load_checkpoint(OUTPUT_CHECKPOINT, gen, disc, opt_gen, opt_disc, LEARNING_RATE)
    # --- MODIFICATION FOR SANITY CHECK ---
    print("--- RUNNING IN SANITY CHECK MODE ---")
    # First, we load the full dataset definition
    full_dataset = ECGPairedDataset(input_dir=INPUT_DIR, target_dir=TARGET_DIR, transform=transform_pipeline)
    
    # Now, we create a tiny subset containing only the first 4 images
    sanity_check_subset = torch.utils.data.Subset(full_dataset, range(4))
    
    # Finally, we create the loader with shuffling turned OFF
    loader = DataLoader(sanity_check_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
    # --- END MODIFICATION ---
    for epoch in range(start_epoch, NUM_EPOCHS):
        loop = tqdm(loader, leave=True)
        for idx, batch_data in enumerate(loop):
            if not batch_data or len(batch_data) < 2:
                continue
            x, y = batch_data
            if x is None:
                continue
            x, y = x.to(DEVICE), y.to(DEVICE)
            # Train Discriminator
            y_fake = gen(x)
            D_real = disc(x, y)
            # Use 0.9 for real labels instead of 1.0
            D_real_loss = BCE(D_real, torch.full_like(D_real, 0.9, device=DEVICE))
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
            loop.set_postfix(D_real=torch.sigmoid(D_real).mean().item(), D_fake=torch.sigmoid(D_fake).mean().item())
            # Save only the first image from the batch for clarity
            if idx == 0:
                y_fake_unnorm = y_fake[0:1] * 0.5 + 0.5 # Take only the first image
                torchvision.utils.save_image(y_fake_unnorm, f"{OUTPUT_SAMPLES_DIR}/y_fake_epoch_{epoch}.png")
        save_checkpoint(gen, disc, opt_gen, opt_disc, epoch)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Disc Loss: {D_loss.item():.4f}, Gen Loss: {G_loss.item():.4f}")

if __name__ == "__main__":
    main()