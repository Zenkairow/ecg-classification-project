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

# (Generator, Discriminator, Config, and other sections remain the same)
# ...

# --- DATA LOADER (ROBUST VERSION) ---
class ECGPairedDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform
        
        # Build a list of only the files that exist in BOTH directories
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
            return None # Return None to be handled by a collate function

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

# Custom collate function to filter out None values from failed loads
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch) if batch else (None, None)

# --- MAIN TRAINING SCRIPT ---
def main():
    # ... (Setup is the same)
    
    dataset = ECGPairedDataset(input_dir=INPUT_DIR, target_dir=TARGET_DIR, transform=transform_pipeline)
    # Use the custom collate_fn
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        loop = tqdm(loader, leave=True)
        for idx, (x, y) in enumerate(loop):
            # NEW: Check for empty batches
            if x is None:
                continue
            
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # ... (Rest of the training loop is the same)

# (The rest of the script is the same)
# ...

            # Train Discriminator
            y_fake = gen(x)
            D_real = disc(x, y)
            D_real_loss = BCE(D_real, torch.ones_like(D_real))
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

            if idx == 0: # Save one sample at the beginning of each epoch
                y_fake_unnorm = y_fake * 0.5 + 0.5
                torchvision.utils.save_image(y_fake_unnorm, f"{OUTPUT_SAMPLES_DIR}/y_fake_epoch_{epoch}.png")
        
        save_checkpoint(gen, disc, opt_gen, opt_disc, epoch)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Disc Loss: {D_loss.item():.4f}, Gen Loss: {G_loss.item():.4f}")

if __name__ == "__main__":
    main()