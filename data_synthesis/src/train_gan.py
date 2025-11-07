import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from tqdm import tqdm
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torchvision # <--- Make sure this import is present
from pytorch_msssim import SSIM

# --- NEW: Import the Real-ESRGAN architecture ---
from basicsr.archs.rrdbnet_arch import RRDBNet

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-5  # CRITICAL: Use a very low learning rate for fine-tuning
BATCH_SIZE = 1        # Reduced for memory
NUM_EPOCHS = 50       # We don't need 100 epochs when fine-tuning
SSIM_LAMBDA = 250     # Weight for the structural similarity loss
NUM_WORKERS = 4

# --- PATHS ---
INPUT_DIR = 'model_1_generated_data/inputs_hyper_realistic/'
TARGET_DIR = 'data_synthesis/output/signal_targets_final/'
OUTPUT_CHECKPOINT = "models/realesrgan_finetune_checkpoint.pth.tar" # New checkpoint name
PRETRAINED_MODEL_PATH = "models/RealESRGAN_x4plus.pth" # Path to our downloaded model
OUTPUT_SAMPLES_DIR = "training_samples/"

# --- AUGMENTATIONS (HEAVY RESTORATION TASK) ---
transform_pipeline = A.Compose(
    [
        # --- Heavy augmentations to create a "damaged" photo ---

        A.GaussNoise(std_range=(0.039, 0.196), p=0.8),

        A.GaussianBlur(blur_limit=(3, 7), p=0.8),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),

        # Use the recommended 'Affine' transform
        A.Affine(scale=0.95, translate_percent=0.05, rotate=15, p=0.7), 

        # Corrected: 'alpha_affine' is no longer valid
        A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05),
        # --------------------------------------------------------

        A.Resize(width=128, height=128), # Keep 128x128 for memory
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

        # Apply transform to input image
        if self.transform:
            augmented = self.transform(image=input_image)
            input_image = augmented["image"]

        # Apply a separate, simple transform to the target image (at 128x128)
        target_transform = A.Compose([
            A.Resize(width=128, height=128), # Reduced resolution
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
            ToTensorV2(),
        ])
        target_image = target_transform(image=target_image)["image"]
        return input_image, target_image

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch) if batch else (None, None)

# --- CHECKPOINT FUNCTIONS (Updated) ---
def save_checkpoint(model, optimizer, epoch, filename=OUTPUT_CHECKPOINT):
    print("=> Saving checkpoint")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(checkpoint, filename)

def load_checkpoint(filename, model, optimizer, lr):
    if os.path.exists(filename):
        print("=> Loading checkpoint")
        checkpoint = torch.load(filename, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print(f"=> Resuming from epoch {checkpoint['epoch'] + 1}")
        return checkpoint["epoch"] + 1
    else:
        print("=> No checkpoint found, loading pre-trained Real-ESRGAN model.")
        # Load the pre-trained model weights
        pretrained_weights = torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE)
        # Handle potential key mismatch if the state_dict is nested
        if "params_ema" in pretrained_weights:
            model.load_state_dict(pretrained_weights["params_ema"])
        elif "params" in pretrained_weights:
             model.load_state_dict(pretrained_weights["params"])
        else:
             model.load_state_dict(pretrained_weights)
        print("=> Loaded pre-trained RealESRGAN_x4plus.pth successfully.")
        return 0

# --- MAIN FINE-TUNING SCRIPT ---
def main():
    print(f"Starting fine-tuning on device: {DEVICE}")
    os.makedirs(OUTPUT_SAMPLES_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_CHECKPOINT), exist_ok=True)

    # Initialize the RRDBNet Generator
    model_g = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4).to(DEVICE)

    optimizer_g = optim.Adam(model_g.parameters(), lr=LEARNING_RATE)

    # Loss functions
    loss_ssim = SSIM(data_range=1.0, size_average=True, channel=3)
    loss_l1 = nn.L1Loss()

    start_epoch = load_checkpoint(OUTPUT_CHECKPOINT, model_g, optimizer_g, LEARNING_RATE)

    dataset = ECGPairedDataset(input_dir=INPUT_DIR, target_dir=TARGET_DIR, transform=transform_pipeline)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)

    print("--- Starting fine-tuning ---")
    for epoch in range(start_epoch, NUM_EPOCHS):
        model_g.train()
        loop = tqdm(loader, leave=True)
        total_loss = 0.0

        for idx, batch_data in enumerate(loop):
            if not batch_data or len(batch_data) < 2:
                continue
            x, y = batch_data
            if x is None:
                continue
            x, y = x.to(DEVICE), y.to(DEVICE)

            # --- Forward Pass ---
            y_fake = model_g(x) # Output will be 512x512

            # --- Calculate High-Fidelity Loss ---
            # Un-normalize images from [-1, 1] to [0, 1] for SSIM
            y_fake_for_ssim = (y_fake + 1) / 2
            y_for_ssim = (y + 1) / 2 # Target y is already 128x128

            # --- CRITICAL FIX: Downscale model output (512x512) to match target (128x128) ---
            y_fake_resized_for_loss = torchvision.transforms.functional.resize(
                y_fake_for_ssim,
                [128, 128], # Target size (height, width)
                antialias=True # Use antialiasing for better quality resize
            )
            # --- END FIX ---

            # Calculate SSIM loss using the *resized* 128x128 output
            ssim_loss_val = (1 - loss_ssim(y_fake_resized_for_loss, y_for_ssim)) * SSIM_LAMBDA

            # Calculate L1 loss using the *resized* 128x128 output vs 128x128 target
            l1_loss_val = loss_l1(y_fake_resized_for_loss, y) * (1.0 - (SSIM_LAMBDA / 1000.0)) # L1 gets remaining weight

            # Combine the two losses
            g_loss = ssim_loss_val + l1_loss_val

            # --- Backward Pass ---
            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()

            total_loss += g_loss.item()
            loop.set_postfix(Epoch=epoch, G_Loss=g_loss.item())

            # Save a sample image (Save the HIGH-RES 512x512 output)
            if idx == 0:
                y_fake_unnorm = y_fake[0:1] * 0.5 + 0.5 # Take only the first image
                torchvision.utils.save_image(y_fake_unnorm, f"{OUTPUT_SAMPLES_DIR}/y_fake_epoch_{epoch}.png")

        avg_loss = total_loss / len(loader)
        save_checkpoint(model_g, optimizer_g, epoch)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Average Gen Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    main()