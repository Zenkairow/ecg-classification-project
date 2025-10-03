import cv2
import albumentations as A
import numpy as np
from PIL import Image

# --- 1. Define the "On-the-Fly" Augmentations with MORE randomness ---
transform = A.Compose([
    A.ShiftScaleRotate(
        shift_limit=0.04,      
        scale_limit=0.07,      
        rotate_limit=5,        # Increased rotation limit
        border_mode=cv2.BORDER_CONSTANT,
        value=(255, 255, 255), 
        p=0.9
    ),
    
    A.Perspective(scale=(0.03, 0.08), pad_mode=cv2.BORDER_CONSTANT, pad_val=(255, 255, 255), p=0.8),
    
    # Increased blur
    A.GaussianBlur(blur_limit=(5, 11), p=0.7), 
    A.MotionBlur(blur_limit=(5, 11), p=0.5), 
    
    # Add camera noise/grain (stronger)
    A.GaussNoise(var_limit=(30.0, 80.0), p=0.9), 
    
    # Randomly change brightness and contrast (stronger)
    A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.9), 
    
    # NEW: Add a warping/distortion effect to simulate crinkled paper
    A.GridDistortion(p=0.5),
    
    # Simulate low bit-depth or old scanning artifacts
    A.Posterize(num_bits=(6, 4), p=0.3),
    
    # Simulate compression artifacts
    A.ImageCompression(quality_lower=40, quality_upper=70, p=0.6), # Increased compression
])


# --- 2. Load Your Clean Input Image ---
try:
    image_pil = Image.open("sample_1.jpg")
    image_numpy = np.array(image_pil)
    print("✅ Successfully loaded sample_1.jpg")
except FileNotFoundError:
    print("❌ ERROR: 'sample_1.jpg' not found. Make sure it's in the same folder as the script.")
    exit()

# --- 3. Apply the Random Augmentations ---
transformed = transform(image=image_numpy)
transformed_image = transformed["image"]
print("✨ Augmentations applied successfully.")

# --- 4. Save the New, Messy Image ---
output_filename = "augmented_sample_v3.jpg"
cv2.imwrite(output_filename, transformed_image)
print(f"✅ New, more distorted image saved as '{output_filename}'.")