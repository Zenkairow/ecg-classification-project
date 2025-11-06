import cv2
import numpy as np
import os
from tqdm import tqdm
import glob

# --- CONFIGURATION ---
INPUT_DIR = 'data_synthesis/output/targets/'
OUTPUT_DIR = 'data_synthesis/output/test_signal_targets/'
LIMIT = 2
# ---------------------

def create_signal_only_image(img_path, output_path):
    """
    Loads a clean ECG plot, isolates all BLACK elements (signal + text)
    using color-based masking, and saves them on a PURE WHITE background.
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}")
            return

        # --- Create a pure white background ---
        white_bg = np.ones_like(img, dtype=np.uint8) * 255

        # --- Convert to HSV color space ---
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # --- Define color range for BLACK ---
        # We define "black" as pixels with very low Value (brightness)
        # Hue and Saturation can be anything
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50]) # Max Hue, Max Sat, Low Val

        # Create the mask for black pixels
        mask = cv2.inRange(hsv, lower_black, upper_black)
        
        # --- LOGIC FIX ---
        # Convert single-channel mask back to 3 channels
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Where the mask is 255 (black), draw black.
        # Everywhere else, draw white.
        signal_only_img = np.where(mask_3ch == 255, (0, 0, 0), white_bg)
        
        cv2.imwrite(output_path, signal_only_img)

    except Exception as e:
        print(f"Error processing {img_path}: {e}")

def main():
    print(f"--- Starting sample generation (2 samples) [v3 - HSV Masking] ---")
    print(f"Input dir: {INPUT_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    target_files = sorted(glob.glob(os.path.join(INPUT_DIR, '*.png')))
    
    if not target_files:
        print(f"Error: No .png files found in {INPUT_DIR}")
        return

    files_to_process = target_files[:LIMIT]
    
    for filename in tqdm(files_to_process, desc="Generating Samples"):
        output_filename = os.path.basename(filename)
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        create_signal_only_image(filename, output_path)
        
    print(f"Done. Generated {len(files_to_process)} test samples in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()