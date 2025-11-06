import cv2
import numpy as np
import os
from tqdm import tqdm
import glob

# --- CONFIGURATION ---
# 1. Our source of perfect, clean plots (grid + signal)
INPUT_DIR = 'data_synthesis/output/targets/'

# 2. Our new test folder
OUTPUT_DIR = 'data_synthesis/output/test_signal_targets/'

# 3. How many samples to generate for this test
LIMIT = 2
# ---------------------

def create_signal_only_image(img_path, output_path):
    """
    Loads a clean ECG plot, isolates the signal, and saves it
    on a pure white background.
    """
    try:
        # Load the original clean plot
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}")
            return

        # Create a pure white background of the same size
        white_bg = np.ones_like(img, dtype=np.uint8) * 255
        
        # Convert to grayscale to find the signal
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # We define the "signal" as any pixel that is not white or red.
        # A threshold of 200 (out of 255) should safely isolate
        # only the dark black signal.
        # THRESH_BINARY_INV makes the signal white (255) and everything else black (0)
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Convert the single-channel mask back to 3 channels
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Use the mask to copy the signal from the original image
        # onto the pure white background.
        signal_only_img = np.where(mask_3ch == 255, img, white_bg)
        
        # Save the new signal-only image
        cv2.imwrite(output_path, signal_only_img)

    except Exception as e:
        print(f"Error processing {img_path}: {e}")

def main():
    print(f"--- Starting sample generation (2 samples) ---")
    print(f"Input dir: {INPUT_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Use glob to find all .png files
    target_files = sorted(glob.glob(os.path.join(INPUT_DIR, '*.png')))
    
    if not target_files:
        print(f"Error: No .png files found in {INPUT_DIR}")
        return

    # Get just the first few files to process
    files_to_process = target_files[:LIMIT]
    
    for filename in tqdm(files_to_process, desc="Generating Samples"):
        # We pass the full path to the function now
        output_filename = os.path.basename(filename)
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        create_signal_only_image(filename, output_path)
        
    print(f"Done. Generated {len(files_to_process)} test samples in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()