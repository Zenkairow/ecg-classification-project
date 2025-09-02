# data_synthesis/src/generate_targets.py
import pandas as pd
import numpy as np
import wfdb
import cv2
import os
from tqdm import tqdm

import sys
sys.path.append('/workspace/project/src')
import config

# We need the minimalist plotting function
from synthesis_utils import plot_clean_ecg

# --- Configuration ---
NUM_SAMPLES_TO_GENERATE = 21799 # Let's generate all of them
OUTPUT_DIR = '/workspace/project/data_synthesis/output/targets' # A new folder for our targets

def generate_target_images():
    """
    Main function to generate the clean, minimalist target images.
    """
    print("Starting target image generation...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_meta = pd.read_csv(os.path.join(config.DATA_PATH, config.METADATA_FILE), index_col='ecg_id')
    
    for ecg_id in tqdm(df_meta.index[:NUM_SAMPLES_TO_GENERATE], desc="Generating Target Images"):
        filename = os.path.join(config.DATA_PATH, df_meta.loc[ecg_id, 'filename_hr'])
        
        try:
            signal_data, signal_metadata = wfdb.rdsamp(filename)
            leads = signal_metadata['sig_name']
            
            # Use our minimalist plotting function
            target_image = plot_clean_ecg(signal_data, leads)
            
            # Save the clean target image
            base_filename = f"sample_{ecg_id}.png"
            cv2.imwrite(os.path.join(OUTPUT_DIR, base_filename), target_image)

        except Exception as e:
            print(f"\nCould not process sample {ecg_id}. Error: {e}")
        
    print(f"\nSuccessfully generated {NUM_SAMPLES_TO_GENERATE} target images in '{OUTPUT_DIR}'")

if __name__ == '__main__':
    generate_target_images()