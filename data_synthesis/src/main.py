# data_synthesis/src/main.py
import pandas as pd
import numpy as np
import wfdb
import cv2
import os
from tqdm import tqdm

import sys
sys.path.append('/workspace/ecg-classification-project/src')
import config

# *** KEY CHANGE: Import the correct, new function name ***
from synthesis_utils import plot_clean_ecg

# --- Configuration ---
# Let's generate the full dataset now
NUM_SAMPLES_TO_GENERATE = 21799 
OUTPUT_DIR = '/workspace/ecg-classification-project/data_synthesis/output/'

def generate_dataset():
    print("Starting minimalist dataset synthesis...")
    
    output_image_dir = os.path.join(OUTPUT_DIR, 'images')
    output_signal_dir = os.path.join(OUTPUT_DIR, 'signals')
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_signal_dir, exist_ok=True)

    df_meta = pd.read_csv(os.path.join(config.DATA_PATH, config.METADATA_FILE), index_col='ecg_id')
    
    for ecg_id in tqdm(df_meta.index[:NUM_SAMPLES_TO_GENERATE], desc="Generating Samples"):
        filename = os.path.join(config.DATA_PATH, df_meta.loc[ecg_id, 'filename_hr'])
        
        try:
            signal_data, signal_metadata = wfdb.rdsamp(filename)
            leads = signal_metadata['sig_name']
            
            # --- Run Minimalist Synthesis Pipeline ---
            # *** KEY CHANGE: Call the correct function ***
            final_image = plot_clean_ecg(signal_data, leads)
            
            base_filename = f"sample_{ecg_id}"
            
            cv2.imwrite(os.path.join(output_image_dir, f"{base_filename}.png"), final_image)
            np.save(os.path.join(output_signal_dir, f"{base_filename}.npy"), signal_data)
        except Exception as e:
            print(f"\nSkipping sample {ecg_id} due to error: {e}")
            
    print(f"\nSuccessfully generated samples in '{OUTPUT_DIR}'")

if __name__ == '__main__':
    generate_dataset()