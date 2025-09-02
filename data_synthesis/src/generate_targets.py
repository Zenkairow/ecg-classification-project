# data_synthesis/src/generate_targets.py
import pandas as pd
import numpy as np
import wfdb
import cv2
import os
from tqdm import tqdm

# *** KEY CHANGE: Add the absolute path to our main src folder ***
import sys
sys.path.append('/workspace/ecg-classification-project/src')
import config

from synthesis_utils import plot_clean_ecg

# --- Configuration ---
NUM_SAMPLES_TO_GENERATE = 21799
OUTPUT_DIR = '/workspace/ecg-classification-project/data_synthesis/output/targets'

def generate_target_images():
    print("Starting target image generation...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_meta = pd.read_csv(os.path.join(config.DATA_PATH, config.METADATA_FILE), index_col='ecg_id')
    
    for ecg_id in tqdm(df_meta.index[:NUM_SAMPLES_TO_GENERATE], desc="Generating Target Images"):
        filename = os.path.join(config.DATA_PATH, df_meta.loc[ecg_id, 'filename_hr'])
        try:
            signal_data, signal_metadata = wfdb.rdsamp(filename)
            leads = signal_metadata['sig_name']
            target_image = plot_clean_ecg(signal_data, leads)
            base_filename = f"sample_{ecg_id}.png"
            cv2.imwrite(os.path.join(OUTPUT_DIR, base_filename), target_image)
        except Exception as e:
            print(f"\nCould not process sample {ecg_id}. Error: {e}")
            
    print(f"\nSuccessfully generated target images in '{OUTPUT_DIR}'")

if __name__ == '__main__':
    generate_target_images()