# data_synthesis/src/generate_targets.py
import numpy as np
import cv2
import re
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import io

# Add the main project's src folder to the path to find the config file
import sys
sys.path.append('/workspace/project/src')
import config

def plot_clinically_perfect_ecg(signal_data, leads, sampling_rate):
    """
    Plots a clinically accurate, high-definition 12-lead ECG on a white background
    with precise grid alignment, standard ECG scaling, lead labels, and a time axis.
    """
    fig, axes = plt.subplots(len(leads), 1, figsize=(20, 15), dpi=300)
    fig.patch.set_facecolor('white') 

    duration_s = signal_data.shape[0] / sampling_rate
    time_axis_s = np.arange(signal_data.shape[0]) / sampling_rate

    for i, lead_name in enumerate(leads):
        axes[i].plot(time_axis_s, signal_data[:, i], color='black', linewidth=0.6)
        
        axes[i].text(-0.02, 0.5, lead_name, transform=axes[i].transAxes, 
                     ha='right', va='center', fontsize=10, weight='bold', color='black')

        # --- Grid Configuration ---
        # Set common limits for all leads for consistency
        y_min, y_max = -2.0, 2.0
        axes[i].set_ylim(y_min, y_max)
        axes[i].set_xlim(0, duration_s)
        
        # Major grid (0.2s or 0.5mV) - dark red
        axes[i].xaxis.set_major_locator(MultipleLocator(0.2))
        axes[i].yaxis.set_major_locator(MultipleLocator(0.5))
        axes[i].grid(which='major', linestyle='-', linewidth='0.5', color='red', alpha=0.9)
        
        # Minor grid (0.04s or 0.1mV) - light red
        axes[i].xaxis.set_minor_locator(MultipleLocator(0.04))
        axes[i].yaxis.set_minor_locator(MultipleLocator(0.1))
        axes[i].grid(which='minor', linestyle='-', linewidth='0.25', color='red', alpha=0.4)

        # Remove Y-axis labels and spines
        axes[i].set_yticks([])
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['bottom'].set_visible(False)
        axes[i].spines['left'].set_visible(False)
        axes[i].set_facecolor('white')

        # --- KEY CHANGE: Add time axis ticks to ALL plots ---
        axes[i].set_xticks(np.arange(0, int(duration_s) + 1, 1))
        # But only show the number labels on the bottom-most plot
        if i < len(leads) - 1:
            axes[i].set_xticklabels([])
        else:
            axes[i].tick_params(axis='x', labelsize=10, colors='black')
            axes[i].set_xlabel('Time (seconds)', fontsize=12, color='black')
            
    plt.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.05, hspace=0)
    
    # Convert plot to a high-quality image in memory
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    plt.close(fig)
    return img

# --- Main script ---
def generate_target_images():
    INPUT_DIR = 'data_synthesis/output/signals'
    OUTPUT_DIR = 'data_synthesis/output/targets'
    SAMPLING_RATE = 500
    
    print("Starting high-quality target image generation...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

        

    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split(r'(\d+)', s)]

    signal_files = sorted(
        [f for f in os.listdir(INPUT_DIR) if f.endswith('.npy')],
        key=natural_sort_key
    )

    
    # Let's generate a small batch of 5 first to test the new quality
    for signal_file in tqdm(signal_files, desc="Generating Target Images"):
        signal_path = os.path.join(INPUT_DIR, signal_file)
        try:
            signal_data = np.load(signal_path)
            leads = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
            target_image = plot_clinically_perfect_ecg(signal_data, leads, SAMPLING_RATE)
            base_filename = signal_file.replace('.npy', '.png')
            cv2.imwrite(os.path.join(OUTPUT_DIR, base_filename), target_image)
        except Exception as e:
            print(f"\nCould not process sample {signal_file}. Error: {e}")
            
    print(f"\nSuccessfully generated all sample target images in '{OUTPUT_DIR}'")

if __name__ == '__main__':
    generate_target_images()