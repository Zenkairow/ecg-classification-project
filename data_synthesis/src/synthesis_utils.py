# data_synthesis/src/synthesis_utils.py
import matplotlib.pyplot as plt
import numpy as np
import cv2
import io

def plot_clean_ecg(signal_data, leads):
    """
    Plots a clean, high-fidelity 12-lead ECG on a plain white background,
    with lead labels and separator lines.
    """
    fig, axes = plt.subplots(len(leads), 1, figsize=(15, 10))
    fig.patch.set_facecolor('white')

    for i, lead_name in enumerate(leads):
        # Plot the signal in black
        axes[i].plot(signal_data[:, i], color='black', linewidth=0.75)
        
        # --- KEY CHANGE 1: Add the lead label text ---
        # Position the text on the left side of the plot
        axes[i].text(-0.01, 0.5, lead_name, transform=axes[i].transAxes, 
                     ha='right', va='center', fontsize=10, weight='bold')

        # --- KEY CHANGE 2: Add a faint separator line ---
        if i < len(leads) - 1:
            axes[i].axhline(y=np.min(signal_data[:, i]) - 0.5, color='lightgray', linestyle='--', linewidth=0.5)

        # Remove all decorations: axes, ticks, grid, and borders
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['bottom'].set_visible(False)
        axes[i].spines['left'].set_visible(False)
        axes[i].set_facecolor('white')

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0, hspace=0)
    
    # Convert plot to an image in memory
    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor='none')
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    plt.close(fig)
    return img