import os
import random
from PIL import Image
from tqdm import tqdm
from google.colab import drive

# --- This cell should be run first in Colab ---
# Mount Google Drive to make your files accessible
drive.mount('/content/drive')

# --- CONFIGURATION for Colab ---
# IMPORTANT: Update this path to where you saved your plots in Google Drive
DRIVE_BASE_PATH = '/content/drive/MyDrive/' # Assuming you save it in the main "My Drive" folder
SOURCE_PLOTS_DIR = os.path.join(DRIVE_BASE_PATH, 'data_synthesis/clean_plots_with_grid/')

# We will create and use a backgrounds folder also in your Drive
BACKGROUNDS_DIR = os.path.join(DRIVE_BASE_PATH, 'data_synthesis/background_images/')

# The new images will also be saved back to your Google Drive
OUTPUT_DIR = os.path.join(DRIVE_BASE_PATH, 'data_synthesis/model_1_data/inputs/')

# --- Test Configuration ---
NUM_TEST_IMAGES = 5 

# --- SCRIPT LOGIC ---
def create_test_dataset():
    """
    Generates a small batch of test images for review using data from Google Drive.
    """
    # Create the output directory in Google Drive if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check if the source directory exists
    if not os.path.exists(SOURCE_PLOTS_DIR):
        print(f"❌ ERROR: Source directory not found at '{SOURCE_PLOTS_DIR}'")
        print("Please make sure the path is correct and the folder has finished uploading.")
        return

    plot_files = [f for f in os.listdir(SOURCE_PLOTS_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    background_files = [f for f in os.listdir(BACKGROUNDS_DIR) if f.endswith(('.png', '.jpg', '.jpeg', '.webp'))]

    if not background_files:
        print(f"❌ ERROR: No background images found in {BACKGROUNDS_DIR}. Please add some images to this folder in your Drive.")
        return
    
    num_to_generate = min(NUM_TEST_IMAGES, len(plot_files))
    print(f"🚀 Starting test run. Generating {num_to_generate} images...")

    # Loop through a limited number of plot files for the test
    for filename in tqdm(plot_files[:num_to_generate], desc="Generating Test Images"):
        try:
            plot_path = os.path.join(SOURCE_PLOTS_DIR, filename)
            plot_img = Image.open(plot_path).convert("RGBA")

            random_bg_filename = random.choice(background_files)
            bg_path = os.path.join(BACKGROUNDS_DIR, random_bg_filename)
            bg_img = Image.open(bg_path).convert("RGBA")

            scale = random.uniform(0.75, 0.95)
            new_plot_width = int(bg_img.width * scale)
            
            aspect_ratio = plot_img.height / plot_img.width
            new_plot_height = int(new_plot_width * aspect_ratio)

            plot_resized = plot_img.resize((new_plot_width, new_plot_height), Image.LANCZOS)

            max_x = bg_img.width - new_plot_width
            max_y = bg_img.height - new_plot_height
            
            paste_x = random.randint(0, max(0, max_x))
            paste_y = random.randint(0, max(0, max_y))

            bg_img.paste(plot_resized, (paste_x, paste_y), plot_resized)

            output_path = os.path.join(OUTPUT_DIR, os.path.splitext(filename)[0] + '.jpg')
            bg_img.convert("RGB").save(output_path, 'JPEG')

        except Exception as e:
            print(f"Could not process {filename}. Error: {e}")

    print(f"\n✅ Test run complete. {num_to_generate} images saved to {OUTPUT_DIR}")

# To run this in Colab, you would call the function in a new cell
# create_test_dataset()