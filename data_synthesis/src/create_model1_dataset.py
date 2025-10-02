import os
import random
from PIL import Image, ImageFilter
from tqdm import tqdm
from google.colab import drive

# Mount Google Drive to make your files accessible
drive.mount('/content/drive')

# --- CONFIGURATION ---
BASE_DRIVE_PATH = '/content/drive/Othercomputers/My Laptop/output'
SOURCE_PLOTS_DIR = os.path.join(BASE_DRIVE_PATH, 'targets/')
BACKGROUNDS_DIR = '/content/ecg-classification-project/data_synthesis/background_images/'
OUTPUT_DIR = '/content/drive/MyDrive/model_1_generated_data/inputs_hyper_realistic/' # The final, official output folder

# --- HELPER FUNCTION (Unchanged) ---
def add_shadow(img):
    shadow_offset = (15, 15)
    shadow_color = (0, 0, 0, 180)
    shadow = Image.new('RGBA', img.size, (0, 0, 0, 0))
    alpha_mask = Image.new('L', img.size, 255)
    shadow.paste(shadow_color, (0,0), alpha_mask)
    shadow = shadow.filter(ImageFilter.GaussianBlur(10))
    final_canvas = Image.new('RGBA', (img.width + shadow_offset[0], img.height + shadow_offset[1]), (0,0,0,0))
    final_canvas.paste(shadow, shadow_offset, shadow)
    final_canvas.paste(img, (0,0), img)
    return final_canvas

# --- FINAL FULL-SCALE GENERATION SCRIPT ---
def create_final_dataset():
    """
    Generates the full, hyper-realistic input dataset for Model 1.
    Checks for the output directory only once at the start for efficiency.
    """
    # --- OPTIMIZED: Check for the output directory ONCE at the start ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(SOURCE_PLOTS_DIR):
        print(f"❌ ERROR: Source directory not found at '{SOURCE_PLOTS_DIR}'")
        return

    plot_files = [f for f in os.listdir(SOURCE_PLOTS_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    background_files = [f for f in os.listdir(BACKGROUNDS_DIR) if f.endswith(('.png', '.jpg', '.jpeg', '.webp'))]

    if not background_files:
        print(f"❌ ERROR: No background images found.")
        return
    
    random.shuffle(background_files)
    
    print(f"🚀 Starting FINAL hyper-realistic dataset generation for {len(plot_files)} images...")
    print(f"   This process will take many hours. Images will be saved to '{OUTPUT_DIR}'")

    for i, filename in enumerate(tqdm(plot_files, desc="Generating Dataset")):
        try:
            # --- Image processing steps (unchanged) ---
            plot_path = os.path.join(SOURCE_PLOTS_DIR, filename)
            plot_img = Image.open(plot_path).convert("RGBA")
            bg_filename = background_files[i % len(background_files)]
            bg_path = os.path.join(BACKGROUNDS_DIR, bg_filename)
            bg_img = Image.open(bg_path).convert("RGB")
            plot_with_shadow = add_shadow(plot_img)
            if bg_img.width < plot_with_shadow.width or bg_img.height < plot_with_shadow.height:
                new_bg_width = int(plot_with_shadow.width * 1.1)
                new_bg_height = int(plot_with_shadow.height * 1.1)
                bg_img = bg_img.resize((new_bg_width, new_bg_height), Image.LANCZOS)
            paste_x = (bg_img.width - plot_with_shadow.width) // 2
            paste_y = (bg_img.height - plot_with_shadow.height) // 2
            bg_img.paste(plot_with_shadow, (paste_x, paste_y), plot_with_shadow)
            
            output_path = os.path.join(OUTPUT_DIR, os.path.splitext(filename)[0] + '.jpg')
            bg_img.save(output_path, 'JPEG', quality=95)

        except Exception as e:
            print(f"\nCould not process {filename}. Error: {e}")

    print(f"\n✅ Full hyper-realistic dataset generation complete. {len(plot_files)} images have been saved.")

# Execute the final generation function
create_final_dataset()