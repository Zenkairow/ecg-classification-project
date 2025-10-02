import os
import random
from PIL import Image, ImageFilter, ImageFile
from tqdm import tqdm

# Allows loading of potentially truncated image files to prevent errors
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- FINAL SERVER PATHS (RELATIVE TO PROJECT ROOT) ---
SOURCE_PLOTS_DIR = 'output/targets/'
BACKGROUNDS_DIR = 'data_synthesis/background_images/'
OUTPUT_DIR = 'model_1_generated_data/inputs_hyper_realistic/'

def add_shadow(img):
    """Adds a soft drop shadow to an image."""
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

def generate_full_dataset():
    """Generates the full hyper-realistic dataset on the server."""
    print(f"--- Starting full dataset generation. ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(SOURCE_PLOTS_DIR):
        print(f"❌ ERROR: Source plots directory not found at '{SOURCE_PLOTS_DIR}'")
        return
    if not os.path.exists(BACKGROUNDS_DIR):
        print(f"❌ ERROR: Backgrounds directory not found at '{BACKGROUNDS_DIR}'")
        return

    plot_filenames = [f for f in os.listdir(SOURCE_PLOTS_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    plot_files = sorted(plot_filenames, key=lambda f: int(f.replace('sample_', '').split('.')[0]))
    background_files = sorted([f for f in os.listdir(BACKGROUNDS_DIR) if f.endswith(('.png', '.jpg', '.jpeg', '.webp'))])

    print(f"Found {len(plot_files)} plots. Output will be saved to: {OUTPUT_DIR}")

    for i, filename in enumerate(tqdm(plot_files, desc="Generating Full Dataset")):
        try:
            plot_path = os.path.join(SOURCE_PLOTS_DIR, filename)
            plot_img = Image.open(plot_path).convert("RGBA")

            bg_index = i % len(background_files)
            bg_filename = background_files[bg_index]
            bg_path = os.path.join(BACKGROUNDS_DIR, bg_filename)
            bg_img = Image.open(bg_path).convert("RGB")

            plot_with_shadow = add_shadow(plot_img)

            if bg_img.width < plot_with_shadow.width or bg_img.height < plot_with_shadow.height:
                new_bg_width = int(plot_with_shadow.width * 1.1)
                new_bg_height = int(plot_with_shadow.height * 1.1)
                bg_img = bg_img.resize((new_bg_width, new_bg_height), Image.Resampling.LANCZOS)

            paste_x = (bg_img.width - plot_with_shadow.width) // 2
            paste_y = (bg_img.height - plot_with_shadow.height) // 2
            bg_img.paste(plot_with_shadow, (paste_x, paste_y), plot_with_shadow)

            output_path = os.path.join(OUTPUT_DIR, os.path.splitext(filename)[0] + '.jpg')
            bg_img.save(output_path, 'JPEG', quality=98)
        except Exception as e:
            print(f"\nCould not process {filename}. Error: {e}")

    final_count = len(os.listdir(OUTPUT_DIR))
    print(f"\n✅ Full dataset generation complete. {final_count} images saved.")

if __name__ == '__main__':
    generate_full_dataset()