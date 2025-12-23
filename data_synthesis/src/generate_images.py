import os
import shutil
from PIL import Image, ImageFilter
from tqdm import tqdm

# --- 1. CONFIGURATION ---
# These relative paths are correct for your ecg-project folder.

# Path to your raw plot images
SOURCE_PLOTS_DIR = 'output/targets/'

# Path to your background images
BACKGROUNDS_DIR = 'backgrounds/'

# Where the final, combined images will be saved
OUTPUT_DIR = 'final_generated_images/'

# Number of sample images to generate for the test run
NUM_SAMPLES_TO_GENERATE = 10
# --- END OF CONFIGURATION ---


def add_shadow(img):
    """Adds a drop shadow effect to a PIL Image with a transparent background."""
    shadow_offset = (15, 15)
    shadow_color = (0, 0, 0, 180)
    shadow = Image.new('RGBA', img.size, (0, 0, 0, 0))
    alpha_mask = img.split()[3]
    shadow.paste(shadow_color, (0, 0), alpha_mask)
    shadow = shadow.filter(ImageFilter.GaussianBlur(10))
    final_canvas = Image.new('RGBA',
                            (img.width + shadow_offset[0], img.height + shadow_offset[1]),
                            (0, 0, 0, 0))
    final_canvas.paste(shadow, shadow_offset, shadow)
    final_canvas.paste(img, (0, 0), img)
    return final_canvas

def generate_images():
    """Main function to find, sort, and process images."""
    print("--- Starting image generation script. ---")

    for path in [SOURCE_PLOTS_DIR, BACKGROUNDS_DIR]:
        if not os.path.isdir(path):
            print(f"❌ ERROR: Directory not found at '{path}'")
            print("Please make sure it exists inside your 'ecg-project' folder.")
            return

    if os.path.exists(OUTPUT_DIR):
        print(f"🧹 Clearing old images in '{OUTPUT_DIR}'...")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"✅ Output directory ready at '{OUTPUT_DIR}'.")

    plot_filenames = [f for f in os.listdir(SOURCE_PLOTS_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    plot_filenames_sorted = sorted(plot_filenames, key=lambda f: int(f.replace('sample_', '').split('.')[0]))

    background_files = sorted([f for f in os.listdir(BACKGROUNDS_DIR) if f.endswith(('.png', '.jpg', '.jpeg', '.webp'))])

    if not plot_filenames_sorted or not background_files:
        print("❌ ERROR: No plot or background images found. Please check the folders.")
        return

    files_to_process = plot_filenames_sorted[:NUM_SAMPLES_TO_GENERATE]
    print(f"Found {len(plot_filenames)} plots. Will process {len(files_to_process)} samples for this test.")

    for i, filename in enumerate(tqdm(files_to_process, desc="Generating Images")):
        try:
            plot_path = os.path.join(SOURCE_PLOTS_DIR, filename)
            plot_img = Image.open(plot_path).convert("RGBA")

            bg_filename = background_files[i % len(background_files)]
            bg_path = os.path.join(BACKGROUNDS_DIR, bg_filename)
            bg_img = Image.open(bg_path).convert("RGB")

            plot_with_shadow = add_shadow(plot_img)

            if bg_img.width < plot_with_shadow.width or bg_img.height < plot_with_shadow.height:
                new_bg_width = max(bg_img.width, int(plot_with_shadow.width * 1.1))
                new_bg_height = max(bg_img.height, int(plot_with_shadow.height * 1.1))
                bg_img = bg_img.resize((new_bg_width, new_bg_height), Image.Resampling.LANCZOS)

            paste_x = (bg_img.width - plot_with_shadow.width) // 2
            paste_y = (bg_img.height - plot_with_shadow.height) // 2
            bg_img.paste(plot_with_shadow, (paste_x, paste_y), plot_with_shadow)

            output_filename = os.path.splitext(filename)[0] + '.jpg'
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            bg_img.save(output_path, 'JPEG', quality=95)

        except Exception as e:
            print(f"\n⚠️ Could not process {filename}. Error: {e}")

    print(f"\n🎉 Success! {len(os.listdir(OUTPUT_DIR))} sample images generated in '{OUTPUT_DIR}'.")


if __name__ == '__main__':
    generate_images()