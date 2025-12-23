import sys
import cv2
import os
import matplotlib.pyplot as plt

# Add production to path so we can import the utils
sys.path.append(os.path.join(os.getcwd(), 'production'))

try:
    from utils.preprocessing import SmartImagePreprocessor
except ImportError:
    # Try alternative path if running from inside production
    sys.path.append(os.getcwd())
    from production.utils.preprocessing import SmartImagePreprocessor

def main():
    if len(sys.argv) < 2:
        print("\nUsage: python test_pipeline.py <path_to_image.jpg>")
        print("Example: python test_pipeline.py my_ecg.jpg\n")
        return

    input_path = sys.argv[1]
    
    if not os.path.exists(input_path):
        print(f"❌ Error: File '{input_path}' not found.")
        return

    print(f"🔍 Processing: {input_path}")
    
    try:
        # Initialize
        preprocessor = SmartImagePreprocessor()
        
        # Run Pipeline (Target 512px for High-Res inspection)
        result = preprocessor.process(input_path, target_size=512)
        
        # Save Output
        output_filename = f"processed_{os.path.basename(input_path)}"
        
        # Convert RGB (Pipeline) to BGR (OpenCV)
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_filename, result_bgr)
        
        print("\n✅ Preprocessing Complete!")
        print(f"--------------------------------")
        print(f"Input:  {input_path}")
        print(f"Output: {output_filename}")
        print(f"Shape:  {result.shape}")
        print(f"--------------------------------")
        print("You can verify the 'Smart Crop' and 'Contrast Enhancement' in the output file.")

    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
