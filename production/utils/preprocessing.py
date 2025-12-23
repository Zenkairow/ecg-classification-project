import cv2
import numpy as np
from PIL import Image
import os

class SmartImagePreprocessor:
    """
    Production-grade preprocessing pipeline for Real-World ECG photos.
    Handles cropping, lighting correction, and safe resizing.
    """
    
    @staticmethod
    def find_and_crop_ecg(image: np.ndarray) -> np.ndarray:
        """
        Locates the paper in a photo and crops to it.
        Pipeline: Grayscale -> Blur -> Adaptive Thresh -> Largest Rect Contour.
        """
        try:
            # 1. Convert to Gray
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
                
            # 2. Gaussian Blur (Reduce noise)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # 3. Adaptive Thresholding (Handle uneven lighting)
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # 4. Find Contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return image # Fallback: Return original
                
            # 5. Find Largest Rectangular Contour
            # Sort by area descending
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            largest_contour = contours[0]
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Basic sanity check: Is it big enough to be the paper? (>10% of image area)
            img_h, img_w = image.shape[:2]
            img_area = img_w * img_h
            if w * h < 0.1 * img_area:
                return image # Too small, probably noise
                
            # --- LEAD NAME PRESERVATION LOGIC ---
            # Add a 5% margin around the grid to capture lead labels (I, II, V1...)
            # We add more margin to the LEFT (where labels usually are)
            margin_x_left = int(w * 0.08)  # 8% left margin for labels
            margin_x_right = int(w * 0.02) # 2% right margin
            margin_y = int(h * 0.02)       # 2% top/bottom margin
            
            x_new = max(0, x - margin_x_left)
            y_new = max(0, y - margin_y)
            w_new = min(img_w - x_new, w + margin_x_left + margin_x_right)
            h_new = min(img_h - y_new, h + margin_y * 2)
            
            # Crop with margins
            cropped = image[y_new:y_new+h_new, x_new:x_new+w_new]
            return cropped
            
        except Exception as e:
            print(f"[SmartPreprocessor] Cropping failed: {e}")
            return image

    @staticmethod
    def enhance_contrast(image: np.ndarray) -> np.ndarray:
        """
        Applies CLAHE to normalize uneven lighting/shadows.
        """
        try:
            # CLAHE requires Lab color space or Grayscale. 
            # We usually work in RGB for the final model, but contrast is best done on Luminance.
            
            if len(image.shape) == 3:
                # Convert to LAB
                lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                
                # Apply CLAHE to L-channel
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                
                # Merge and convert back
                limg = cv2.merge((cl, a, b))
                enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
                return enhanced
            else:
                # Grayscale
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                return clahe.apply(image)
                
        except Exception as e:
            print(f"[SmartPreprocessor] Contrast enhancement failed: {e}")
            return image

    @staticmethod
    def resize_with_padding(image: np.ndarray, target_size: int = 224) -> np.ndarray:
        """
        Resizes longest side to target_size and pads the rest to maintain aspect ratio.
        Prevents signal squashing.
        """
        try:
            h, w = image.shape[:2]
            scale = target_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Create canvas
            canvas = np.full((target_size, target_size, 3), 255, dtype=np.uint8)
            
            # Center the image
            x_offset = (target_size - new_w) // 2
            y_offset = (target_size - new_h) // 2
            
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            return canvas
            
        except Exception as e:
            print(f"[SmartPreprocessor] Resizing failed: {e}")
            return cv2.resize(image, (target_size, target_size))

    def process(self, image_input, target_size=224) -> np.ndarray:
        """
        Master pipeline: Load -> Crop -> Enhance -> Resize-Pad.
        Input: filepath (str) or PIL Image or numpy array.
        Output: numpy array (RGB) ready for Tensor conversion.
        """
        # 1. Load / Convert to Numpy
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            image = np.array(image_input)
        elif isinstance(image_input, np.ndarray):
            image = image_input
        else:
            raise ValueError("Unsupported image format")
            
        # 2. Find and Crop
        cropped = self.find_and_crop_ecg(image)
        
        # 3. Enhance Contrast
        enhanced = self.enhance_contrast(cropped)
        
        # 4. Smart Resize
        final = self.resize_with_padding(enhanced, target_size=target_size)
        
        return final

if __name__ == "__main__":
    # Test Block
    print("Running SmartImagePreprocessor Test...")
    
    # Create a dummy image with a "paper" rectangle in the center
    dummy = np.zeros((800, 600, 3), dtype=np.uint8) # Dark background
    
    # Draw a white "paper" rotated/offset
    cv2.rectangle(dummy, (100, 150), (500, 650), (255, 255, 255), -1) 
    # Add some noise/shadow simulated by grey
    cv2.circle(dummy, (200, 200), 50, (200, 200, 200), -1)
    
    # Save dummy
    cv2.imwrite("debug_dummy_input.jpg", dummy)
    
    processor = SmartImagePreprocessor()
    result = processor.process("debug_dummy_input.jpg")
    
    # Save result
    # Convert RGB back to BGR for OpenCV saving
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite("debug_crop.jpg", result_bgr)
    
    print(f"Test Complete. Input: {dummy.shape}, Output: {result.shape}")
    print("Check 'debug_crop.jpg' to verify.")
