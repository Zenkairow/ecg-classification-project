import cv2
import numpy as np
from PIL import Image
import os

class ECGScanner:
    def __init__(self, target_size=(512, 512)):
        self.target_size = target_size

    def order_points(self, pts):
        # Order points: top-left, top-right, bottom-right, bottom-left
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def four_point_transform(self, image, pts):
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        # Compute width of new image
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # Compute height of new image
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    def detect_and_warp(self, image):
        # Image expected as numpy array (cv2 format)
        # 1. Resize for faster detection (ratio)
        height, width = image.shape[:2]
        
        # If image is small, don't resize
        if height < 500:
            ratio = 1.0
            resized = image
        else:
            ratio = height / 500.0
            resized = cv2.resize(image, (int(width / ratio), 500))

        # 2. Edge Detection
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 75, 200)

        # 3. Find Contours
        cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

        screenCnt = None
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                screenCnt = approx
                break

        if screenCnt is None:
            # Failed to find paper, return original
            # print("Scanner: No paper contour found. Using original image.")
            return image

        # 4. Warp
        warped = self.four_point_transform(image, screenCnt.reshape(4, 2) * ratio)
        # print("Scanner: Paper detected and warped.")
        return warped

    def enhance_image_clahe(self, image):
        # Convert to LAB to process Lightness channel only
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply Safe CLAHE
        # ClipLimit 2.0 is conservative (prevents noise amplification)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        # Merge back
        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return final

    def preprocess(self, image_path):
        # Load
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image at {image_path}")

        # 1. Warp (Document Scan)
        # Try to detect paper. If not found, it keeps original.
        processed = self.detect_and_warp(image)

        # 2. Enhance (CLAHE) - Safe contrast fix
        processed = self.enhance_image_clahe(processed)

        # 3. Resize to Target (512x512) for ResNet/EfficientNet
        processed = cv2.resize(processed, self.target_size, interpolation=cv2.INTER_AREA)

        # Convert to PIL for model (RGB)
        processed_pil = Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
        return processed_pil

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        scanner = ECGScanner()
        out = scanner.preprocess(sys.argv[1])
        out.save("preprocessed_output.png")
        print("Test complete. Saved to preprocessed_output.png")
