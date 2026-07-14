import os
import glob
import cv2
import easyocr
import numpy as np
from roboflow import Roboflow

# ==========================================
# 1. CONFIGURATION
# ==========================================
# API Details
API_KEY = "JijZ4WpwP6nWxQpqV5uM"
PROJECT_NAME = "nlpalpr2026"
WORKSPACE_NAME = "nasi-goreng-pattaya"
VERSION_NUMBER = 1

# Folders
INPUT_FOLDER = "test_images"
OUTPUT_FOLDER = "results_final"

# Settings
CONF_THRESHOLD = 20      # Lower confidence to catch difficult plates
OCR_ALLOWLIST = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789' # Only read these chars

# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize Models
print("[INIT] Loading Roboflow Detector...")
rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE_NAME).project(PROJECT_NAME)
model = project.version(VERSION_NUMBER).model

print("[INIT] Loading EasyOCR...")
reader = easyocr.Reader(['en'], gpu=False)

# ==========================================
# 2. INNOVATION 1: IMAGE ENHANCEMENT
# ==========================================
def preprocess_full_image(img):
    """
    Apply CLAHE (Adaptive Histogram Equalization) to handle
    glare, night shots, and shadows.
    """
    # Convert to YUV (Y = Brightness)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    
    # Apply CLAHE to Y channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
    
    # Convert back to BGR
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

# ==========================================
# 3. INNOVATION 2: OCR OPTIMIZATION
# ==========================================
def preprocess_crop_for_ocr(plate_img):
    """
    Upscale and Binarize the crop to make characters
    huge and sharp for the OCR engine.
    """
    # 1. Upscale (2x Zoom)
    height, width = plate_img.shape[:2]
    plate_img = cv2.resize(plate_img, (width*2, height*2), interpolation=cv2.INTER_CUBIC)
    
    # 2. Grayscale & Blur (Reduce noise)
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    
    # 3. Otsu Thresholding (Pure Black & White)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary

# ==========================================
# 4. INNOVATION 3: LOGIC CLEANING
# ==========================================
def clean_plate_text(text):
    """
    Fixes common OCR confusion (e.g. '5' vs 'S') based on
    Malaysian plate standard (First char is Letter).
    """
    if len(text) == 0: return text
    
    text = list(text.upper())
    
    # MAPPING: Number -> Letter (For the first position)
    num_to_char = {'0': 'O', '1': 'I', '2': 'Z', '3': 'J', '4': 'A', '5': 'S', '6': 'G', '8': 'B'}
    
    # MAPPING: Letter -> Number (For the rest)
    char_to_num = {'O': '0', 'I': '1', 'Z': '2', 'J': '3', 'A': '4', 'S': '5', 'G': '6', 'B': '8'}

    # RULE 1: First character MUST be a Letter
    if text[0] in num_to_char:
        text[0] = num_to_char[text[0]]

    # RULE 2: If we have >3 chars, the last few are likely numbers
    # (Simplified logic to avoid breaking special plates)
    if len(text) > 3:
        for i in range(1, len(text)):
            if text[i] in char_to_num:
                # Only swap if it looks like a number context (heuristic)
                # For safety, we can apply this loosely or skip it.
                # Let's apply it only to the very last character for safety.
                if i == len(text) - 1:
                     text[i] = char_to_num[text[i]]

    return "".join(text)

# ==========================================
# 5. MAIN PIPELINE
# ==========================================
# Find images
extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
image_files = []
for ext in extensions:
    image_files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))
    image_files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext.upper())))

print(f"[INFO] Found {len(image_files)} images. Starting processing...")

for index, img_path in enumerate(image_files):
    filename = os.path.basename(img_path)
    print(f"[{index+1}/{len(image_files)}] Processing {filename}...")
    
    # Load Image
    original_frame = cv2.imread(img_path)
    if original_frame is None: continue

    # A. PREPROCESS (Global)
    enhanced_frame = preprocess_full_image(original_frame)

    # B. DETECT
    try:
        # Use enhanced frame for detection
        prediction = model.predict(enhanced_frame, confidence=CONF_THRESHOLD, overlap=30).json()
    except Exception as e:
        print(f"  -> Error: {e}")
        continue

    # C. CROP & READ
    found_plate = False
    for pred in prediction['predictions']:
        x, y, w, h = int(pred['x']), int(pred['y']), int(pred['width']), int(pred['height'])
        
        # Coordinates
        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)

        # Safety Check
        H, W, _ = original_frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        # Aspect Ratio Filter (Ignore vertical boxes)
        if w / h < 1.2: continue 

        # CROP from ENHANCED image
        plate_crop = enhanced_frame[y1:y2, x1:x2]
        
        # D. ENHANCE CROP (Upscale + Binary)
        ocr_ready_crop = preprocess_crop_for_ocr(plate_crop)

        # E. READ TEXT
        try:
            results = reader.readtext(ocr_ready_crop, detail=0, allowlist=OCR_ALLOWLIST)
            if len(results) > 0:
                raw_text = "".join(results)
                
                # F. CLEAN LOGIC
                final_text = clean_plate_text(raw_text)
                
                found_plate = True
                print(f"    -> Plate: {final_text} (Conf: {pred['confidence']:.2f})")
                
                # DRAW (Green Box + Text)
                cv2.rectangle(original_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Text Background
                (tw, th), _ = cv2.getTextSize(final_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(original_frame, (x1, y1 - 35), (x1 + tw, y1), (0, 0, 255), -1)
                cv2.putText(original_frame, final_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Save the binary crop for report (proof of innovation)
                crop_name = os.path.join(OUTPUT_FOLDER, "debug_crop_" + filename)
                cv2.imwrite(crop_name, ocr_ready_crop)
                
        except Exception as e:
            print("    -> OCR Failed")

    # Save Result
    save_path = os.path.join(OUTPUT_FOLDER, "result_" + filename)
    cv2.imwrite(save_path, original_frame)

print(f"\n[DONE] Results saved to '{OUTPUT_FOLDER}'.")