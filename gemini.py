import os
import glob
import cv2
import easyocr
import numpy as np
from roboflow import Roboflow
import torch

# ==========================================
# 1. SETUP
# ==========================================
USE_GPU = torch.cuda.is_available()
print(f"[SYSTEM] GPU Acceleration: {'ENABLED' if USE_GPU else 'DISABLED'}")

API_KEY = "JijZ4WpwP6nWxQpqV5uM"
PROJECT_NAME = "nlpalpr2026"
WORKSPACE_NAME = "nasi-goreng-pattaya"
VERSION_NUMBER = 1

INPUT_FOLDER = "test_images"
OUTPUT_FOLDER = "results_debug"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# EXTREMELY LOW THRESHOLD (1%) to see everything the AI "thinks" might be a plate
CONFIDENCE_THRESHOLD = 1 

print("[INIT] Loading Engines...")
rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE_NAME).project(PROJECT_NAME)
model = project.version(VERSION_NUMBER).model
reader = easyocr.Reader(['en'], gpu=USE_GPU)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def create_night_vision(img):
    """ Brightens shadows to find hidden plates """
    gamma = 2.0 
    table = np.array([((i / 255.0) ** (1.0/gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def enhance_for_ocr(crop):
    """ Make text pop out """
    # Upscale 3x
    h, w = crop.shape[:2]
    zoom = cv2.resize(crop, (w*3, h*3), interpolation=cv2.INTER_CUBIC)
    # Grayscale
    gray = cv2.cvtColor(zoom, cv2.COLOR_BGR2GRAY)
    # Denoise
    clean = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    # Binary Threshold
    _, binary = cv2.threshold(clean, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

# ==========================================
# 3. MAIN LOOP
# ==========================================
files = glob.glob(os.path.join(INPUT_FOLDER, "*"))
print(f"Found {len(files)} images.")

for index, fpath in enumerate(files):
    if not (fpath.lower().endswith(('.jpg', '.png', '.jpeg'))): continue
    name = os.path.basename(fpath)
    print(f"[{index+1}] Debugging {name}...")
    
    original = cv2.imread(fpath)
    if original is None: continue
    final_img = original.copy()

    # --- DETECTION PASS ---
    # We use the Night Vision version for detection
    nv_img = create_night_vision(original)
    
    try:
        # Get ALL predictions, even weak ones
        preds = model.predict(nv_img, confidence=CONFIDENCE_THRESHOLD, overlap=30).json()['predictions']
    except Exception as e:
        print(f"  -> API Error: {e}")
        continue

    if len(preds) == 0:
        print("  -> NO OBJECTS DETECTED (Model failed)")

    for det in preds:
        # Get Coordinates
        x, y, w, h = int(det['x']), int(det['y']), int(det['width']), int(det['height'])
        conf = det['confidence'] * 100
        
        # Calculate Box
        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)
        
        # Clamp to image
        h_img, w_img = original.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        
        # ALWAYS DRAW THE BOX (Red = Detection only)
        cv2.rectangle(final_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # --- OCR PASS ---
        crop = original[y1:y2, x1:x2]
        if crop.size == 0: continue
        
        # Enhance for reading
        ocr_ready = enhance_for_ocr(crop)
        
        # Read RAW text (no filtering)
        results = reader.readtext(ocr_ready, detail=0)
        raw_text = " ".join(results)
        
        # Clean text
        clean_text = ''.join(e for e in raw_text if e.isalnum()).upper()
        
        print(f"  -> Detected Box at ({x},{y}) | Conf: {conf:.1f}% | Raw OCR: '{raw_text}'")

        # If we found text, turn box Green and show it
        if len(clean_text) > 1:
            cv2.rectangle(final_img, (x1, y1), (x2, y2), (0, 255, 0), 3) # Green = Read Success
            
            # Label
            label = f"{clean_text} ({conf:.0f}%)"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(final_img, (x1, y1-30), (x1+tw, y1), (0, 255, 0), -1)
            cv2.putText(final_img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        else:
            # Show "UNREADABLE" if OCR failed
            cv2.putText(final_img, "UNREADABLE", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Save
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"debug_{name}"), final_img)

print("Check 'results_debug' folder.")