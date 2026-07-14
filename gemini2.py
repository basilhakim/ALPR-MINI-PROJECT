import os
import glob
import cv2
import easyocr
import numpy as np
import urllib.request
from roboflow import Roboflow
import torch

# ==========================================
# 1. SETUP & HARDWARE CHECK
# ==========================================
USE_GPU = torch.cuda.is_available()
print(f"[SYSTEM] AI Accelerator: {'ENABLED (Fast)' if USE_GPU else 'DISABLED (Slow)'}")

API_KEY = "JijZ4WpwP6nWxQpqV5uM"
PROJECT_NAME = "nlpalpr2026"
WORKSPACE_NAME = "nasi-goreng-pattaya"
VERSION_NUMBER = 1

INPUT_FOLDER = "test_images"
OUTPUT_FOLDER = "results_forensic"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# MODEL CONFIG
CONFIDENCE_THRESHOLD = 5  # Find EVERYTHING, filter later
OCR_STRICTNESS = False    # Try to read even garbage text

# ==========================================
# 2. INNOVATION: AI SUPER-RESOLUTION (EDSR)
# ==========================================
# This function downloads a 30MB Neural Network file if you don't have it
def get_super_resolution_model():
    model_path = "EDSR_x4.pb"
    url = "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb"
    
    if not os.path.exists(model_path):
        print("[INSTALL] Downloading Super-Resolution AI (38MB)...")
        urllib.request.urlretrieve(url, model_path)
        print("[INSTALL] Download Complete.")
    
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("edsr", 4) # Upscale 4x using Deep Learning
    if USE_GPU:
        sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    return sr

print("[INIT] Loading AI Models...")
rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE_NAME).project(PROJECT_NAME)
model = project.version(VERSION_NUMBER).model
reader = easyocr.Reader(['en'], gpu=USE_GPU)
sr_model = get_super_resolution_model() # Load the Enhancer

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def enhance_plate(plate_img):
    """
    Transform a blurry, pixelated plate into a sharp High-Res image
    using the EDSR Neural Network.
    """
    try:
        # 1. AI Super Resolution (The Magic)
        enhanced = sr_model.upsample(plate_img)
        
        # 2. Convert to Grayscale
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        
        # 3. Shadow Removal (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast = clahe.apply(gray)
        
        # 4. Binarization (Black & White)
        _, binary = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    except Exception as e:
        print(f"Warning: SR Failed ({e}), using fallback.")
        return cv2.resize(plate_img, (0,0), fx=4, fy=4)

def validate_text(text):
    """ Cleans text and checks if it looks like a plate """
    clean = ''.join(e for e in text if e.isalnum()).upper()
    
    # Common fixes
    clean = clean.replace('I', '1').replace('O', '0')
    if clean.startswith('0'): clean = 'W' + clean[1:]
    if clean.startswith('5'): clean = 'S' + clean[1:]
    
    if len(clean) < 3: return None
    return clean

# ==========================================
# 4. MAIN LOOP
# ==========================================
files = glob.glob(os.path.join(INPUT_FOLDER, "*"))
print(f"Found {len(files)} images.")

for index, fpath in enumerate(files):
    if not (fpath.lower().endswith(('.jpg', '.png', '.jpeg'))): continue
    name = os.path.basename(fpath)
    print(f"[{index+1}] Analyzing {name}...")
    
    original = cv2.imread(fpath)
    if original is None: continue

    # A. DETECT (Using Cloud API)
    try:
        preds = model.predict(original, confidence=CONFIDENCE_THRESHOLD, overlap=30).json()['predictions']
    except:
        continue

    found_anything = False
    
    for det in preds:
        x, y, w, h = int(det['x']), int(det['y']), int(det['width']), int(det['height'])
        
        # Padding (Crucial for OCR)
        pad = int(w * 0.1)
        x1 = max(0, int(x - w/2) - pad)
        y1 = max(0, int(y - h/2) - pad)
        x2 = min(original.shape[1], int(x + w/2) + pad)
        y2 = min(original.shape[0], int(y + h/2) + pad)
        
        plate_crop = original[y1:y2, x1:x2]
        if plate_crop.size == 0: continue

        # B. ENHANCE (AI Super Resolution)
        # This is slow but POWERFUL
        final_plate = enhance_plate(plate_crop)

        # C. READ
        results = reader.readtext(final_plate, detail=0)
        raw_text = "".join(results)
        clean_text = validate_text(raw_text)

        if clean_text:
            found_anything = True
            print(f"   -> LOCKED ON: {clean_text}")
            
            # Draw Green Box
            cv2.rectangle(original, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Draw Label
            label = f"{clean_text}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(original, (x1, y1-40), (x1+tw, y1), (0, 255, 0), -1)
            cv2.putText(original, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            # Save the "Enhanced" plate to prove your innovation
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"evidence_{name}_enhanced.jpg"), final_plate)

    if not found_anything:
        print("   -> No readable text found.")

    cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"result_{name}"), original)

print("\nProcess Complete. Check 'results_forensic' folder.")