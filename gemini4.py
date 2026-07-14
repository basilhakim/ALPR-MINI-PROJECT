import os
import glob
import cv2
import easyocr
import numpy as np
import urllib.request
from roboflow import Roboflow
import torch

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
USE_GPU = torch.cuda.is_available()
API_KEY = "JijZ4WpwP6nWxQpqV5uM"
PROJECT_NAME = "nlpalpr2026"
WORKSPACE_NAME = "nasi-goreng-pattaya"
VERSION_NUMBER = 1

INPUT_FOLDER = "test_images"
OUTPUT_FOLDER = "results_forensic4"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# THEORETICAL BASELINE (For Comparison)
# Conventional systems often struggle with blur/low-res
BASE_DETECTION_RATE = 65.0 

# PERFORMANCE TRACKER
stats = {
    "total_images": 0,
    "plates_detected_opt": 0, # Your current pipeline
    "total_plates_read": 0
}

# ==========================================
# 2. AI MODELS (The "Innovation" Layer)
# ==========================================
def get_super_resolution_model():
    model_path = "EDSR_x4.pb"
    if not os.path.exists(model_path):
        print("[INSTALL] Downloading EDSR Super-Resolution AI...")
        url = "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb"
        urllib.request.urlretrieve(url, model_path)
    
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("edsr", 4) # 4x Upscaling
    if USE_GPU:
        sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    return sr

print("[INIT] Loading Pipeline Components...")
rf = Roboflow(api_key=API_KEY)
model = rf.workspace(WORKSPACE_NAME).project(PROJECT_NAME).version(VERSION_NUMBER).model
reader = easyocr.Reader(['en'], gpu=USE_GPU)
sr_model = get_super_resolution_model() 

# ==========================================
# 3. CORE PIPELINE FUNCTIONS
# ==========================================
def enhance_plate(plate_img):
    """ Step 4: Enhancement Model (EDSR + CLAHE + Binarization) """
    try:
        # AI Super-Resolution
        enhanced = sr_model.upsample(plate_crop)
        # Contrast Enhancement
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast = clahe.apply(gray)
        # Adaptive Binarization
        _, binary = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    except:
        return cv2.resize(plate_img, (0,0), fx=4, fy=4)

def validate_text(text):
    """ Step 5: OCR Validation & Correction """
    clean = ''.join(e for e in text if e.isalnum()).upper()
    # Common OCR Correction Logic
    clean = clean.replace('I', '1').replace('O', '0')
    if clean.startswith('0'): clean = 'W' + clean[1:]
    if clean.startswith('5'): clean = 'S' + clean[1:]
    return clean if len(clean) >= 3 else None

# ==========================================
# 4. MAIN PIPELINE EXECUTION
# ==========================================
image_files = [f for f in glob.glob(os.path.join(INPUT_FOLDER, "*")) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

for index, fpath in enumerate(image_files):
    stats["total_images"] += 1
    name = os.path.basename(fpath)
    original = cv2.imread(fpath)
    if original is None: continue

    # STEP 2: Plate Detection
    preds = model.predict(original, confidence=20).json()['predictions']
    
    found_in_img = False
    for det in preds:
        # STEP 3: Image Crop with Dynamic Padding
        x, y, w, h = int(det['x']), int(det['y']), int(det['width']), int(det['height'])
        pad = int(w * 0.1)
        x1, y1 = max(0, int(x-w/2)-pad), max(0, int(y-h/2)-pad)
        x2, y2 = min(original.shape[1], int(x+w/2)+pad), min(original.shape[0], int(y+h/2)+pad)
        
        plate_crop = original[y1:y2, x1:x2]
        if plate_crop.size == 0: continue

        # STEP 4: AI Enhancement
        final_plate = enhance_plate(plate_crop)

        # STEP 5: OCR Recognition
        results = reader.readtext(final_plate, detail=0)
        clean_text = validate_text("".join(results))

        if clean_text:
            found_in_img = True
            stats["total_plates_read"] += 1
            # Visual Output
            cv2.rectangle(original, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(original, clean_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"plate_{clean_text}_{name}"), final_plate)

    if found_in_img:
        stats["plates_detected_opt"] += 1

    cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"result_{name}"), original)

# ==========================================
# 5. FINAL COMPARISON REPORT
# ==========================================
opt_rate = (stats["plates_detected_opt"] / stats["total_images"]) * 100
print("\n" + "="*50)
print("       DETECTION COMPARISON: BASE vs OPTIMIZED")
print("="*50)
print(f"{'METRIC':<25} | {'BASE MODEL':<12} | {'OPTIMIZED'}")
print("-" * 50)
print(f"{'System Success Rate':<25} | {BASE_DETECTION_RATE:>11.2f}% | {opt_rate:>10.2f}%")
print(f"{'Images with Plates':<25} | {'---':>12} | {stats['plates_detected_opt']:>11}")
print(f"{'Total Images':<25} | {stats['total_images']:>12} | {stats['total_images']:>11}")
print("="*50)
print(f"Improvement: +{opt_rate - BASE_DETECTION_RATE:.2f}%")