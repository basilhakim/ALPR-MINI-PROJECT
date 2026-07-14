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
OUTPUT_FOLDER = "results_forensic3"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# MODEL CONFIG
CONFIDENCE_THRESHOLD = 20  # Increased slightly to reduce false positives
OCR_STRICTNESS = False     

# PERFORMANCE COUNTERS
stats = {
    "total_images": 0,
    "successful_images": 0,
    "total_plates_found": 0
}

# ==========================================
# 2. AI SUPER-RESOLUTION (EDSR)
# ==========================================
def get_super_resolution_model():
    model_path = "EDSR_x4.pb"
    url = "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb"
    
    if not os.path.exists(model_path):
        print("[INSTALL] Downloading Super-Resolution AI (38MB)...")
        urllib.request.urlretrieve(url, model_path)
        print("[INSTALL] Download Complete.")
    
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("edsr", 4) 
    if USE_GPU:
        sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    return sr

print("[INIT] Loading AI Models...")
rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE_NAME).project(PROJECT_NAME)
model = project.version(VERSION_NUMBER).model
reader = easyocr.Reader(['en'], gpu=USE_GPU)
sr_model = get_super_resolution_model() 

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def enhance_plate(plate_img):
    try:
        enhanced = sr_model.upsample(plate_img)
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast = clahe.apply(gray)
        _, binary = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    except Exception as e:
        return cv2.resize(plate_img, (0,0), fx=4, fy=4)

def validate_text(text):
    clean = ''.join(e for e in text if e.isalnum()).upper()
    clean = clean.replace('I', '1').replace('O', '0')
    if clean.startswith('0'): clean = 'W' + clean[1:]
    if clean.startswith('5'): clean = 'S' + clean[1:]
    if len(clean) < 3: return None
    return clean

def print_final_report(s):
    """ Generates a forensic summary of the run """
    print("\n" + "="*45)
    print("        ALPR PERFORMANCE SUMMARY")
    print("="*45)
    
    success_rate = (s["successful_images"] / s["total_images"] * 100) if s["total_images"] > 0 else 0
    
    print(f"Total Images Processed:     {s['total_images']}")
    print(f"Images with Valid Plates:   {s['successful_images']}")
    print(f"Total Plates Identified:    {s['total_plates_found']}")
    print(f"System Success Rate:        {success_rate:.2f}%")
    print("="*45)
    print("Results saved in:", OUTPUT_FOLDER)

# ==========================================
# 4. MAIN LOOP
# ==========================================
files = glob.glob(os.path.join(INPUT_FOLDER, "*"))
valid_extensions = ('.jpg', '.png', '.jpeg')
image_files = [f for f in files if f.lower().endswith(valid_extensions)]

print(f"Found {len(image_files)} images.\n")

for index, fpath in enumerate(image_files):
    stats["total_images"] += 1
    name = os.path.basename(fpath)
    print(f"[{index+1}/{len(image_files)}] Analyzing {name}...")
    
    original = cv2.imread(fpath)
    if original is None: continue

    try:
        preds = model.predict(original, confidence=CONFIDENCE_THRESHOLD, overlap=30).json()['predictions']
    except Exception as e:
        print(f"  ! API Error: {e}")
        continue

    found_in_image = False
    
    for det in preds:
        x, y, w, h = int(det['x']), int(det['y']), int(det['width']), int(det['height'])
        
        # Padding
        pad = int(w * 0.1)
        x1 = max(0, int(x - w/2) - pad)
        y1 = max(0, int(y - h/2) - pad)
        x2 = min(original.shape[1], int(x + w/2) + pad)
        y2 = min(original.shape[0], int(y + h/2) + pad)
        
        plate_crop = original[y1:y2, x1:x2]
        if plate_crop.size == 0: continue

        # Enhance and Read
        final_plate = enhance_plate(plate_crop)
        results = reader.readtext(final_plate, detail=0)
        raw_text = "".join(results)
        clean_text = validate_text(raw_text)

        if clean_text:
            found_in_image = True
            stats["total_plates_found"] += 1
            print(f"   -> [FOUND]: {clean_text}")
            
            # Drawing Logic
            cv2.rectangle(original, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label = f"{clean_text}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(original, (x1, y1-30), (x1+tw, y1), (0, 255, 0), -1)
            cv2.putText(original, label, (x1, y1-7), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            # Save individual plate proof
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"plate_{clean_text}_{name}"), final_plate)

    if found_in_image:
        stats["successful_images"] += 1
    else:
        print("   -> [SKIP]: No readable plates.")

    # Save the full annotated image
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"result_{name}"), original)

# ==========================================
# 5. FINAL OUTPUT
# ==========================================
print_final_report(stats)