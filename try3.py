import cv2
import easyocr
import os
import re
import torch
from ultralytics import YOLO

# ==========================================
# 1. SETUP
# ==========================================
# Innovation: Use a specialized 'license_plate' model for the 'small box' effect
MODEL_NAME = 'yolov8n.pt' 
INPUT_FOLDER = 'test_images_2'
OUTPUT_FOLDER = 'results_final2'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print(f"[INIT] Loading Model: {MODEL_NAME}...")
model = YOLO(MODEL_NAME) 

print("[INIT] Loading EasyOCR...")
reader = easyocr.Reader(['en'], gpu=True) 

# ==========================================
# 2. INNOVATION: TEXT FILTERING & CLEANING
# ==========================================
def clean_plate_text(ocr_results):
    """
    Keeps only alphanumeric characters and filters by length.
    Prevents reading long company names from the side of trucks.
    """
    full_text = "".join(ocr_results).upper()
    # Remove all non-alphanumeric characters
    clean_text = re.sub(r'[^A-Z0-9]', '', full_text)
    
    # Logic: Real plates are usually 3 to 9 characters long
    if 3 <= len(clean_text) <= 9:
        return clean_text
    return ""

# ==========================================
# 3. PROCESSING LOOP
# ==========================================
image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

for img_file in image_files:
    img_path = os.path.join(INPUT_FOLDER, img_file)
    img = cv2.imread(img_path)
    
    # Run Detection
    results = model.predict(img_path, conf=0.35) 
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_name = model.names[int(box.cls[0])]
            
            # --- ADVANCED CROP & PRE-PROCESS ---
            crop = img[y1:y2, x1:x2]
            ocr_text = ""
            
            if crop.size > 0:
                try:
                    # 1. Upscale (2x) to help OCR find small text
                    resized = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    # 2. Grayscale
                    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                    # 3. Bilateral Filter (Removes noise, keeps letter edges sharp)
                    enhanced = cv2.bilateralFilter(gray, 11, 17, 17)
                    
                    ocr_results = reader.readtext(enhanced, detail=0)
                    ocr_text = clean_plate_text(ocr_results)
                    
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        torch.cuda.empty_cache()
                    continue

            # --- DRAWING ---
            if ocr_text:
                # Green Box for the vehicle
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Label Background
                label = f"{class_name}: {ocr_text}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(img, (x1, y1 - 30), (x1 + w, y1), (0, 255, 0), -1)
                
                # Label Text
                cv2.putText(img, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                print(f"   -> {class_name} Plate Found: {ocr_text}")

    # Save and free GPU memory
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"alpr_{img_file}"), img)
    torch.cuda.empty_cache() 

print(f"Done! Check the '{OUTPUT_FOLDER}' folder.")