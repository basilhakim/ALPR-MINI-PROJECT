import cv2
import easyocr
import os
import re
import torch
import csv
from datetime import datetime
from ultralytics import YOLO

# ==========================================
# 1. SETUP
# ==========================================
MODEL_NAME = 'yolov8n.pt' 
INPUT_FOLDER = 'test_images_2'
OUTPUT_FOLDER = 'results_inno'
CSV_FILE = 'detections_log.csv' # Added Logging
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize CSV file with headers
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Filename', 'Vehicle_Type', 'Plate_Number'])

model = YOLO(MODEL_NAME) 
reader = easyocr.Reader(['en'], gpu=True) 

# ==========================================
# 2. INNOVATION: SMART TEXT CLEANING
# ==========================================
def filter_plate_text(ocr_results):
    full_text = "".join(ocr_results).upper()
    clean_text = re.sub(r'[^A-Z0-9]', '', full_text)
    
    # Logic: Most plates are 3-8 chars. 
    # Too long? Probably a company name. Too short? Probably noise.
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
    results = model.predict(img_path, conf=0.35) # Slightly higher confidence
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_name = model.names[int(box.cls[0])]
            
            # --- IMPROVED CROP & PRE-PROCESS ---
            crop = img[y1:y2, x1:x2]
            ocr_text = ""
            
            if crop.size > 0:
                try:
                    # Resize + Bilateral Filter (Sharper edges for OCR)
                    resized = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                    enhanced = cv2.bilateralFilter(gray, 11, 17, 17) # Sharpness Innovation
                    
                    ocr_results = reader.readtext(enhanced, detail=0)
                    ocr_text = filter_plate_text(ocr_results)
                    
                except Exception as e:
                    torch.cuda.empty_cache()
                    continue

            # --- DRAWING & LOGGING ---
            if ocr_text:
                # Log to CSV
                with open(CSV_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([datetime.now(), img_file, class_name, ocr_text])

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, ocr_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"final_{img_file}"), img)
    torch.cuda.empty_cache() 

print(f"Success! Data logged to {CSV_FILE}")