import cv2
import easyocr
import os
from ultralytics import YOLO

# ==========================================
# 1. SETUP
# ==========================================
# Use 'yolov8n.pt' for testing. 
# Once you have your own trained model, change this to 'best.pt'
MODEL_NAME = 'best.pt' 

INPUT_FOLDER = 'test_images_2'
OUTPUT_FOLDER = 'results_final'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print(f"[INIT] Loading Model: {MODEL_NAME}...")
try:
    # This automatically handles model loading (no more Hydra errors)
    model = YOLO(MODEL_NAME) 
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

print("[INIT] Loading EasyOCR...")
# Set gpu=True if you have NVIDIA
reader = easyocr.Reader(['en'], gpu=True) 

# ==========================================
# 2. PROCESSING LOOP
# ==========================================
if not os.path.exists(INPUT_FOLDER):
    print(f"ERROR: Could not find folder '{INPUT_FOLDER}'")
    print(f"Make sure '{INPUT_FOLDER}' is inside: {os.getcwd()}")
    exit()

image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
print(f"Found {len(image_files)} images.")

for img_file in image_files:
    img_path = os.path.join(INPUT_FOLDER, img_file)
    print(f"Analyzing {img_file}...")

    # 1. Run Detection
    # conf=0.25 is standard. Increase if you see too much garbage.
    results = model.predict(img_path, conf=0.25)
    
    # 2. Get Image for Drawing
    img = cv2.imread(img_path)
    
    # 3. Process Detections
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Class Name
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            
            # --- OCR STEP ---
            # Crop the detected area
            crop = img[y1:y2, x1:x2]
            ocr_text = ""
            
            if crop.size > 0:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                # detail=0 returns just the text string
                ocr_results = reader.readtext(gray, detail=0)
                if ocr_results:
                    ocr_text = " ".join(ocr_results).upper()

            # --- DRAWING ---
            # Green Box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Label
            label = f"{class_name}"
            if ocr_text:
                label += f" | {ocr_text}"
            
            # Label Background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
            
            # Label Text
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            print(f"   -> {class_name} @ {x1},{y1} | Text: {ocr_text}")

    # 4. Save
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"res_{img_file}"), img)

print(f"Done! Check the '{OUTPUT_FOLDER}' folder.")
