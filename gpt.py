import os
import glob
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# ======================================================
# 1. DEVICE
# ======================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("[SYSTEM] Using:", device)

# ======================================================
# 2. PATHS (THIS IS WHAT YOU ASKED FOR)
# ======================================================
INPUT_FOLDER = "test_images"
OUTPUT_FOLDER = "results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ======================================================
# 3. MODELS
# ======================================================
plate_detector = YOLO("yolov8n.pt")  # replace with your plate model if available

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
ocr_model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-base-printed"
).to(device)

# Super-resolution (EDSR)
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel("EDSR_x2.pb")
sr.setModel("edsr", 2)

# ======================================================
# 4. PLATE RECTIFICATION
# ======================================================
def rectify_plate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 200)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img

    cnt = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    angle = rect[-1]
    if angle < -45:
        angle += 90

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)

# ======================================================
# 5. TRANSFORMER OCR
# ======================================================
def transformer_ocr(img):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pixels = processor(images=pil, return_tensors="pt").pixel_values.to(device)
    ids = ocr_model.generate(pixels, max_length=12)
    text = processor.batch_decode(ids, skip_special_tokens=True)[0]
    return ''.join(c for c in text if c.isalnum()).upper()

# ======================================================
# 6. MAIN PIPELINE (FOLDER LOOP)
# ======================================================
image_files = glob.glob(os.path.join(INPUT_FOLDER, "*"))
print(f"[INFO] Found {len(image_files)} images")

for idx, img_path in enumerate(image_files):
    if not img_path.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    filename = os.path.basename(img_path)
    print(f"[{idx+1}] Processing {filename}")

    img = cv2.imread(img_path)
    if img is None:
        continue

    results = plate_detector(img)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        plate = img[y1:y2, x1:x2]
        if plate.size == 0:
            continue

        plate = rectify_plate(plate)
        plate = sr.upsample(plate)
        text = transformer_ocr(plate)

        if len(text) > 2:
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(img, text, (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0,0,255), 2)

    cv2.imwrite(os.path.join(OUTPUT_FOLDER, filename), img)

print("\n[DONE] All images processed. Results saved in 'results/'")
