import cv2
import easyocr
import os
import numpy as np
import re
import urllib.request
from ultralytics import YOLO

# ==========================================
# 1. CONFIGURATION
# ==========================================
# This MUST match the name of the file you trained
MODEL_PATH = 'carplate.pt' 
INPUT_FOLDER = 'test_images'
OUTPUT_FOLDER = 'final_results'

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 1.1 AUTO-DOWNLOAD SUPER RESOLUTION MODEL
# We need this file to make the tiny construction plate readable
edsr_path = "EDSR_x4.pb"
if not os.path.exists(edsr_path):
    print("Downloading Super-Resolution Model (EDSR_x4)...")
    url = "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb"
    urllib.request.urlretrieve(url, edsr_path)

# Initialize Engines
print("Loading Custom YOLO Model...")
try:
    model = YOLO(MODEL_PATH)
except:
    print(f"[WARNING] '{MODEL_PATH}' not found. Using generic YOLOv8n (Results will be worse).")
    model = YOLO('yolov8n.pt')

print("Loading OCR Engine...")
reader = easyocr.Reader(['en'], gpu=True)

print("Loading Super-Resolution Engine...")
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(edsr_path)
sr.setModel("edsr", 4) # 4x Upscaling

# ==========================================
# 2. THE LOGIC "BRAIN"
# ==========================================

def get_best_reading(crop):
    """
    Tries multiple methods to read the plate until it passes the Logic Check.
    """
    # 1. UPSCALING (The "Construction Vehicle" Fix)
    # Turn tiny 20px text into crisp 80px text
    try:
        upscaled = sr.upsample(crop)
    except:
        upscaled = cv2.resize(crop, (0,0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    # 2. GENERATE VIEWS
    views = []
    
    # View A: Grayscale Standard
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    views.append(gray)
    
    # View B: Binary (For Mud/Dirt)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
    views.append(binary)
    
    # View C: Dark Mode (For Night)
    # Gamma Correction
    gamma = 1.5
    table = np.array([((i / 255.0) ** (1.0/gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
    dark_mode = cv2.LUT(gray, table)
    views.append(dark_mode)

    # 3. SCAN ALL VIEWS
    for view in views:
        results = reader.readtext(view, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        text = "".join(results)
        
        # 4. LOGIC FILTER
        fixed_text = fix_malaysian_text(text)
        if fixed_text:
            return fixed_text # Return immediately if we found a valid plate
            
    return None

def fix_malaysian_text(text):
    """
    Forces text to match 'ABC 1234' format.
    Fixes 'J8' -> 'JBG' errors.
    """
    clean = re.sub(r'[^A-Z0-9]', '', text.upper())
    if len(clean) < 3: return None
    
    # Correction Maps
    num2char = {'0':'D', '1':'I', '4':'A', '5':'S', '6':'G', '8':'B'}
    char2num = {'O':'0', 'I':'1', 'Z':'2', 'J':'3', 'A':'4', 'S':'5', 'G':'6', 'B':'8', 'Q':'0'}
    
    # Split hypothesis (Letter vs Number section)
    for i in range(1, len(clean)):
        L = list(clean[:i])
        R = list(clean[i:])
        
        # Check for suffix (W 1234 R)
        S = ""
        if len(R) > 1 and R[-1].isalpha():
            S = R.pop()

        # Force Logic
        L = [num2char.get(c, c) if c.isdigit() else c for c in L]
        R = [char2num.get(c, c) if c.isalpha() else c for c in R]
        
        candidate = "".join(L) + "".join(R) + S
        
        # Validation: [Letters] [Numbers] [Optional Letter]
        if re.match(r'^[A-Z]+[0-9]+[A-Z]*$', candidate):
            return candidate
            
    return None

# ==========================================
# 3. MAIN LOOP
# ==========================================

if __name__ == "__main__":
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.jpg', '.png'))]
    print(f"Processing {len(files)} images with Enhanced Logic...")

    for f in files:
        img_path = os.path.join(INPUT_FOLDER, f)
        original = cv2.imread(img_path)
        if original is None: continue

        # 1. DETECT (Using your new CUSTOM model)
        # We can trust this model more, so we don't need extremely low confidence
        # But we keep it somewhat low to be safe.
        results = model.predict(original, conf=0.20, verbose=False)
        
        plate_found = False
        
        for r in results:
            for box in r.boxes:
                # Get Coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # GEOMETRIC CHECK (Square filter)
                # Even with a custom model, this is a good safety net against headlights
                w, h = x2-x1, y2-y1
                if w < h * 1.3: continue 

                # 2. CROP WITH PADDING (Fixes 'J' cut-off)
                h_img, w_img = original.shape[:2]
                pad_x = int(w * 0.15)
                pad_y = int(h * 0.15)
                
                cx1 = max(0, x1 - pad_x)
                cy1 = max(0, y1 - pad_y)
                cx2 = min(w_img, x2 + pad_x)
                cy2 = min(h_img, y2 + pad_y)
                
                crop = original[cy1:cy2, cx1:cx2]
                
                # 3. INTELLIGENT READING
                final_text = get_best_reading(crop)
                
                if final_text:
                    print(f"  [MATCH] {f} -> {final_text}")
                    plate_found = True
                    
                    # Draw Green Box & Text
                    cv2.rectangle(original, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(original, final_text, (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    break # Stop if we found a valid plate
        
        if not plate_found:
            print(f"  [FAIL] {f} - No valid text found.")
            
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"RESULT_{f}"), original)

    print("Done.")