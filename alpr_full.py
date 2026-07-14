import cv2
import easyocr
import os
import numpy as np
import re
import time
from ultralytics import YOLO

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
MODEL_PATH = 'best.pt'
INPUT_FOLDER = 'test_images_2'
OUTPUT_FOLDER = 'forensic_results'
DEBUG_FOLDER = os.path.join(OUTPUT_FOLDER, 'steps') # To save intermediate "treatment" images

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(DEBUG_FOLDER, exist_ok=True)

# "Thinking" Pause (in seconds) so you can see it working
THINK_TIME = 1.0 

# ==============================================================================
# 2. THE "DOCTOR" (Diagnostic & Treatment Engine)
# ==============================================================================

def diagnose_and_treat(crop, filename_prefix):
    """
    Step 1: DIAGNOSE the image condition (Dark? Blur?)
    Step 2: TREAT it with specific filters.
    """
    print(f"      [DIAGNOSING] Analyzing crop condition...")
    
    treatment_log = []
    processed = crop.copy()
    
    # --- TEST 1: BRIGHTNESS CHECK ---
    # Convert to HSV to measure Value (Brightness)
    hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
    avg_brightness = np.mean(hsv[:, :, 2])
    
    if avg_brightness < 90: # Threshold for "Dark"
        print(f"      [!] Diagnosis: LOW LIGHT (Level {int(avg_brightness)})")
        print("      [+] Treatment: Applying Adaptive Gamma Correction...")
        
        # Heavy Gamma boost for night plates
        gamma = 2.0 
        table = np.array([((i / 255.0) ** (1.0/gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
        processed = cv2.LUT(processed, table)
        treatment_log.append("Gamma")
    else:
        print(f"      [OK] Brightness is adequate ({int(avg_brightness)}).")

    # --- TEST 2: BLUR CHECK ---
    # Variance of Laplacian measures sharpness
    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    if blur_score < 100: # Threshold for "Blurry"
        print(f"      [!] Diagnosis: BLURRY (Score {int(blur_score)})")
        print("      [+] Treatment: Applying Sharpening Kernel...")
        
        # Sharpening Kernel
        kernel = np.array([[0, -1, 0], 
                           [-1, 5,-1], 
                           [0, -1, 0]])
        processed = cv2.filter2D(processed, -1, kernel)
        treatment_log.append("Sharpen")
    else:
        print(f"      [OK] Focus is sharp (Score {int(blur_score)}).")
        
    # --- TEST 3: CONTRAST CHECK ---
    # Always good to maximize contrast for OCR
    print("      [+] Routine Care: Maximizing Contrast (CLAHE)...")
    # We apply CLAHE to the Grayscale version for final OCR
    if len(processed.shape) == 3:
        gray_final = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    else:
        gray_final = processed
        
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    final_output = clahe.apply(gray_final)
    
    # Save the "Treated" image so you can inspect it
    save_path = os.path.join(DEBUG_FOLDER, f"treated_{filename_prefix}.jpg")
    cv2.imwrite(save_path, final_output)
    
    return final_output

# ==============================================================================
# 3. THE "LINGUIST" (Smart Substitution Logic)
# ==============================================================================

def smart_substitution(raw_text):
    """
    Fixes specific character confusions based on Malaysian Plate Logic.
    Examples: 'POA' -> 'PQA', 'A' -> '4' in numbers section.
    """
    # 1. Clean garbage (keep alphanumeric only)
    clean = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
    
    if len(clean) < 3: return None
    
    # 2. DEFINE CONFUSION MAPS
    # Letters that get confused as Numbers
    let_to_num = {'O':'0', 'I':'1', 'Z':'2', 'J':'3', 'A':'4', 'S':'5', 'G':'6', 'B':'8', 'Q':'0', 'D':'0'}
    # Numbers that get confused as Letters
    num_to_let = {'0':'D', '1':'I', '4':'A', '5':'S', '6':'G', '8':'B'}
    # Special: 'Q' and 'O' confusion (for 'PQA' plates)
    
    # 3. SPLIT & ANALYZE
    # We try to split the text into [LETTERS] [NUMBERS]
    # We iterate through every possible split point to find the one that makes sense.
    
    best_candidate = None
    
    for i in range(1, len(clean)):
        L = list(clean[:i]) # Left Part (Expected Letters)
        R = list(clean[i:]) # Right Part (Expected Numbers)
        
        # Check for suffix letter (e.g. W 1234 A)
        suffix = []
        if len(R) > 1 and R[-1].isalpha() and R[-1] not in ['Q']: # 'Q' is usually '0' at the end
             # But be careful, sometimes last char is legit letter.
             pass 

        # --- FIX LEFT (LETTERS) ---
        # If we see '0' in letters, it's likely 'D' or 'O' or 'Q'.
        # For 'P0A' -> 'PQA' or 'POA'
        for idx, char in enumerate(L):
            if char.isdigit():
                # Force swap based on map
                L[idx] = num_to_let.get(char, char)
            
            # Specific Fix: 'O' is rare in first position, usually 'D' or 'Q' or 'P'
            if char == '0': L[idx] = 'D'

        # --- FIX RIGHT (NUMBERS) ---
        for idx, char in enumerate(R):
            if char.isalpha():
                # Force swap based on map
                R[idx] = let_to_num.get(char, char)
                
        candidate = "".join(L) + "".join(R)
        
        # REGEX VALIDATOR: [Letters] [Numbers]
        # Malaysian plates: 1-3 Letters + 1-4 Numbers (+ optional letter)
        if re.match(r'^[A-Z]{1,3}[0-9]{1,4}[A-Z]?$', candidate):
            return candidate

    return None

# ==============================================================================
# 4. MAIN PROCESS
# ==============================================================================

def main():
    print(">>> LOADING SYSTEM...")
    try:
        model = YOLO(MODEL_PATH)
    except:
        print("[WARN] Custom model not found. Using YOLOv8n.")
        model = YOLO('yolov8n.pt')
        
    reader = easyocr.Reader(['en'], gpu=True)
    
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.jpg', '.png'))]
    print(f">>> FOUND {len(files)} IMAGES. STARTING SERIAL PROCESSING...")
    print("---------------------------------------------------------------")

    for f in files:
        # 1. LOAD
        print(f"\n[IMAGE] Loading: {f}")
        time.sleep(THINK_TIME * 0.5) # Artificial pause for "Thinking"
        
        img_path = os.path.join(INPUT_FOLDER, f)
        original = cv2.imread(img_path)
        if original is None: continue

        # 2. DETECT
        print("   [SCAN] Scanning for license plates...")
        # Use low confidence to find dirty/dark plates
        results = model.predict(original, conf=0.15, verbose=False)
        
        plate_found_in_image = False
        
        for r in results:
            for box in r.boxes:
                coords = list(map(int, box.xyxy[0]))
                
                # --- GEOMETRIC FILTER ---
                w = coords[2] - coords[0]
                h = coords[3] - coords[1]
                if w < h * 1.3: 
                    continue # Skip squares (headlights)
                
                # --- CROP & EXPAND (20%) ---
                h_img, w_img = original.shape[:2]
                pad_x = int(w * 0.20)
                pad_y = int(h * 0.20)
                
                cx1 = max(0, coords[0] - pad_x)
                cy1 = max(0, coords[1] - pad_y)
                cx2 = min(w_img, coords[2] + pad_x)
                cy2 = min(h_img, coords[3] + pad_y)
                
                raw_crop = original[cy1:cy2, cx1:cx2]
                
                # 3. DIAGNOSE & TREAT
                # We analyze THIS specific crop
                treated_crop = diagnose_and_treat(raw_crop, f"{f}_{cx1}")
                
                # 4. READ (OCR)
                print("      [READING] Extracting text...")
                ocr_result = reader.readtext(treated_crop, detail=0)
                raw_text = "".join(ocr_result)
                print(f"      [RAW OUTPUT] '{raw_text}'")
                
                # 5. SMART LOGIC FIX
                final_text = smart_substitution(raw_text)
                
                if final_text:
                    print(f"      [SUCCESS] VALID PLATE FOUND: {final_text}")
                    plate_found_in_image = True
                    
                    # Draw
                    cv2.rectangle(original, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 2)
                    cv2.putText(original, final_text, (coords[0], coords[1]-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # Save "Evidence"
                    cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"EVIDENCE_{final_text}.jpg"), treated_crop)
                    break # Stop looking at other boxes if we found the plate
                else:
                    print("      [FAIL] Could not validate text structure.")

        if not plate_found_in_image:
            print("   [RESULT] No valid plate detected in this image.")
            
        # Save Result
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"RESULT_{f}"), original)
        print("---------------------------------------------------------------")

if __name__ == "__main__":
    main()