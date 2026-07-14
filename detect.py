import os
import glob
import cv2
from roboflow import Roboflow

# ==========================================
# 1. SETUP
# ==========================================
API_KEY = "JijZ4WpwP6nWxQpqV5uM" 
PROJECT_NAME = "nlpalpr2026"
WORKSPACE_NAME = "nasi-goreng-pattaya"
VERSION_NUMBER = 1

INPUT_FOLDER = "test_images"
OUTPUT_FOLDER = "results"

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("Connecting to Roboflow...")
rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE_NAME).project(PROJECT_NAME)
model = project.version(VERSION_NUMBER).model

# ==========================================
# 2. SMART FILE FINDER
# ==========================================
# Look for common image formats only
extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
image_files = []

for ext in extensions:
    # Look for files ending with .jpg, .png, etc.
    image_files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))
    # Also look for uppercase versions (.JPG, .PNG)
    image_files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext.upper())))

print(f"Found {len(image_files)} valid images in '{INPUT_FOLDER}'.")

if len(image_files) == 0:
    print("ERROR: No images found! Make sure your pictures are directly inside the 'test_images' folder.")
    print("Example: test_images/car.jpg")
    exit()

# ==========================================
# 3. PROCESS LOOP
# ==========================================
for index, img_path in enumerate(image_files):
    
    filename = os.path.basename(img_path)
    
    # Read the image
    frame = cv2.imread(img_path)
    
    # Double check if image loaded correctly
    if frame is None:
        print(f"Skipping '{filename}' - Could not be read (Is it a corrupted file?)")
        continue

    print(f"[{index+1}/{len(image_files)}] Processing {filename}...")

    # Detect
    try:
        prediction = model.predict(frame, confidence=40, overlap=30).json()
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        continue

    # Draw Boxes
    detected = False
    for pred in prediction['predictions']:
        x, y, w, h = int(pred['x']), int(pred['y']), int(pred['width']), int(pred['height'])
        conf = pred['confidence']

        # --- FILTERS ---
        if conf < 0.50: continue 
        
        aspect_ratio = w / h
        if aspect_ratio < 1.5: continue 

        # --- DRAW ---
        detected = True
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Plate: {conf:.2f}"
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Save Result
    output_path = os.path.join(OUTPUT_FOLDER, "result_" + filename)
    cv2.imwrite(output_path, frame)

print("\nDone! Check the 'results' folder.")