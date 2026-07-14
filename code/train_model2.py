from ultralytics import YOLO
from roboflow import Roboflow
import os

# --- CRITICAL: Everything must be inside this main block for Windows compatibility ---
if __name__ == '__main__':
    
    print(">>> STEP 1: DOWNLOADING DATASET...")
    try:
        # 1. Initialize Roboflow
        # REPLACE 'PASTE_YOUR_API_KEY_HERE' WITH YOUR ACTUAL KEY
        rf = Roboflow(api_key="JijZ4WpwP6nWxQpqV5uM")
        
        # 2. Access the Public 'gocar' Workspace
        # This dataset is specific to Malaysian plates and has high accuracy
        print("   Connecting to 'gocar' workspace...")
        project = rf.workspace("gocar").project("malaysia-car-plate-number")
        
        # 3. Download Version 3 (Latest/Best)
        print("   Downloading dataset (v3)...")
        dataset = project.version(3).download("yolov8")
        
        print(f"   [SUCCESS] Dataset saved to: {dataset.location}")

    except Exception as e:
        print(f"\n[ERROR] Download Failed: {e}")
        print("Possible fixes:")
        print("1. Check your API Key.")
        print("2. Make sure you have internet access.")
        exit()

    # --- VERIFY DATA PATH ---
    yaml_path = f"{dataset.location}/data.yaml"
    if not os.path.exists(yaml_path):
        print(f"[CRITICAL] Could not find 'data.yaml' at {yaml_path}")
        exit()

    print("\n>>> STEP 2: STARTING TRAINING...")
    try:
        # Load the standard YOLOv8 Nano model (Small & Fast)
        model = YOLO("yolov8n.pt") 

        # Train on the downloaded data
        results = model.train(
            data=yaml_path, 
            epochs=50,       # 50 is usually enough for a solid result
            imgsz=640,       # Standard resolution
            
            # HARDWARE SETTINGS
            batch=4,         # Keep low to prevent memory crashes
            workers=1,       # Keep at 1 for Windows safety
            device=0,        # Use GPU (Change to 'cpu' if no NVIDIA card)
            
            # NAMING
            name="malaysian_plate_expert_v1"
        )
        
        print("\n>>> [DONE] TRAINING COMPLETE!")
        print(f"Your new best.pt file is located at: runs/detect/malaysian_plate_expert_v1/weights/best.pt")
        print("Use this file in your detection script to fix the false positives.")

    except Exception as e:
        print(f"\n[CRITICAL] Training Crashed: {e}")
