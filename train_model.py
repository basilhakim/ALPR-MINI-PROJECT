from ultralytics import YOLO
from roboflow import Roboflow
import os

# --- THE FIX: EVERYTHING MUST BE INSIDE THIS BLOCK ---
if __name__ == '__main__':
    
    # 1. DOWNLOAD DATA
    # The workers won't run this part anymore, solving the Network Timeout.
    print("Downloading dataset...")
    rf = Roboflow(api_key="JijZ4WpwP6nWxQpqV5uM")
    project = rf.workspace("nasi-goreng-pattaya").project("nlpalpr2026")
    dataset = project.version(1).download("yolov8")

    # 2. VERIFY PATH
    print(f"Dataset location: {dataset.location}")
    yaml_path = f"{dataset.location}/data.yaml"
    if not os.path.exists(yaml_path):
        print("CRITICAL ERROR: data.yaml not found! Check download.")
        exit()

    # 3. START TRAINING
    print("Starting Training...")
    model = YOLO("yolov8n.pt") 

    try:
        results = model.train(
            data=yaml_path, 
            epochs=50, 
            imgsz=640,
            
            # SETTINGS
            batch=4,        
            workers=1,      # Keeping this low is safer for Windows
            device=0,       
            
            name="my_custom_plate_model"
        )
        print("Training Complete!")
        print("Your new brain is at: runs/detect/my_custom_plate_model/weights/best.pt")

    except Exception as e:
        print(f"TRAINING CRASHED: {e}")