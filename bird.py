import cv2
import os
import numpy as np
from ultralytics import YOLO

# --- STEP 1: SETUP & ERROR CHECKING ---
# Define the video file name (Make sure this matches your file!)
video_filename = "carroadvideo.mp4"

# Check if file exists before running
if not os.path.exists(video_filename):
    # Try looking in the current script's folder just in case
    script_folder = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(script_folder, video_filename)
    
    if not os.path.exists(video_path):
        print(f"ERROR: Could not find {video_filename}")
        print(f"Please put the video inside: {os.getcwd()}")
        exit()
else:
    video_path = video_filename

# --- STEP 2: LOAD MODEL & VIDEO ---
# Load the YOLO model [cite: 122]
print("Loading YOLO model...")
model = YOLO("yolov8n.pt")  # Using nano for speed (or use 'yolov8x.pt' if you have it)

# Open the video [cite: 126]
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) # Get frames per second [cite: 128]

# --- STEP 3: CONFIGURATION (FROM NOTES) ---
# Define the two counting lines (y-coordinates) [cite: 130]
# Format: [x1, y1, x2, y2]
Line1 = [0, 150, 1200, 150]  # Start Line
Line2 = [0, 95, 1200, 95]    # End Line

# Real-world distance between these lines (in meters) 
distance_between_lines = 12 

# Dictionary to store data for each car ID [cite: 133]
vehicle_data = {}

# --- STEP 4: MAIN LOOP ---
frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Video finished.")
        break

    frame_count += 1

    # Run YOLO tracking (ByteTrack) [cite: 142, 144]
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

    # Process detections
    if results[0].boxes.id is not None:
        # Get boxes, IDs, and Classes [cite: 149-151]
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        
        for box, id in zip(boxes, ids):
            # Calculate Center Point (cx, cy) [cite: 155]
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Initialize data for new cars [cite: 157]
            if id not in vehicle_data:
                vehicle_data[id] = {"line1_frame": None, "line2_frame": None, "speed": None}

            # --- LOGIC: CHECK CROSSING LINE 1 (START) --- [cite: 164]
            # We use a small buffer (+/- 10 pixels) to catch the car crossing
            if Line1[1] - 10 < cy < Line1[1] + 10:
                if vehicle_data[id]["line1_frame"] is None:
                    vehicle_data[id]["line1_frame"] = frame_count

            # --- LOGIC: CHECK CROSSING LINE 2 (END) --- [cite: 168]
            if Line2[1] - 10 < cy < Line2[1] + 10:
                # Only calculate if they have already crossed Line 1
                if vehicle_data[id]["line2_frame"] is None and vehicle_data[id]["line1_frame"] is not None:
                    vehicle_data[id]["line2_frame"] = frame_count
                    
                    # --- SPEED CALCULATION [cite: 172-176] ---
                    # 1. Frames Elapsed
                    frames_elapsed = vehicle_data[id]["line2_frame"] - vehicle_data[id]["line1_frame"]
                    
                    # 2. Time Elapsed (Seconds)
                    time_elapsed = frames_elapsed / fps
                    
                    # 3. Speed in m/s (Distance / Time)
                    # Use absolute value in case car goes backwards
                    if time_elapsed > 0:
                        speed_ms = distance_between_lines / time_elapsed
                        speed_kmh = speed_ms * 3.6  # Convert to km/h
                        vehicle_data[id]["speed"] = speed_kmh

            # --- VISUALIZATION --- [cite: 179]
            # Draw Bounding Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            # Draw Center Dot
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            # Display Speed if calculated [cite: 184]
            if vehicle_data[id]["speed"] is not None:
                label = f"{vehicle_data[id]['speed']:.1f} km/h"
                color = (0, 255, 0) # Green for speed
            else:
                label = f"ID: {id}"
                color = (0, 255, 255) # Yellow for just tracking

            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw the Reference Lines [cite: 192]
    cv2.line(frame, (Line1[0], Line1[1]), (Line1[2], Line1[3]), (0, 255, 0), 2) # Green Line
    cv2.line(frame, (Line2[0], Line2[1]), (Line2[2], Line2[3]), (0, 0, 255), 2) # Red Line
    
    # Add Labels for lines
    cv2.putText(frame, "Line 1 (Start)", (10, Line1[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    cv2.putText(frame, "Line 2 (End)", (10, Line2[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    # Show the video
    cv2.imshow("Exam Project: Speed Estimation", frame)

    # Press 'q' to quit [cite: 214]
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()