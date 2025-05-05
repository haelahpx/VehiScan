import cv2
import os
import numpy as np
import easyocr
from sort.sort import *
import datetime
from ultralytics import YOLO
import torch 

# === CONFIG ===
CAR_MODEL_PATH = "models/test.pt"
PLATE_MODEL_PATH = "models/plat_model/weights/best.pt"
VIDEO_INPUT_PATH = os.path.join("assets", "video_testing4.mp4")

# Create unique output filenames
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_video_filename = f"video_result_{timestamp}.mp4"
VIDEO_OUTPUT_PATH = os.path.join("outputs", output_video_filename)
LOG_OUTPUT_PATH = os.path.join("outputs", "logs.txt")

CONFIDENCE_THRESHOLD = 0.25

# === LOAD MODELS AND OCR ===
car_model = YOLO(CAR_MODEL_PATH)
plate_model = YOLO(PLATE_MODEL_PATH)

# Move models to GPU (if available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
car_model.to(device)  # Move the model to GPU (or CPU if CUDA is not available)
plate_model.to(device)  # Same for the plate model

# Check if CUDA is available and print the result
if torch.cuda.is_available():
    print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Using CPU.")

reader = easyocr.Reader(['en'], gpu=True)  # OCR with GPU
tracker = Sort()

# === OPEN VIDEO ===
cap = cv2.VideoCapture(VIDEO_INPUT_PATH)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_number = 0

# Responsive Line Position
LINE_POSITION = int(frame_height * 0.8)
LINE_OFFSET = int(frame_height * 0.01)

# === SETUP VIDEO WRITER ===
os.makedirs("outputs", exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))

# === INIT COUNTING ===
vehicle_count = 0
counted_ids = set()
plate_history = {}  # track_id: list of (text, confidence)
previous_positions = {}  # track_id: previous y position

label_colors = {
    'car': (0, 255, 0),
    'truck': (255, 0, 0),
    'bus': (0, 0, 255),
    'van': (255, 255, 0)
}

# === PROCESS FRAMES ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1
    print(f"Processing: {(frame_number / total_frames) * 100:.2f}%", end="\r")

    # === Detect Vehicles ===
    car_results = car_model.predict(source=frame, conf=CONFIDENCE_THRESHOLD, save=False, verbose=False)
    detections = []
    class_names = []

    for r in car_results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = car_model.names[cls]
            if label.lower() in label_colors:
                detections.append([x1, y1, x2, y2, conf])
                class_names.append(label.lower())

    # Convert detections to NumPy array for SORT
    dets_np = np.array(detections) if detections else np.empty((0, 5))
    tracks = tracker.update(dets_np)

    current_ids = set()
    for track in tracks:
        track_id = int(track[4])
        current_ids.add(track_id)

    to_remove = [tid for tid in previous_positions if tid not in current_ids]
    for tid in to_remove:
        del previous_positions[tid]

    for i, track in enumerate(tracks):
        x1, y1, x2, y2, track_id = map(int, track)
        center_y = int((y1 + y2) / 2)

        previous_y = previous_positions.get(track_id, None)
        previous_positions[track_id] = center_y
        # Count vehicle 
        if previous_y is not None and center_y > previous_y:
            if (previous_y < LINE_POSITION and center_y >= LINE_POSITION) or \
               (LINE_POSITION - LINE_OFFSET < center_y < LINE_POSITION + LINE_OFFSET):
                if track_id not in counted_ids:
                    vehicle_count += 1
                    counted_ids.add(track_id)

        # === Draw bounding box and label ===
        label = class_names[i] if i < len(class_names) else "vehicle"
        box_color = label_colors.get(label, (255, 255, 255))

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(frame, f"{label.capitalize()} {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        # === License Plate Detection ===
        plate_h = (y2 - y1) // 3
        plate_w = (x2 - x1) // 2
        plate_x1 = x1 + (x2 - x1 - plate_w) // 2
        plate_y1 = y2 - plate_h
        plate_x2 = plate_x1 + plate_w
        plate_y2 = y2

        cropped = frame[plate_y1:plate_y2, plate_x1:plate_x2]

        if cropped.size > 0 and cropped.shape[0] > 0 and cropped.shape[1] > 0:
            plate_results = plate_model.predict(source=cropped, conf=CONFIDENCE_THRESHOLD, save=False, verbose=False)
            plate_texts = reader.readtext(cropped)

            if track_id not in plate_history:
                plate_history[track_id] = []

            for result in plate_texts:
                text, confidence = result[1], result[2]
                if len(text.strip()) > 3:
                    plate_history[track_id].append((text, confidence))

            enlarged = cv2.resize(cropped, (180, 60))
        else:
            enlarged = np.ones((60, 180, 3), dtype=np.uint8) * 255

        best_plate = None
        if track_id in plate_history and plate_history[track_id]:
            sorted_plates = sorted(plate_history[track_id], key=lambda x: x[1], reverse=True)
            best_plate = sorted_plates[0][0]

        if best_plate:
            last_char = best_plate.strip()[-1]
            plate_type = "Odd" if last_char.isdigit() and int(last_char) % 2 != 0 else "Even"

            label_img = np.ones((120, 180, 3), dtype=np.uint8) * 255
            cv2.putText(label_img, f"{best_plate}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.putText(label_img, f"({plate_type})", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            label_img[60:120, 0:180] = enlarged

            label_x = max(0, x1)
            label_y = max(0, y1 - 130)
            h, w = label_img.shape[:2]

            if label_y + h < frame.shape[0] and label_x + w < frame.shape[1]:
                frame[label_y:label_y + h, label_x:label_x + w] = label_img

    # === Draw line and counter ===
    cv2.line(frame, (0, LINE_POSITION), (frame_width, LINE_POSITION), (255, 255, 0), 2)
    cv2.putText(frame, f"Count: {vehicle_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    cv2.putText(frame, f"Count: {vehicle_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

    out.write(frame)

# === CLEANUP ===
cap.release()
out.release()
cv2.destroyAllWindows()

# === SAVE LOG FILE ===
with open(LOG_OUTPUT_PATH, "a") as log_file:
    log_file.write(f"\n=== Processed at {datetime.datetime.now()} ===\n")
    log_file.write(f"Processed Video: {VIDEO_OUTPUT_PATH}\n")
    log_file.write(f"Total Vehicles Counted: {vehicle_count}\n")
    log_file.write("Detected Plates:\n")
    for track_id, plates in plate_history.items():
        if plates:
            sorted_plates = sorted(plates, key=lambda x: x[1], reverse=True)
            best_plate = sorted_plates[0][0]
            log_file.write(f"  ID {track_id}: {best_plate}\n")
    log_file.write("="*40 + "\n")

print(f"\nProcessed video saved to {VIDEO_OUTPUT_PATH}")
print(f"All results appended into {LOG_OUTPUT_PATH}")