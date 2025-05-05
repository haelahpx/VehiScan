import cv2
import os
import numpy as np
import easyocr
from sort.sort import *
import datetime
from ultralytics import YOLO
import torch 
import re
import statistics

# === CONFIG ===
CAR_MODEL_PATH = "models/vehicle_model(detrac-16)/weights/best.pt"
PLATE_MODEL_PATH = "models/plat_model/weights/best.pt"
VIDEO_INPUT_PATH = os.path.join("assets", "video_testing4.mp4")

# Create unique output filenames
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_video_filename = f"video_result_{timestamp}.mp4"
VIDEO_OUTPUT_PATH = os.path.join("outputs", output_video_filename)
LOG_OUTPUT_PATH = os.path.join("outputs", "logs.txt")

# Detection parameters
CONFIDENCE_THRESHOLD = 0.25
PLATE_CONFIDENCE_THRESHOLD = 0.4  # Higher confidence for plate detection

# === LOAD MODELS AND OCR ===
car_model = YOLO(CAR_MODEL_PATH)
plate_model = YOLO(PLATE_MODEL_PATH)

# Move models to GPU (if available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
car_model.to(device)
plate_model.to(device)

# Check if CUDA is available and print the result
if torch.cuda.is_available():
    print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Using CPU.")

# Initialize OCR with better parameters
reader = easyocr.Reader(['en'], gpu=True, model_storage_directory='./easyocr_models')
tracker = Sort(max_age=30, min_hits=5)  # Improved tracking parameters

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

# Function to clean plate text
def clean_plate_text(text):
    # Remove spaces and non-alphanumeric chars except dash
    text = re.sub(r'[^A-Z0-9-]', '', text.upper())
    
    # If the text looks like a real plate (at least 2 letters and 2 numbers)
    if len(re.findall(r'[A-Z]', text)) >= 2 and len(re.findall(r'[0-9]', text)) >= 2:
        return text
    return None

# Function to enhance the plate image
def enhance_plate_image(img):
    if img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
        return np.ones((60, 180, 3), dtype=np.uint8) * 255
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(enhanced, 11, 17, 17)
    
    # Convert back to color for display purposes
    enhanced_color = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
    
    # Resize to desired dimensions
    resized = cv2.resize(enhanced_color, (180, 60), interpolation=cv2.INTER_CUBIC)
    
    return resized

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

        # === License Plate Detection with multiple approaches ===
        vehicle_height = y2 - y1
        vehicle_width = x2 - x1
        
        # Approach 1: Bottom third of vehicle
        plate_h1 = vehicle_height // 3
        plate_w1 = vehicle_width // 2
        plate_x1_1 = x1 + (vehicle_width - plate_w1) // 2
        plate_y1_1 = y2 - plate_h1
        
        # Approach 2: Bottom quarter with wider width
        plate_h2 = vehicle_height // 4
        plate_w2 = int(vehicle_width * 0.7)
        plate_x1_2 = x1 + (vehicle_width - plate_w2) // 2
        plate_y1_2 = y2 - plate_h2
        
        # Approach 3: Middle-bottom area
        plate_h3 = vehicle_height // 3
        plate_w3 = int(vehicle_width * 0.6)
        plate_x1_3 = x1 + (vehicle_width - plate_w3) // 2
        plate_y1_3 = y1 + int(vehicle_height * 0.6)
        
        plate_regions = [
            frame[plate_y1_1:y2, plate_x1_1:plate_x1_1+plate_w1],
            frame[plate_y1_2:y2, plate_x1_2:plate_x1_2+plate_w2],
            frame[plate_y1_3:plate_y1_3+plate_h3, plate_x1_3:plate_x1_3+plate_w3]
        ]
        
        best_ocr_results = []
        
        for idx, plate_region in enumerate(plate_regions):
            if plate_region.size == 0 or plate_region.shape[0] <= 10 or plate_region.shape[1] <= 10:
                continue
                
            # Apply plate detection model to refine
            plate_results = plate_model.predict(source=plate_region, conf=PLATE_CONFIDENCE_THRESHOLD, save=False, verbose=False)
            
            # Process detected plates if any
            if len(plate_results) > 0 and len(plate_results[0].boxes) > 0:
                for plate_box in plate_results[0].boxes:
                    px1, py1, px2, py2 = map(int, plate_box.xyxy[0])
                    
                    # Ensure valid coordinates
                    if px1 >= 0 and py1 >= 0 and px2 < plate_region.shape[1] and py2 < plate_region.shape[0]:
                        plate_crop = plate_region[py1:py2, px1:px2]
                        
                        if plate_crop.size > 0 and plate_crop.shape[0] > 5 and plate_crop.shape[1] > 5:
                            # Enhance the plate image for better OCR
                            enhanced_plate = enhance_plate_image(plate_crop)
                            
                            # Run OCR with adjusted parameters
                            plate_texts = reader.readtext(
                                enhanced_plate, 
                                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-',
                                decoder='beamsearch',
                                beamWidth=5,
                                batch_size=1
                            )
                            
                            for detection in plate_texts:
                                bbox, text, conf = detection
                                cleaned_text = clean_plate_text(text)
                                if cleaned_text and conf > 0.4 and len(cleaned_text) >= 4:
                                    best_ocr_results.append((cleaned_text, conf, enhanced_plate))
            
            # If no plate detected by YOLO, try direct OCR on the region
            if not best_ocr_results:
                enhanced_region = enhance_plate_image(plate_region)
                plate_texts = reader.readtext(
                    enhanced_region,
                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-',
                    decoder='beamsearch',
                    beamWidth=5,
                    batch_size=1
                )
                
                for detection in plate_texts:
                    bbox, text, conf = detection
                    cleaned_text = clean_plate_text(text)
                    if cleaned_text and conf > 0.4 and len(cleaned_text) >= 4:
                        best_ocr_results.append((cleaned_text, conf, enhanced_region))
        
        # Get the best result based on confidence
        if best_ocr_results:
            best_ocr_results.sort(key=lambda x: x[1], reverse=True)
            best_text, best_conf, best_image = best_ocr_results[0]
            
            if track_id not in plate_history:
                plate_history[track_id] = []
            
            plate_history[track_id].append((best_text, best_conf))
            enlarged = best_image
        else:
            # Create a blank image if no plate is detected
            enlarged = np.ones((60, 180, 3), dtype=np.uint8) * 255

        # Get the best plate text from history using voting
        best_plate = None
        if track_id in plate_history and plate_history[track_id]:
            # Group by text and calculate mean confidence
            text_groups = {}
            for text, conf in plate_history[track_id]:
                if text not in text_groups:
                    text_groups[text] = []
                text_groups[text].append(conf)
            
            # Find text with highest average confidence and occurrence count
            best_score = 0
            for text, confs in text_groups.items():
                avg_conf = statistics.mean(confs)
                count = len(confs)
                score = avg_conf * count
                if score > best_score:
                    best_score = score
                    best_plate = text

        # Create an improved display for the plate
        if best_plate:
            last_char = best_plate.strip()[-1] if best_plate.strip() else ""
            plate_type = "Odd" if last_char.isdigit() and int(last_char) % 2 != 0 else "Even"
            
            # Create a more visually appealing plate display with rounded corners
            label_img = np.ones((135, 200, 3), dtype=np.uint8) * 245  # Light gray background
            
            # Add border with rounded corners (simulated)
            cv2.rectangle(label_img, (5, 5), (195, 130), (100, 100, 100), 2)
            
            # Add plate text with shadow effect for better visibility
            cv2.putText(label_img, f"{best_plate}", (12, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 50), 3)  # Shadow
            cv2.putText(label_img, f"{best_plate}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)  # Text
            
            # Add plate type with better styling
            cv2.putText(label_img, f"({plate_type})", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)  # Shadow
            cv2.putText(label_img, f"({plate_type})", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 70, 140), 2)  # Blue text
            
            # Add the enhanced plate image
            label_img[70:130, 10:190] = cv2.resize(enlarged, (180, 60))
            
            # Place the label in a better position (top-left of vehicle bounding box)
            label_x = max(0, x1)
            label_y = max(0, y1 - 140)
            h, w = label_img.shape[:2]
            
            # Make sure it fits within the frame
            if label_y + h > frame.shape[0]:
                label_y = max(0, frame.shape[0] - h)
            if label_x + w > frame.shape[1]:
                label_x = max(0, frame.shape[1] - w)
            
            # Create a semi-transparent overlay for better visibility
            overlay = frame.copy()
            cv2.rectangle(overlay, (label_x-5, label_y-5), (label_x+w+5, label_y+h+5), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            # Add the plate info display
            frame[label_y:label_y+h, label_x:label_x+w] = label_img

    # === Draw line and counter ===
    cv2.line(frame, (0, LINE_POSITION), (frame_width, LINE_POSITION), (255, 255, 0), 2)
    
    # Add a more visible counter with background
    counter_bg = frame.copy()
    cv2.rectangle(counter_bg, (10, 10), (250, 60), (0, 0, 0), -1)
    cv2.addWeighted(counter_bg, 0.6, frame, 0.4, 0, frame)
    
    cv2.putText(frame, f"Count: {vehicle_count}", (22, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4)  # Shadow
    cv2.putText(frame, f"Count: {vehicle_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)  # Text

    out.write(frame)

# === CLEANUP ===
cap.release()
out.release()
cv2.destroyAllWindows()

# === SAVE LOG FILE WITH MORE DETAILS ===
with open(LOG_OUTPUT_PATH, "a") as log_file:
    log_file.write(f"\n=== Processed at {datetime.datetime.now()} ===\n")
    log_file.write(f"Processed Video: {VIDEO_OUTPUT_PATH}\n")
    log_file.write(f"Total Vehicles Counted: {vehicle_count}\n")
    log_file.write("Detected Plates:\n")
    
    # Group plates by type (odd/even)
    odd_plates = []
    even_plates = []
    
    for track_id, plates in plate_history.items():
        if plates:
            # Group by text and calculate mean confidence
            text_groups = {}
            for text, conf in plates:
                if text not in text_groups:
                    text_groups[text] = []
                text_groups[text].append(conf)
            
            # Find text with highest average confidence
            best_text = None
            best_score = 0
            for text, confs in text_groups.items():
                avg_conf = statistics.mean(confs)
                count = len(confs)
                score = avg_conf * count
                if score > best_score:
                    best_score = score
                    best_text = text
            
            if best_text:
                last_char = best_text.strip()[-1] if best_text.strip() else ""
                is_odd = last_char.isdigit() and int(last_char) % 2 != 0
                
                if is_odd:
                    odd_plates.append((track_id, best_text))
                else:
                    even_plates.append((track_id, best_text))
                
                log_file.write(f"  ID {track_id}: {best_text} ({'Odd' if is_odd else 'Even'})\n")
    
    log_file.write(f"\nTotal Odd Plates: {len(odd_plates)}\n")
    log_file.write(f"Total Even Plates: {len(even_plates)}\n")
    log_file.write("="*40 + "\n")

print(f"\nProcessed video saved to {VIDEO_OUTPUT_PATH}")
print(f"All results appended into {LOG_OUTPUT_PATH}")