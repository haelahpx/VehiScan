import cv2
import os
import numpy as np
import easyocr
from sort.sort import *
import datetime
from ultralytics import YOLO
import torch
from collections import Counter, deque

# === CONFIG ===
CAR_MODEL_PATH = "models/best.pt"
PLATE_MODEL_PATH = "models/yusuf/best.pt"
VIDEO_INPUT_PATH = os.path.join("assets", "video_testing4.mp4")
CONFIDENCE_THRESHOLD = 0.5

# === IMPROVED ROI CONFIGURATION ===
# Define ROI as a rectangular region for counting
ROI_Y1 = int(0.7 * 1080)  # Adjust based on your video height (e.g., 1080p)
ROI_Y2 = int(0.9 * 1080)
ROI_COLOR = (255, 255, 0)  # Yellow for ROI visualization

# === NEW COUNTING PARAMETERS ===
COUNTING_THRESHOLD = 0.5  # Crossing threshold within ROI (percentage of ROI height)
MIN_DETECTION_QUALITY = 0.6  # Minimum track quality to count a vehicle
MIN_FRAMES_IN_TRACK = 10  # Minimum frames a vehicle must be tracked to be valid

# === LOAD MODELS AND OCR ===
car_model = YOLO(CAR_MODEL_PATH)
plate_model = YOLO(PLATE_MODEL_PATH)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
car_model.to(device)
plate_model.to(device)

if torch.cuda.is_available():
    print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Using CPU.")

reader = easyocr.Reader(['en'], gpu=True)

# === IMPROVED SORT TRACKER INITIALIZATION ===
# Adjusted parameters for more robust tracking
tracker = Sort(max_age=45, min_hits=3, iou_threshold=0.3)  # Increased max_age, stricter iou_threshold

# === OPEN VIDEO ===
cap = cv2.VideoCapture(VIDEO_INPUT_PATH)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_number = 0

vehicle_count = 0
counted_ids = set()
plate_history = {}

# === NEW TRACKING DATA STRUCTURES ===
track_positions = {}  # Store position history for each track
track_labels = {}
class_history = {}
position_history = {}
track_quality = {}
CLASS_HISTORY_MAX = 10
CLASS_SWITCH_THRESHOLD = 3

# Track ROI crossing
roi_crossing_status = {}  # 0=not started, 1=in progress, 2=completed

label_colors = {
    'car': (0, 255, 0),
    'truck': (255, 0, 0),
    'bus': (0, 0, 255),
    'van': (255, 255, 0)
}

def preprocess_plate(cropped):
    """Preprocess plate image to improve OCR with minimal slowdown."""
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return morph

def get_dominant_class(track_id, new_class=None, confidence=0):
    """Maintain class history and return dominant class using voting"""
    if track_id not in class_history:
        class_history[track_id] = []
    if new_class:
        class_history[track_id].append((new_class, confidence))
        if len(class_history[track_id]) > CLASS_HISTORY_MAX:
            class_history[track_id].pop(0)
    if not class_history[track_id]:
        return new_class
    votes = {}
    for cls, conf in class_history[track_id]:
        if cls not in votes:
            votes[cls] = 0
        votes[cls] += conf
    if votes:
        return max(votes.items(), key=lambda x: x[1])[0]
    return "vehicle"

def smooth_position(track_id, box, alpha=0.7):
    """Smooth bounding box position using exponential moving average"""
    if track_id not in position_history:
        position_history[track_id] = box
        return box
    x1, y1, x2, y2 = box
    px1, py1, px2, py2 = position_history[track_id]
    sx1 = int(alpha * px1 + (1 - alpha) * x1)
    sy1 = int(alpha * py1 + (1 - alpha) * y1)
    sx2 = int(alpha * px2 + (1 - alpha) * x2)
    sy2 = int(alpha * py2 + (1 - alpha) * y2)
    position_history[track_id] = (sx1, sy1, sx2, sy2)
    return (sx1, sy1, sx2, sy2)

def assess_track_quality(track_id, detection_count, is_detected):
    """Assess track quality based on detection consistency"""
    if track_id not in track_quality:
        track_quality[track_id] = {'count': 0, 'detections': 0}
    track_quality[track_id]['count'] += 1
    if is_detected:
        track_quality[track_id]['detections'] += 1
    quality = track_quality[track_id]['detections'] / track_quality[track_id]['count']
    return quality

# Create an initially empty display frame
display_frame = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1
    print(f"Processing: {(frame_number / total_frames) * 100:.2f}%", end="\r")

    ocr_frame = frame.copy()
    display_frame = frame.copy()

    car_results = car_model.predict(source=frame, conf=CONFIDENCE_THRESHOLD, save=False, verbose=False)
    detections = []
    detection_classes = []
    detection_confidences = []

    for r in car_results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = car_model.names[cls].lower()
            if label in label_colors:
                detections.append([x1, y1, x2, y2, conf])
                detection_classes.append(label)
                detection_confidences.append(conf)

    dets_np = np.array(detections) if detections else np.empty((0, 5))
    tracks = tracker.update(dets_np)

    current_ids = set()
    detected_ids = set()

    detection_track_map = {}
    if len(detections) > 0 and len(tracks) > 0:
        for det_idx, det in enumerate(detections):
            x1, y1, x2, y2, _ = det
            det_box = [x1, y1, x2, y2]
            best_iou = 0
            best_track_idx = -1
            for track_idx, track in enumerate(tracks):
                tx1, ty1, tx2, ty2, track_id = map(int, track)
                track_box = [tx1, ty1, tx2, ty2]
                xx1 = max(det_box[0], track_box[0])
                yy1 = max(det_box[1], track_box[1])
                xx2 = min(det_box[2], track_box[2])
                yy2 = min(det_box[3], track_box[3])
                w = max(0, xx2 - xx1)
                h = max(0, yy2 - yy1)
                intersection = w * h
                det_area = (det_box[2] - det_box[0]) * (det_box[3] - det_box[1])
                track_area = (track_box[2] - track_box[0]) * (track_box[3] - track_box[1])
                union = det_area + track_area - intersection
                iou = intersection / union if union > 0 else 0
                if iou > best_iou and iou > 0.5:
                    best_iou = iou
                    best_track_idx = track_idx
            if best_track_idx >= 0:
                track_id = int(tracks[best_track_idx][4])
                detection_track_map[track_id] = (det_idx, best_iou)
                detected_ids.add(track_id)

    for track_idx, track in enumerate(tracks):
        x1, y1, x2, y2, track_id = map(int, track)
        track_id = int(track_id)
        current_ids.add(track_id)

        was_detected = track_id in detected_ids
        if was_detected:
            det_idx, iou = detection_track_map[track_id]
            detected_class = detection_classes[det_idx]
            detected_conf = detection_confidences[det_idx]
        else:
            detected_class = None
            detected_conf = 0

        quality = assess_track_quality(track_id, frame_number, was_detected)
        x1, y1, x2, y2 = smooth_position(track_id, (x1, y1, x2, y2))
        dominant_class = get_dominant_class(track_id, detected_class, detected_conf)

        if track_id not in track_labels:
            track_labels[track_id] = dominant_class
        elif was_detected and detected_class != track_labels[track_id]:
            if detected_conf > CONFIDENCE_THRESHOLD + 0.1:
                if dominant_class != track_labels[track_id]:
                    class_counts = Counter([c for c, _ in class_history[track_id][-CLASS_SWITCH_THRESHOLD:]])
                    if class_counts.get(dominant_class, 0) >= CLASS_SWITCH_THRESHOLD:
                        track_labels[track_id] = dominant_class

        # === IMPROVED ROI-BASED COUNTING ===
        # Calculate the bottom center of the vehicle
        center_x = int((x1 + x2) / 2)
        bottom_y = y2  # Use bottom of bounding box for more accurate position
        
        # Track the vehicle's positions
        if track_id not in track_positions:
            track_positions[track_id] = deque(maxlen=30)  # Store last 30 positions
            roi_crossing_status[track_id] = 0  # Not started crossing
        
        # Add current position
        track_positions[track_id].append((center_x, bottom_y))
        
        # Define ROI crossing threshold (percentage of ROI height)
        roi_height = ROI_Y2 - ROI_Y1
        threshold_y = ROI_Y1 + (roi_height * COUNTING_THRESHOLD)
        
        # Only process vehicles with sufficient track history and quality
        valid_track = (quality >= MIN_DETECTION_QUALITY and 
                     len(track_positions[track_id]) >= MIN_FRAMES_IN_TRACK)
        
        # Determine vehicle's position relative to ROI
        in_roi = ROI_Y1 <= bottom_y <= ROI_Y2
        crossed_threshold = bottom_y >= threshold_y
        
        # Handle ROI crossing state machine
        if valid_track and track_id not in counted_ids:
            # Check if this is the first time we're seeing vehicle in ROI
            if roi_crossing_status[track_id] == 0 and in_roi:
                roi_crossing_status[track_id] = 1  # Started crossing
            
            # Check if vehicle has crossed the counting threshold within ROI
            elif roi_crossing_status[track_id] == 1 and crossed_threshold:
                # Check direction (must be moving downward)
                if len(track_positions[track_id]) >= 2:
                    prev_pos = track_positions[track_id][-2]
                    if bottom_y > prev_pos[1]:  # Moving downward
                        vehicle_count += 1
                        counted_ids.add(track_id)
                        roi_crossing_status[track_id] = 2  # Completed crossing
        
        # Visualization for debug
        roi_status_color = (0, 0, 255)  # Default red
        if roi_crossing_status[track_id] == 1:
            roi_status_color = (0, 255, 255)  # Yellow for in progress
        elif roi_crossing_status[track_id] == 2:
            roi_status_color = (0, 255, 0)  # Green for counted
        
        # Draw a small circle on the bottom center to show the counting point
        cv2.circle(display_frame, (center_x, bottom_y), 4, roi_status_color, -1)

        label = track_labels[track_id]
        box_color = label_colors.get(label, (255, 255, 255))

        cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.rectangle(display_frame, (x1, y1), (x1 + 120, y1 + 20), box_color, -1)
        cv2.putText(display_frame, f"{label} {track_id} Q:{quality:.2f}", (x1 + 2, y1 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        plate_h = (y2 - y1) // 4
        plate_w = (x2 - x1) // 2
        margin = 2
        plate_x1 = max(x1 + (x2 - x1 - plate_w) // 2 - margin, 0)
        plate_y1 = max(y2 - plate_h - margin, 0)
        plate_x2 = min(plate_x1 + plate_w + margin, frame.shape[1])
        plate_y2 = min(plate_y1 + plate_h + margin, frame.shape[0])

        cv2.rectangle(display_frame, (plate_x1, plate_y1), (plate_x2, plate_y2), (0, 255, 255), 1)
        cropped = ocr_frame[plate_y1:plate_y2, plate_x1:plate_x2]

        if cropped.size > 0 and cropped.shape[0] > 0 and cropped.shape[1] > 0:
            preprocessed = preprocess_plate(cropped)
            plate_texts = reader.readtext(preprocessed, detail=1, paragraph=False, min_size=5)
            if len(plate_texts) == 0:
                plate_texts = reader.readtext(cropped, detail=1, paragraph=False, min_size=5)
            if track_id not in plate_history:
                plate_history[track_id] = []
            for result in plate_texts:
                text, confidence = result[1], result[2]
                text = text.strip().replace(" ", "")
                if any(pattern in text.lower() for pattern in ["car", "truck", "bus", "van"]):
                    continue
                if text.isdigit() and len(text) < 3:
                    continue
                if len(text) >= 4 and confidence >= 0.4:
                    if all(c.isalnum() or c in "-." for c in text):
                        plate_history[track_id].append((text, confidence))
            enlarged = cv2.resize(cropped, (180, 60))
        else:
            enlarged = np.ones((60, 180, 3), dtype=np.uint8) * 255

        best_plate = None
        if track_id in plate_history and plate_history[track_id]:
            sorted_plates = sorted(plate_history[track_id], key=lambda x: x[1], reverse=True)
            best_plate = sorted_plates[0][0]

        if best_plate:
            is_valid_plate = True
            lower_plate = best_plate.lower()
            for vehicle_type in ["car", "truck", "bus", "van"]:
                if vehicle_type in lower_plate:
                    is_valid_plate = False
                    break
            if is_valid_plate:
                last_char = best_plate.strip()[-1]
                plate_type = "Odd" if last_char.isdigit() and int(last_char) % 2 != 0 else "Even"
                label_box_height = 20
                label_box_width = x2 - x1
                cv2.rectangle(display_frame, (x1, y2 - label_box_height), (x1 + label_box_width, y2), (255, 255, 255), -1)
                cv2.putText(display_frame, f"{best_plate} ({plate_type})", (x1 + 2, y2 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                thumb_w = 120
                thumb_h = 40
                thumb_x = x2 - thumb_w - 5
                thumb_y = y1 + 5
                thumb_x = max(x1, thumb_x)
                thumb_y = max(y1, thumb_y)
                available_width = x2 - thumb_x - 5
                if available_width < thumb_w:
                    thumb_w = available_width
                available_height = y2 - thumb_y - 5
                if available_height < thumb_h:
                    thumb_h = available_height
                if thumb_w > 10 and thumb_h > 10:
                    thumb_img = cv2.resize(enlarged, (thumb_w, thumb_h))
                    cv2.rectangle(display_frame, (thumb_x, thumb_y), (thumb_x + thumb_w, thumb_y + thumb_h), (0, 255, 255), 1)
                    if (thumb_x + thumb_w <= display_frame.shape[1] and 
                        thumb_y + thumb_h <= display_frame.shape[0]):
                        display_frame[thumb_y:thumb_y + thumb_h, thumb_x:thumb_x + thumb_w] = thumb_img

    # Remove old tracks
    to_remove = [tid for tid in roi_crossing_status if tid not in current_ids]
    for tid in to_remove:
        if tid in track_positions:
            del track_positions[tid]
        if tid in track_labels:
            del track_labels[tid]
        if tid in class_history:
            del class_history[tid]
        if tid in position_history:
            del position_history[tid]
        if tid in track_quality:
            del track_quality[tid]
        if tid in roi_crossing_status:
            del roi_crossing_status[tid]

    # === Draw ROI Region ===
    cv2.rectangle(display_frame, (0, ROI_Y1), (frame_width, ROI_Y2), ROI_COLOR, 2)
    
    # Draw threshold line
    threshold_y = int(ROI_Y1 + ((ROI_Y2 - ROI_Y1) * COUNTING_THRESHOLD))
    cv2.line(display_frame, (0, threshold_y), (frame_width, threshold_y), (0, 0, 255), 1)
    
    cv2.putText(display_frame, "ROI", (10, ROI_Y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ROI_COLOR, 2)
    cv2.putText(display_frame, f"Count: {vehicle_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    cv2.putText(display_frame, f"Count: {vehicle_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

    cv2.imshow("Vehicle & Plate Detection", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Real-time video processing finished.")