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
PLATE_MODEL_PATH = "models/car_plat6.pt"
VIDEO_INPUT_PATH = os.path.join("assets", "video_testing5.mp4")
CONFIDENCE_THRESHOLD = 0.5
PLATE_CONFIDENCE_THRESHOLD = 0.4  # Slightly lower threshold for plate detection

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
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.25)

# === OPEN VIDEO ===
cap = cv2.VideoCapture(VIDEO_INPUT_PATH)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_number = 0

LINE_POSITION = int(frame_height * 0.8)
LINE_OFFSET = int(frame_height * 0.01)

vehicle_count = 0
counted_ids = set()
plate_history = {}

# === IMPROVED TRACKING DATA STRUCTURES ===
previous_positions = {}  # Track position history
track_labels = {}        # Final assigned labels
class_history = {}       # Class voting history
position_history = {}    # Position smoothing
track_quality = {}       # Track quality assessment

CLASS_HISTORY_MAX = 10
CLASS_SWITCH_THRESHOLD = 3

label_colors = {
    'car': (0, 255, 0),
    'truck': (255, 0, 0),
    'bus': (0, 0, 255),
    'van': (255, 255, 0)
}

def preprocess_plate(cropped):
    """Improved plate preprocessing for better OCR results"""
    # Convert to grayscale
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Apply morphological operations to clean up the image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Apply sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(morph, -1, kernel)
    
    return sharpened

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

def clean_plate_text(text):
    """Clean and validate plate text"""
    # Remove unwanted characters
    text = ''.join(c for c in text if c.isalnum() or c in '- ')
    
    # Common false positives to filter out
    false_positives = ['car', 'truck', 'bus', 'van', 'motor', 'vehicle']
    if any(fp in text.lower() for fp in false_positives):
        return None
    
    # Minimum length requirement
    if len(text) < 4:
        return None
        
    return text.upper()

def read_plate_with_retry(cropped):
    """Try multiple OCR approaches to read plate"""
    # Try with preprocessing first
    preprocessed = preprocess_plate(cropped)
    results = reader.readtext(preprocessed, detail=1, paragraph=False, min_size=10)
    
    if not results:
        # Fallback to original image with different parameters
        results = reader.readtext(cropped, detail=1, paragraph=False, 
                                min_size=5, text_threshold=0.4, low_text=0.3)
    
    valid_results = []
    for result in results:
        text, confidence = result[1], result[2]
        cleaned_text = clean_plate_text(text)
        if cleaned_text and confidence > 0.4:
            valid_results.append((cleaned_text, confidence))
    
    return valid_results

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
        
        center_y = int((y1 + y2) / 2)
        previous_y = previous_positions.get(track_id, None)
        previous_positions[track_id] = center_y

        if previous_y is not None and center_y > previous_y:
            if (previous_y < LINE_POSITION and center_y >= LINE_POSITION) or \
               (LINE_POSITION - LINE_OFFSET < center_y < LINE_POSITION + LINE_OFFSET):
                if track_id not in counted_ids and quality > 0.5:
                    vehicle_count += 1
                    counted_ids.add(track_id)

        label = track_labels[track_id]
        box_color = label_colors.get(label, (255, 255, 255))

        cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.rectangle(display_frame, (x1, y1), (x1 + 120, y1 + 20), box_color, -1)
        cv2.putText(display_frame, f"{label} {track_id} Q:{quality:.2f}", (x1 + 2, y1 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # === IMPROVED PLATE DETECTION AND READING ===
        margin = 10
        vehicle_x1 = max(0, x1 - margin)
        vehicle_y1 = max(0, y1 - margin)
        vehicle_x2 = min(frame.shape[1], x2 + margin)
        vehicle_y2 = min(frame.shape[0], y2 + margin)
        
        vehicle_crop = ocr_frame[vehicle_y1:vehicle_y2, vehicle_x1:vehicle_x2]
        
        best_plate_box = None
        highest_conf = 0
        
        if vehicle_crop.size > 0 and vehicle_crop.shape[0] > 0 and vehicle_crop.shape[1] > 0:
            plate_results = plate_model.predict(source=vehicle_crop, conf=PLATE_CONFIDENCE_THRESHOLD, verbose=False)
            
            for plate_r in plate_results:
                plate_boxes = plate_r.boxes
                for plate_box in plate_boxes:
                    px1, py1, px2, py2 = map(int, plate_box.xyxy[0])
                    plate_conf = float(plate_box.conf[0])
                    
                    plate_x1 = vehicle_x1 + px1
                    plate_y1 = vehicle_y1 + py1
                    plate_x2 = vehicle_x1 + px2
                    plate_y2 = vehicle_y1 + py2
                    
                    if plate_conf > highest_conf:
                        highest_conf = plate_conf
                        best_plate_box = (plate_x1, plate_y1, plate_x2, plate_y2)
        
        # Only process plate if we have a good detection
        if best_plate_box is not None:
            plate_x1, plate_y1, plate_x2, plate_y2 = best_plate_box
            cv2.rectangle(display_frame, best_plate_box[:2], best_plate_box[2:], (0, 255, 0), 2)
            
            cropped = ocr_frame[plate_y1:plate_y2, plate_x1:plate_x2]

            if cropped.size > 0 and cropped.shape[0] > 0 and cropped.shape[1] > 0:
                plate_texts = read_plate_with_retry(cropped)

                if track_id not in plate_history:
                    plate_history[track_id] = []

                for text, confidence in plate_texts:
                    plate_history[track_id].append((text, confidence))

                # Get the best plate for this track
                best_plate = None
                if plate_history[track_id]:
                    sorted_plates = sorted(plate_history[track_id], key=lambda x: x[1], reverse=True)
                    best_plate = sorted_plates[0][0]

                if best_plate:
                    # Determine odd/even
                    last_char = best_plate.strip()[-1]
                    plate_type = "Odd" if last_char.isdigit() and int(last_char) % 2 != 0 else "Even"
                    
                    # Draw plate information
                    label_box_height = 20
                    label_box_width = x2 - x1
                    cv2.rectangle(display_frame, (x1, y2 - label_box_height), (x1 + label_box_width, y2), (255, 255, 255), -1)
                    cv2.putText(display_frame, f"{best_plate} ({plate_type})", (x1 + 2, y2 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Remove old tracks
    to_remove = [tid for tid in previous_positions if tid not in current_ids]
    for tid in to_remove:
        if tid in previous_positions:
            del previous_positions[tid]
        if tid in track_labels:
            del track_labels[tid]
        if tid in class_history:
            del class_history[tid]
        if tid in position_history:
            del position_history[tid]
        if tid in track_quality:
            del track_quality[tid]

    # Draw counting line
    cv2.line(display_frame, (0, LINE_POSITION), (frame_width, LINE_POSITION), (255, 255, 0), 2)
    cv2.putText(display_frame, f"Count: {vehicle_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    cv2.putText(display_frame, f"Count: {vehicle_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

    cv2.imshow("Vehicle & Plate Detection", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Real-time video processing finished.")