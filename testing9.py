import cv2
import os
import numpy as np
import easyocr
from sort.sort import *
import datetime
from ultralytics import YOLO
import torch
from collections import Counter, deque
import re

# === CONFIG ===
CAR_MODEL_PATH = "models/best.pt"
PLATE_MODEL_PATH = "models/car_plat6.pt"
VIDEO_INPUT_PATH = os.path.join("assets", "video_testing4.mp4")
VEHICLE_CONFIDENCE_THRESHOLD = 0.5
PLATE_CONFIDENCE_THRESHOLD = 0.4

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

# === TRACKER INITIALIZATION ===
vehicle_tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.25)
plate_tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.4)

# === VIDEO SETUP ===
cap = cv2.VideoCapture(VIDEO_INPUT_PATH)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_number = 0

LINE_POSITION = int(frame_height * 0.8)
LINE_OFFSET = int(frame_height * 0.01)

vehicle_count = 0
counted_ids = set()

# === TRACKING DATA STRUCTURES ===
previous_positions = {}
track_labels = {}
class_history = {}
position_history = {}
track_quality = {}
plate_to_vehicle_map = {}
plate_ocr_history = {}

CLASS_HISTORY_MAX = 10
CLASS_SWITCH_THRESHOLD = 3
PLATE_PATTERN = re.compile(r'^[A-Z0-9\-\.]{4,10}$')

label_colors = {
    'car': (0, 255, 0),
    'truck': (255, 0, 0),
    'bus': (0, 0, 255),
    'van': (255, 255, 0)
}

def preprocess_plate(cropped):
    """Enhanced plate preprocessing with multiple techniques"""
    if cropped.size == 0 or cropped.shape[0] == 0 or cropped.shape[1] == 0:
        return None
    
    results = []
    
    # Version 1: Standard grayscale with CLAHE
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    _, thresh1 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results.append(thresh1)
    
    # Version 2: Increased contrast
    adjusted = cv2.convertScaleAbs(gray, alpha=1.3, beta=0)
    _, thresh2 = cv2.threshold(adjusted, 110, 255, cv2.THRESH_BINARY)
    results.append(thresh2)
    
    # Version 3: Edge enhancement
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 100, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    results.append(dilated)
    
    # Version 4: Original color image
    resized = cv2.resize(cropped, (cropped.shape[1]*2, cropped.shape[0]*2))
    results.append(resized)
    
    return results

def is_valid_plate(text):
    """Check if detected text is likely a valid license plate"""
    if not text:
        return False
    
    cleaned = text.replace(" ", "")
    
    if len(cleaned) < 4 or len(cleaned) > 10:
        return False
    
    lower_text = cleaned.lower()
    if any(vehicle_type in lower_text for vehicle_type in ["car", "truck", "bus", "van"]):
        return False
    
    has_letter = any(c.isalpha() for c in cleaned)
    has_number = any(c.isdigit() for c in cleaned)
    
    special_chars = sum(1 for c in cleaned if not c.isalnum())
    if special_chars > 2:
        return False
    
    return has_letter or has_number

def determine_plate_type(plate_text):
    """Determine if plate is odd or even based on last digit"""
    if not plate_text:
        return "Unknown"
    
    digits = [c for c in plate_text if c.isdigit()]
    
    if not digits:
        return "Unknown"
    
    last_digit = int(digits[-1])
    return "Odd" if last_digit % 2 != 0 else "Even"

def get_dominant_class(track_id, new_class=None, confidence=0):
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
    if track_id not in track_quality:
        track_quality[track_id] = {'count': 0, 'detections': 0}
    
    track_quality[track_id]['count'] += 1
    if is_detected:
        track_quality[track_id]['detections'] += 1
    
    quality = track_quality[track_id]['detections'] / track_quality[track_id]['count']
    return quality

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0.0

def assign_plates_to_vehicles(vehicle_tracks, plate_tracks):
    vehicle_plate_assignments = {}
    plate_vehicle_assignments = {}
    
    for plate_track in plate_tracks:
        plate_id = int(plate_track[4])
        plate_box = plate_track[:4].astype(int)
        best_iou = 0
        best_vehicle_id = None
        
        for vehicle_track in vehicle_tracks:
            vehicle_id = int(vehicle_track[4])
            vehicle_box = vehicle_track[:4].astype(int)
            
            px_center = (plate_box[0] + plate_box[2]) / 2
            py_center = (plate_box[1] + plate_box[3]) / 2
            
            vehicle_width = vehicle_box[2] - vehicle_box[0]
            vehicle_height = vehicle_box[3] - vehicle_box[1]
            
            expanded_box = [
                vehicle_box[0] - int(vehicle_width * 0.05),
                vehicle_box[1] - int(vehicle_height * 0.05),
                vehicle_box[2] + int(vehicle_width * 0.05),
                vehicle_box[3] + int(vehicle_height * 0.05)
            ]
            
            if (expanded_box[0] <= px_center <= expanded_box[2] and 
                expanded_box[1] <= py_center <= expanded_box[3]):
                
                iou = calculate_iou(expanded_box, plate_box)
                
                plate_width = plate_box[2] - plate_box[0]
                plate_height = plate_box[3] - plate_box[1]
                
                if (plate_width < vehicle_width * 0.8 and 
                    plate_height < vehicle_height * 0.8):
                    
                    if py_center > (vehicle_box[1] + vehicle_box[3]) / 2:
                        iou *= 1.5
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_vehicle_id = vehicle_id
        
        if plate_id in plate_to_vehicle_map and best_vehicle_id is not None:
            prev_vehicle_id = plate_to_vehicle_map[plate_id]
            
            if prev_vehicle_id in [int(vt[4]) for vt in vehicle_tracks]:
                for vehicle_track in vehicle_tracks:
                    if int(vehicle_track[4]) == prev_vehicle_id:
                        prev_vehicle_box = vehicle_track[:4].astype(int)
                        prev_iou = calculate_iou(prev_vehicle_box, plate_box)
                        
                        if prev_iou > 0.1 and best_iou < prev_iou * 1.5:
                            best_vehicle_id = prev_vehicle_id
                            best_iou = prev_iou
        
        if best_vehicle_id is not None and best_iou > 0.1:
            if best_vehicle_id not in vehicle_plate_assignments:
                vehicle_plate_assignments[best_vehicle_id] = []
            
            vehicle_plate_assignments[best_vehicle_id].append((plate_box, plate_id))
            plate_to_vehicle_map[plate_id] = best_vehicle_id
            plate_vehicle_assignments[plate_id] = best_vehicle_id
    
    return vehicle_plate_assignments, plate_vehicle_assignments

def process_text_candidates(text_candidates):
    if not text_candidates:
        return None, 0
    
    text_counts = {}
    text_confidences = {}
    
    for text, conf in text_candidates:
        cleaned_text = text.strip().replace(" ", "").upper()
        
        if not is_valid_plate(cleaned_text):
            continue
        
        if cleaned_text not in text_counts:
            text_counts[cleaned_text] = 0
            text_confidences[cleaned_text] = []
        
        text_counts[cleaned_text] += 1
        text_confidences[cleaned_text].append(conf)
    
    best_text = None
    best_score = 0
    
    for text, count in text_counts.items():
        avg_conf = sum(text_confidences[text]) / len(text_confidences[text])
        score = count * avg_conf
        
        if score > best_score:
            best_score = score
            best_text = text
    
    if best_text:
        avg_confidence = sum(text_confidences[best_text]) / len(text_confidences[best_text])
        return best_text, avg_confidence
    
    return None, 0

def update_ocr_history(vehicle_id, plate_text, confidence):
    if not plate_text:
        return
    
    if vehicle_id not in plate_ocr_history:
        plate_ocr_history[vehicle_id] = {}
    
    if plate_text not in plate_ocr_history[vehicle_id]:
        plate_ocr_history[vehicle_id][plate_text] = []
    
    plate_ocr_history[vehicle_id][plate_text].append(confidence)
    
    if len(plate_ocr_history[vehicle_id][plate_text]) > 10:
        plate_ocr_history[vehicle_id][plate_text] = plate_ocr_history[vehicle_id][plate_text][-10:]

def get_best_plate_text(vehicle_id):
    if vehicle_id not in plate_ocr_history:
        return None, 0
    
    best_text = None
    best_score = 0
    
    for text, confidences in plate_ocr_history[vehicle_id].items():
        if len(confidences) >= 2:
            avg_conf = sum(confidences) / len(confidences)
            score = len(confidences) * avg_conf
            
            if score > best_score:
                best_score = score
                best_text = text
    
    if best_text and best_score > 1.0:
        avg_confidence = sum(plate_ocr_history[vehicle_id][best_text]) / len(plate_ocr_history[vehicle_id][best_text])
        return best_text, avg_confidence
    
    return None, 0

# Main processing loop
frame_counter = 0
fps_start_time = datetime.datetime.now()
fps = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1
    frame_counter += 1
    
    # Calculate FPS
    current_time = datetime.datetime.now()
    time_diff = (current_time - fps_start_time).total_seconds()
    if time_diff >= 1.0:
        fps = frame_counter / time_diff
        frame_counter = 0
        fps_start_time = current_time
    
    print(f"Processing: {(frame_number / total_frames) * 100:.2f}% | FPS: {fps:.1f}", end="\r")

    ocr_frame = frame.copy()
    display_frame = frame.copy()

    # === VEHICLE DETECTION ===
    car_results = car_model.predict(source=frame, conf=VEHICLE_CONFIDENCE_THRESHOLD, save=False, verbose=False)
    vehicle_detections = []
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
                vehicle_detections.append([x1, y1, x2, y2, conf])
                detection_classes.append(label)
                detection_confidences.append(conf)

    # === PLATE DETECTION ===
    plate_results = plate_model.predict(source=frame, conf=PLATE_CONFIDENCE_THRESHOLD, save=False, verbose=False)
    plate_detections = []
    
    for r in plate_results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            if (x2 - x1) > 5 and (y2 - y1) > 5:
                plate_detections.append([x1, y1, x2, y2, conf])
    
    # Track vehicles and plates
    vehicle_dets_np = np.array(vehicle_detections) if vehicle_detections else np.empty((0, 5))
    vehicle_tracks = vehicle_tracker.update(vehicle_dets_np)
    
    plate_dets_np = np.array(plate_detections) if plate_detections else np.empty((0, 5))
    plate_tracks = plate_tracker.update(plate_dets_np)
    
    # Assign plates to vehicles
    vehicle_plate_assignments, plate_vehicle_map = assign_plates_to_vehicles(vehicle_tracks, plate_tracks)
    
    # Process each vehicle track
    current_vehicle_ids = set()
    detected_vehicle_ids = set()
    detection_track_map = {}
    
    if len(vehicle_detections) > 0 and len(vehicle_tracks) > 0:
        for det_idx, det in enumerate(vehicle_detections):
            x1, y1, x2, y2, _ = det
            det_box = [x1, y1, x2, y2]
            
            best_iou = 0
            best_track_idx = -1
            
            for track_idx, track in enumerate(vehicle_tracks):
                tx1, ty1, tx2, ty2, track_id = map(int, track)
                track_box = [tx1, ty1, tx2, ty2]
                
                iou = calculate_iou(det_box, track_box)
                
                if iou > best_iou and iou > 0.5:
                    best_iou = iou
                    best_track_idx = track_idx
            
            if best_track_idx >= 0:
                track_id = int(vehicle_tracks[best_track_idx][4])
                detection_track_map[track_id] = (det_idx, best_iou)
                detected_vehicle_ids.add(track_id)

    for track_idx, track in enumerate(vehicle_tracks):
        x1, y1, x2, y2, track_id = map(int, track)
        track_id = int(track_id)
        current_vehicle_ids.add(track_id)
        
        was_detected = track_id in detected_vehicle_ids
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
            if detected_conf > VEHICLE_CONFIDENCE_THRESHOLD + 0.1:
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

        # === PROCESS LICENSE PLATES ===
        if track_id in vehicle_plate_assignments:
            best_plate_box = None
            best_plate_id = None
            
            vehicle_plates = vehicle_plate_assignments[track_id]
            
            if vehicle_plates:
                largest_area = 0
                for plate_box, plate_id in vehicle_plates:
                    px1, py1, px2, py2 = plate_box
                    area = (px2 - px1) * (py2 - py1)
                    if area > largest_area:
                        largest_area = area
                        best_plate_box = plate_box
                        best_plate_id = plate_id
                
                px1, py1, px2, py2 = best_plate_box
                
                if f"plate_{best_plate_id}" in position_history:
                    px1, py1, px2, py2 = smooth_position(f"plate_{best_plate_id}", (px1, py1, px2, py2), alpha=0.6)
                else:
                    position_history[f"plate_{best_plate_id}"] = (px1, py1, px2, py2)
                
                best_plate_box = (px1, py1, px2, py2)
                
                # Draw plate bounding box
                cv2.rectangle(display_frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
                cv2.putText(display_frame, f"Plate #{best_plate_id}", (px1, py1-5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # === Improved plate crop with margin ===
                margin = 5
                plate_x1 = max(px1 - margin, 0)
                plate_y1 = max(py1 - margin, 0)
                plate_x2 = min(px2 + margin, frame.shape[1])
                plate_y2 = min(py2 + margin, frame.shape[0])
                
                # Draw plate region outline
                cv2.rectangle(display_frame, (plate_x1, plate_y1), (plate_x2, plate_y2), (0, 255, 255), 1)
                
                # Crop plate image
                cropped = ocr_frame[plate_y1:plate_y2, plate_x1:plate_x2]
                
                if cropped.size > 0 and cropped.shape[0] > 0 and cropped.shape[1] > 0:
                    preprocessed_images = preprocess_plate(cropped)
                    
                    if preprocessed_images:
                        all_text_candidates = []
                        
                        for img in preprocessed_images:
                            if img is None:
                                continue
                                
                            plate_texts = reader.readtext(img, detail=1, paragraph=False, min_size=5)
                            
                            for result in plate_texts:
                                text, confidence = result[1], result[2]
                                text = text.strip().replace(" ", "").upper()
                                
                                if is_valid_plate(text) and confidence > 0.3:
                                    all_text_candidates.append((text, confidence))
                        
                        best_text, confidence = process_text_candidates(all_text_candidates)
                        
                        if best_text:
                            update_ocr_history(track_id, best_text, confidence)
                
                # Get best plate text with temporal consistency
                final_plate_text, final_confidence = get_best_plate_text(track_id)
                
                if final_plate_text:
                    plate_type = determine_plate_type(final_plate_text)
                    
                    # Create display text
                    plate_type_score = 1.0
                    plate_type_display = f"{final_plate_text} {plate_type}:{plate_type_score:.1f}"
                    
                    # Determine colors
                    text_color = (0, 0, 0)
                    if final_confidence > 0.7:
                        bg_color = (0, 255, 0)
                    elif final_confidence > 0.5:
                        bg_color = (255, 255, 0)
                    else:
                        bg_color = (255, 255, 255)
                    
                    # Draw info box
                    label_box_height = 20
                    label_box_width = x2 - x1
                    cv2.rectangle(display_frame, (x1, y2 - label_box_height), 
                                 (x1 + label_box_width, y2), bg_color, -1)
                    
                    # Put plate info
                    cv2.putText(display_frame, plate_type_display, 
                               (x1 + 2, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                    
                    # Add confidence indicator
                    confidence_display = f"OCR:{final_confidence:.2f}"
                    cv2.putText(display_frame, confidence_display,
                               (x1 + label_box_width - 70, y2 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                    
                    # Create and display thumbnail
                    enlarged = cv2.resize(cropped, (180, 60)) if cropped.size > 0 else np.ones((60, 180, 3), dtype=np.uint8) * 255
                    
                    thumb_w = 120
                    thumb_h = 40
                    thumb_x = x2 - thumb_w - 5
                    thumb_y = y1 + 25
                    
                    thumb_x = max(x1, thumb_x)
                    thumb_y = max(y1, thumb_y)
                    
                    available_width = x2 - thumb_x - 5
                    if available_width < thumb_w:
                        thumb_w = available_width
                        
                    available_height = y2 - thumb_y - 5
                    if available_height < thumb_h:
                        thumb_h = available_height
                    
                    if thumb_w > 10 and thumb_h > 10 and cropped.size > 0:
                        try:
                            thumb_img = cv2.resize(cropped, (thumb_w, thumb_h))
                            cv2.rectangle(display_frame, (thumb_x, thumb_y), 
                                        (thumb_x + thumb_w, thumb_y + thumb_h), (0, 255, 255), 1)
                            
                            if (thumb_x + thumb_w <= display_frame.shape[1] and 
                                thumb_y + thumb_h <= display_frame.shape[0]):
                                display_frame[thumb_y:thumb_y + thumb_h, thumb_x:thumb_x + thumb_w] = thumb_img
                        except Exception:
                            pass

    # Process unassigned plates for debugging
    for plate_track in plate_tracks:
        px1, py1, px2, py2, plate_id = map(int, plate_track)
        plate_id = int(plate_id)
        
        if plate_id not in plate_vehicle_map:
            cv2.rectangle(display_frame, (px1, py1), (px2, py2), (255, 0, 255), 1)

    # Clean up old tracks
    to_remove = [tid for tid in previous_positions if tid not in current_vehicle_ids]
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
    
    if frame_number % 100 == 0:
        active_plate_ids = set()
        for track in plate_tracks:
            active_plate_ids.add(int(track[4]))
            
        plate_keys = [k for k in position_history.keys() if isinstance(k, str) and k.startswith("plate_")]
        for key in plate_keys:
            plate_id = int(key.split("_")[1])
            if plate_id not in active_plate_ids:
                del position_history[key]

    # Draw counting line and info
    cv2.line(display_frame, (0, LINE_POSITION), (frame_width, LINE_POSITION), (255, 255, 0), 2)
    cv2.putText(display_frame, f"Count: {vehicle_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    cv2.putText(display_frame, f"Count: {vehicle_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
    cv2.putText(display_frame, f"FPS: {fps:.1f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
    cv2.putText(display_frame, f"FPS: {fps:.1f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Display frame
    cv2.imshow("Vehicle & Plate Detection", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nReal-time video processing finished.")