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
VIDEO_INPUT_PATH = os.path.join("assets", "testing_video6.mp4")
VEHICLE_CONFIDENCE_THRESHOLD = 0.5
PLATE_CONFIDENCE_THRESHOLD = 0.4  # Lower threshold for plates as they're smaller

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

# Initialize EasyOCR with allowlist to improve accuracy
reader = easyocr.Reader(['en'], gpu=True, )
                      

# === IMPROVED SORT TRACKER INITIALIZATION ===
# Tune these parameters for better tracking performance
# max_age: Frames to keep track even if not detected
# min_hits: Minimum detections before track confirmation
# iou_threshold: Lower value makes it more strict on matching
vehicle_tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.25)
plate_tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.4)  # Separate tracker for plates

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
detected_plates = {}     # Store detected plates and their positions by frame
plate_to_vehicle_map = {}  # Map plate track IDs to vehicle track IDs

# OCR confidence history for each plate
plate_ocr_history = {}   # {vehicle_id: {plate_text: [confidences]}}

# How many frames of class history to keep
CLASS_HISTORY_MAX = 10

# Confidence factor for class switching
CLASS_SWITCH_THRESHOLD = 3

# License plate pattern regex (customize for your region if needed)
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
    
    # Create multiple versions with different processing
    results = []
    
    # Version 1: Standard grayscale with CLAHE
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    _, thresh1 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results.append(thresh1)
    
    # Version 2: Increased contrast with different threshold
    adjusted = cv2.convertScaleAbs(gray, alpha=1.3, beta=0)
    _, thresh2 = cv2.threshold(adjusted, 110, 255, cv2.THRESH_BINARY)
    results.append(thresh2)
    
    # Version 3: Edge enhancement
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 100, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    results.append(dilated)
    
    # Version 4: Original color image (some OCR engines do better with color)
    resized = cv2.resize(cropped, (cropped.shape[1]*2, cropped.shape[0]*2))
    results.append(resized)
    
    return results

def is_valid_plate(text):
    """Check if detected text is likely a valid license plate"""
    if not text:
        return False
    
    # Remove spaces
    cleaned = text.replace(" ", "")
    
    # Check length (typical license plates are 4-10 characters)
    if len(cleaned) < 4 or len(cleaned) > 10:
        return False
    
    # Check for vehicle type words
    lower_text = cleaned.lower()
    if any(vehicle_type in lower_text for vehicle_type in ["car", "truck", "bus", "van"]):
        return False
    
    # Make sure it has at least one number and one letter
    has_letter = any(c.isalpha() for c in cleaned)
    has_number = any(c.isdigit() for c in cleaned)
    
    # Ensure it's mostly alphanumeric with few special chars
    special_chars = sum(1 for c in cleaned if not c.isalnum())
    if special_chars > 2:  # Allow max 2 special chars like '-' or '.'
        return False
    
    # Most license plates have both letters and numbers
    return has_letter or has_number

def get_dominant_class(track_id, new_class=None, confidence=0):
    """Maintain class history and return dominant class using voting"""
    if track_id not in class_history:
        class_history[track_id] = []
    
    # Add new class with confidence if provided
    if new_class:
        # Add confidence as weight
        class_history[track_id].append((new_class, confidence))
        # Keep history limited to prevent memory issues
        if len(class_history[track_id]) > CLASS_HISTORY_MAX:
            class_history[track_id].pop(0)
    
    # If no history, return the new class
    if not class_history[track_id]:
        return new_class
    
    # Count votes with confidence weighting
    votes = {}
    for cls, conf in class_history[track_id]:
        if cls not in votes:
            votes[cls] = 0
        votes[cls] += conf  # Add confidence as weight
    
    # Return class with highest weighted votes
    if votes:
        return max(votes.items(), key=lambda x: x[1])[0]
    return "vehicle"  # Default fallback

def smooth_position(track_id, box, alpha=0.7):
    """Smooth bounding box position using exponential moving average"""
    if track_id not in position_history:
        position_history[track_id] = box
        return box
    
    # Apply EMA smoothing to reduce jitter
    x1, y1, x2, y2 = box
    px1, py1, px2, py2 = position_history[track_id]
    
    # Calculate smoothed box
    sx1 = int(alpha * px1 + (1 - alpha) * x1)
    sy1 = int(alpha * py1 + (1 - alpha) * y1)
    sx2 = int(alpha * px2 + (1 - alpha) * x2)
    sy2 = int(alpha * py2 + (1 - alpha) * y2)
    
    # Store smoothed position for next frame
    position_history[track_id] = (sx1, sy1, sx2, sy2)
    
    return (sx1, sy1, sx2, sy2)

def assess_track_quality(track_id, detection_count, is_detected):
    """Assess track quality based on detection consistency"""
    if track_id not in track_quality:
        track_quality[track_id] = {'count': 0, 'detections': 0}
    
    track_quality[track_id]['count'] += 1
    if is_detected:
        track_quality[track_id]['detections'] += 1
    
    # Calculate quality as percentage of frames with detections
    quality = track_quality[track_id]['detections'] / track_quality[track_id]['count']
    
    return quality

def calculate_iou(box1, box2):
    """Calculate IOU between two boxes [x1,y1,x2,y2]"""
    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Check if boxes overlap
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    # Return IOU
    return intersection / union if union > 0 else 0.0

def assign_plates_to_vehicles(vehicle_tracks, plate_tracks):
    """Match plate tracks to vehicle tracks based on spatial relationship and tracking"""
    vehicle_plate_assignments = {}
    plate_vehicle_assignments = {}
    
    for plate_track in plate_tracks:
        plate_id = int(plate_track[4])
        plate_box = plate_track[:4].astype(int)
        best_iou = 0
        best_vehicle_id = None
        
        # Check against all vehicle tracks
        for vehicle_track in vehicle_tracks:
            vehicle_id = int(vehicle_track[4])
            vehicle_box = vehicle_track[:4].astype(int)
            
            # Calculate plate's position relative to vehicle
            px_center = (plate_box[0] + plate_box[2]) / 2
            py_center = (plate_box[1] + plate_box[3]) / 2
            
            # Vehicle dimensions
            vehicle_width = vehicle_box[2] - vehicle_box[0]
            vehicle_height = vehicle_box[3] - vehicle_box[1]
            
            # Add some margin to vehicle box (especially at the front/back where plates are)
            expanded_box = [
                vehicle_box[0] - int(vehicle_width * 0.05),
                vehicle_box[1] - int(vehicle_height * 0.05),
                vehicle_box[2] + int(vehicle_width * 0.05),
                vehicle_box[3] + int(vehicle_height * 0.05)
            ]
            
            # Check if center of plate is in expanded vehicle box
            if (expanded_box[0] <= px_center <= expanded_box[2] and 
                expanded_box[1] <= py_center <= expanded_box[3]):
                
                # Calculate overlap
                iou = calculate_iou(expanded_box, plate_box)
                
                # Check plate size relative to vehicle (plates shouldn't be too large)
                plate_width = plate_box[2] - plate_box[0]
                plate_height = plate_box[3] - plate_box[1]
                
                if (plate_width < vehicle_width * 0.8 and 
                    plate_height < vehicle_height * 0.8):
                    
                    # Prioritize plates at bottom half (front/rear of vehicles)
                    if py_center > (vehicle_box[1] + vehicle_box[3]) / 2:
                        iou *= 1.5  # Boost score for bottom plates
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_vehicle_id = vehicle_id
        
        # Consider temporal consistency - prefer to keep previous assignments
        if plate_id in plate_to_vehicle_map and best_vehicle_id is not None:
            prev_vehicle_id = plate_to_vehicle_map[plate_id]
            
            # If the previous vehicle is still being tracked and has reasonable IoU,
            # stick with it unless the new match is significantly better
            if prev_vehicle_id in [int(vt[4]) for vt in vehicle_tracks]:
                for vehicle_track in vehicle_tracks:
                    if int(vehicle_track[4]) == prev_vehicle_id:
                        prev_vehicle_box = vehicle_track[:4].astype(int)
                        prev_iou = calculate_iou(prev_vehicle_box, plate_box)
                        
                        # If previous assignment still valid, keep it unless new one is much better
                        if prev_iou > 0.1 and best_iou < prev_iou * 1.5:
                            best_vehicle_id = prev_vehicle_id
                            best_iou = prev_iou
        
        # Only assign if IoU is reasonable
        if best_vehicle_id is not None and best_iou > 0.1:
            if best_vehicle_id not in vehicle_plate_assignments:
                vehicle_plate_assignments[best_vehicle_id] = []
            
            vehicle_plate_assignments[best_vehicle_id].append((plate_box, plate_id))
            plate_to_vehicle_map[plate_id] = best_vehicle_id
            plate_vehicle_assignments[plate_id] = best_vehicle_id
    
    return vehicle_plate_assignments, plate_vehicle_assignments

def process_text_candidates(text_candidates):
    """Process multiple OCR results to find the most likely plate text"""
    if not text_candidates:
        return None, 0
    
    # Count frequency of each text
    text_counts = {}
    text_confidences = {}
    
    for text, conf in text_candidates:
        cleaned_text = text.strip().replace(" ", "").upper()
        
        # Skip invalid texts
        if not is_valid_plate(cleaned_text):
            continue
        
        if cleaned_text not in text_counts:
            text_counts[cleaned_text] = 0
            text_confidences[cleaned_text] = []
        
        text_counts[cleaned_text] += 1
        text_confidences[cleaned_text].append(conf)
    
    # Find the most frequent text with good confidence
    best_text = None
    best_score = 0
    
    for text, count in text_counts.items():
        avg_conf = sum(text_confidences[text]) / len(text_confidences[text])
        score = count * avg_conf  # Combine frequency and confidence
        
        if score > best_score:
            best_score = score
            best_text = text
    
    if best_text:
        avg_confidence = sum(text_confidences[best_text]) / len(text_confidences[best_text])
        return best_text, avg_confidence
    
    return None, 0

def update_ocr_history(vehicle_id, plate_text, confidence):
    """Update OCR history with temporal consistency"""
    if not plate_text:
        return
    
    if vehicle_id not in plate_ocr_history:
        plate_ocr_history[vehicle_id] = {}
    
    if plate_text not in plate_ocr_history[vehicle_id]:
        plate_ocr_history[vehicle_id][plate_text] = []
    
    plate_ocr_history[vehicle_id][plate_text].append(confidence)
    
    # Limit history length
    if len(plate_ocr_history[vehicle_id][plate_text]) > 10:
        plate_ocr_history[vehicle_id][plate_text] = plate_ocr_history[vehicle_id][plate_text][-10:]

def get_best_plate_text(vehicle_id):
    """Get the most confident plate text across multiple frames"""
    if vehicle_id not in plate_ocr_history:
        return None, 0
    
    best_text = None
    best_score = 0
    
    for text, confidences in plate_ocr_history[vehicle_id].items():
        # Score is based on frequency and average confidence
        if len(confidences) >= 2:  # Require at least 2 detections
            avg_conf = sum(confidences) / len(confidences)
            # Weight by number of detections and confidence
            score = len(confidences) * avg_conf
            
            if score > best_score:
                best_score = score
                best_text = text
    
    if best_text and best_score > 1.0:  # Threshold for reporting
        # Calculate average confidence of the best text
        avg_confidence = sum(plate_ocr_history[vehicle_id][best_text]) / len(plate_ocr_history[vehicle_id][best_text])
        return best_text, avg_confidence
    
    return None, 0

# Create an initially empty display frame
display_frame = None

# Setup visualization display
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

    # Create a clean copy of the frame for OCR processing
    ocr_frame = frame.copy()
    
    # Use the original frame for display
    display_frame = frame.copy()

    # === DETECT VEHICLES ===
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

    # === DETECT LICENSE PLATES ===
    plate_results = plate_model.predict(source=frame, conf=PLATE_CONFIDENCE_THRESHOLD, save=False, verbose=False)
    plate_detections = []
    
    for r in plate_results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Make sure the plate detection box has reasonable dimensions
            if (x2 - x1) > 5 and (y2 - y1) > 5:
                # Add detected plates with confidence
                plate_detections.append([x1, y1, x2, y2, conf])
    
    # Track vehicles using SORT
    vehicle_dets_np = np.array(vehicle_detections) if vehicle_detections else np.empty((0, 5))
    vehicle_tracks = vehicle_tracker.update(vehicle_dets_np)
    
    # Track plates separately using SORT
    plate_dets_np = np.array(plate_detections) if plate_detections else np.empty((0, 5))
    plate_tracks = plate_tracker.update(plate_dets_np)
    
    # Assign plates to vehicles based on spatial relationship
    vehicle_plate_assignments, plate_vehicle_map = assign_plates_to_vehicles(vehicle_tracks, plate_tracks)
    
    # Track active IDs and mark whether they were detected in this frame
    current_vehicle_ids = set()
    detected_vehicle_ids = set()
    
    # Match detections to tracks based on IOU to preserve class information
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
                
                # Calculate IOU
                iou = calculate_iou(det_box, track_box)
                
                if iou > best_iou and iou > 0.5:  # 0.5 IOU threshold for matching
                    best_iou = iou
                    best_track_idx = track_idx
            
            if best_track_idx >= 0:
                track_id = int(vehicle_tracks[best_track_idx][4])
                detection_track_map[track_id] = (det_idx, best_iou)
                detected_vehicle_ids.add(track_id)

    # Process each vehicle track
    for track_idx, track in enumerate(vehicle_tracks):
        x1, y1, x2, y2, track_id = map(int, track)
        track_id = int(track_id)  # Ensure track_id is an integer
        current_vehicle_ids.add(track_id)
        
        # Get detection info if this track was detected in this frame
        was_detected = track_id in detected_vehicle_ids
        if was_detected:
            det_idx, iou = detection_track_map[track_id]
            detected_class = detection_classes[det_idx]
            detected_conf = detection_confidences[det_idx]
        else:
            detected_class = None
            detected_conf = 0
        
        # Assess track quality
        quality = assess_track_quality(track_id, frame_number, was_detected)
        
        # Smooth box position to reduce jitter
        x1, y1, x2, y2 = smooth_position(track_id, (x1, y1, x2, y2))
        
        # Update class with voting system
        dominant_class = get_dominant_class(track_id, detected_class, detected_conf)
        
        # Decide final class label
        if track_id not in track_labels:
            # New track, assign dominant class
            track_labels[track_id] = dominant_class
        elif was_detected and detected_class != track_labels[track_id]:
            # Consider class switch only if confidence is high enough
            if detected_conf > VEHICLE_CONFIDENCE_THRESHOLD + 0.1:
                # Check if dominant class is different with high confidence
                if dominant_class != track_labels[track_id]:
                    # Only switch if dominant class is consistently different
                    class_counts = Counter([c for c, _ in class_history[track_id][-CLASS_SWITCH_THRESHOLD:]])
                    if class_counts.get(dominant_class, 0) >= CLASS_SWITCH_THRESHOLD:
                        track_labels[track_id] = dominant_class
        
        # Get center position for crossing line detection
        center_y = int((y1 + y2) / 2)
        previous_y = previous_positions.get(track_id, None)
        previous_positions[track_id] = center_y

        # Count vehicle if it crosses the line
        if previous_y is not None and center_y > previous_y:
            if (previous_y < LINE_POSITION and center_y >= LINE_POSITION) or \
               (LINE_POSITION - LINE_OFFSET < center_y < LINE_POSITION + LINE_OFFSET):
                if track_id not in counted_ids and quality > 0.5:  # Only count high quality tracks
                    vehicle_count += 1
                    counted_ids.add(track_id)

        # Get the current label for this track
        label = track_labels[track_id]
        box_color = label_colors.get(label, (255, 255, 255))

        # Draw vehicle bounding box on display frame
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)

        # Draw ID text inside box top-left
        cv2.rectangle(display_frame, (x1, y1), (x1 + 120, y1 + 20), box_color, -1)
        cv2.putText(display_frame, f"{label} {track_id} Q:{quality:.2f}", (x1 + 2, y1 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # === PROCESS LICENSE PLATES FOR THIS VEHICLE ===
        if track_id in vehicle_plate_assignments:
            best_plate_box = None
            best_plate_confidence = 0
            best_plate_id = None
            
            # Get all plates assigned to this vehicle
            vehicle_plates = vehicle_plate_assignments[track_id]
            
            if vehicle_plates:
                # Choose the largest plate (often the most readable)
                largest_area = 0
                for plate_box, plate_id in vehicle_plates:
                    px1, py1, px2, py2 = plate_box
                    area = (px2 - px1) * (py2 - py1)
                    if area > largest_area:
                        largest_area = area
                        best_plate_box = plate_box
                        best_plate_id = plate_id
                
                px1, py1, px2, py2 = best_plate_box
                
                # Apply temporal smoothing to plate box if we've seen it before
                if f"plate_{best_plate_id}" in position_history:
                    px1, py1, px2, py2 = smooth_position(f"plate_{best_plate_id}", (px1, py1, px2, py2), alpha=0.6)
                else:
                    position_history[f"plate_{best_plate_id}"] = (px1, py1, px2, py2)
                
                # Update the best plate box with smoothed coordinates
                best_plate_box = (px1, py1, px2, py2)
                
                # Draw the plate box with smooth tracking
                cv2.rectangle(display_frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
                cv2.putText(display_frame, f"Plate #{best_plate_id}", (px1, py1-5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Extract the plate image for OCR - add a small margin for better results
                margin_x = int((px2 - px1) * 0.05)
                margin_y = int((py2 - py1) * 0.05)
                
                # Ensure margins don't go outside the frame
                crop_x1 = max(0, px1 - margin_x)
                crop_y1 = max(0, py1 - margin_y)
                crop_x2 = min(frame.shape[1], px2 + margin_x)
                crop_y2 = min(frame.shape[0], py2 + margin_y)
                
                cropped = ocr_frame[crop_y1:crop_y2, crop_x1:crop_x2]
                
                if cropped.size > 0 and cropped.shape[0] > 0 and cropped.shape[1] > 0:
                    # Process with multiple methods for more robust OCR
                    preprocessed_images = preprocess_plate(cropped)
                    
                    if preprocessed_images:
                        all_text_candidates = []
                        
                        # Try OCR on each preprocessed image
                        for img in preprocessed_images:
                            if img is None:
                                continue
                                
                            plate_texts = reader.readtext(img, detail=1, paragraph=False, min_size=5)
                            
                            for result in plate_texts:
                                text, confidence = result[1], result[2]
                                text = text.strip().replace(" ", "").upper()
                                
                                if is_valid_plate(text) and confidence > 0.3:
                                    all_text_candidates.append((text, confidence))
                        
                        # Process candidates to find the most likely plate text
                        best_text, confidence = process_text_candidates(all_text_candidates)
                        
                        if best_text:
                            # Update OCR history for temporal consistency
                            update_ocr_history(track_id, best_text, confidence)
                
                # Get the best plate text with temporal consistency
                final_plate_text, final_confidence = get_best_plate_text(track_id)
                
                if final_plate_text:
                    # Get the last character for odd/even determination
                    last_char = final_plate_text.strip()[-1]
                    plate_type = "Odd" if last_char.isdigit() and int(last_char) % 2 != 0 else "Even"
                    
                    # Determine plate color based on confidence
                    text_color = (0, 0, 0)
                    bg_color = (255, 255, 255)
                    if final_confidence > 0.7:
                        bg_color = (0, 255, 0)  # Green background for high confidence
                    elif final_confidence > 0.5:
                        bg_color = (255, 255, 0)  # Yellow for medium confidence
                    
                    # Draw plate text at bottom inside vehicle box
                    label_box_height = 20
                    label_box_width = x2 - x1
                    cv2.rectangle(display_frame, (x1, y2 - label_box_height), (x1 + label_box_width, y2), bg_color, -1)
                    cv2.putText(display_frame, f"{final_plate_text} ({plate_type}) {final_confidence:.2f}", 
                               (x1 + 2, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                    
                    # Resize cropped plate for display
                    thumb_w = 120
                    thumb_h = 40
                    thumb_x = x2 - thumb_w - 5
                    thumb_y = y1 + 25  # Position below the label
                    
                    # Ensure thumbnail fits within the vehicle box and display frame
                    thumb_x = max(x1, thumb_x)
                    thumb_y = max(y1, thumb_y)
                    
                    # Adjust dimensions to fit available space
                    available_width = x2 - thumb_x - 5
                    if available_width < thumb_w:
                        thumb_w = available_width
                        
                    available_height = y2 - thumb_y - 5
                    if available_height < thumb_h:
                        thumb_h = available_height
                    
                    # Only draw if dimensions are reasonable
                    if thumb_w > 10 and thumb_h > 10 and cropped.size > 0:
                        try:
                            thumb_img = cv2.resize(cropped, (thumb_w, thumb_h))
                            
                            # Draw border around thumbnail
                            cv2.rectangle(display_frame, (thumb_x, thumb_y), 
                                        (thumb_x + thumb_w, thumb_y + thumb_h), (0, 255, 255), 1)
                            
                            # Ensure the thumbnail fits in the display frame
                            if (thumb_x + thumb_w <= display_frame.shape[1] and 
                                thumb_y + thumb_h <= display_frame.shape[0]):
                                display_frame[thumb_y:thumb_y + thumb_h, thumb_x:thumb_x + thumb_w] = thumb_img
                        except Exception:
                            pass  # Skip thumbnail if resizing fails

    # Process each plate track for visualization and debugging
    for plate_track in plate_tracks:
        px1, py1, px2, py2, plate_id = map(int, plate_track)
        plate_id = int(plate_id)
        
        # Only visualize plates that aren't assigned to vehicles
        if plate_id not in plate_vehicle_map:
            # Draw unassigned plates with different color
            cv2.rectangle(display_frame, (px1, py1), (px2, py2), (255, 0, 255), 1)  # Magenta for unassigned
    
    # Remove old vehicle tracks
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
    
    # Clean up plate tracking data for plates not seen recently (after 100 frames)
    if frame_number % 100 == 0:
        # Find active plate IDs
        active_plate_ids = set()
        for track in plate_tracks:
            active_plate_ids.add(int(track[4]))
            
        # Remove position history for inactive plates
        plate_keys = [k for k in position_history.keys() if isinstance(k, str) and k.startswith("plate_")]
        for key in plate_keys:
            plate_id = int(key.split("_")[1])
            if plate_id not in active_plate_ids:
                del position_history[key]

    # Draw counting line
    cv2.line(display_frame, (0, LINE_POSITION), (frame_width, LINE_POSITION), (255, 255, 0), 2)
    
    # Add shadow for better visibility of count text
    cv2.putText(display_frame, f"Count: {vehicle_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    cv2.putText(display_frame, f"Count: {vehicle_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
    
    # Display FPS for performance monitoring
    cv2.putText(display_frame, f"FPS: {fps:.1f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
    cv2.putText(display_frame, f"FPS: {fps:.1f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Display frame
    cv2.imshow("Vehicle & Plate Detection", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nReal-time video processing finished.")