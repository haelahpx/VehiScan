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
CAR_MODEL_PATH = "models/testv3.pt"
PLATE_MODEL_PATH = "models/plat_model/weights/best.pt"
VIDEO_INPUT_PATH = os.path.join("assets", "testing_video6.mp4")
CONFIDENCE_THRESHOLD = 0.5

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
# Tune these parameters for better tracking performance
# max_age: Frames to keep track even if not detected
# min_hits: Minimum detections before track confirmation
# iou_threshold: Lower value makes it more strict on matching
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

# How many frames of class history to keep
CLASS_HISTORY_MAX = 10

# Confidence factor for class switching
CLASS_SWITCH_THRESHOLD = 3

label_colors = {
    'car': (0, 255, 0),
    'truck': (255, 0, 0),
    'bus': (0, 0, 255),
    'van': (255, 255, 0)
}

def preprocess_plate(cropped):
    """Preprocess plate image to improve OCR with minimal slowdown."""
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    
    # increase contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # simple threshold
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # morph close to fill small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return morph

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

# Create an initially empty display frame
display_frame = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1
    print(f"Processing: {(frame_number / total_frames) * 100:.2f}%", end="\r")

    # Create a clean copy of the frame for OCR processing
    ocr_frame = frame.copy()
    
    # Use the original frame for display
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

    # Track active IDs and mark whether they were detected in this frame
    current_ids = set()
    detected_ids = set()
    
    # Match detections to tracks based on IOU to preserve class information
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
                
                # Calculate IOU
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
                
                if iou > best_iou and iou > 0.5:  # 0.5 IOU threshold for matching
                    best_iou = iou
                    best_track_idx = track_idx
            
            if best_track_idx >= 0:
                track_id = int(tracks[best_track_idx][4])
                detection_track_map[track_id] = (det_idx, best_iou)
                detected_ids.add(track_id)

    # Process each track
    for track_idx, track in enumerate(tracks):
        x1, y1, x2, y2, track_id = map(int, track)
        track_id = int(track_id)  # Ensure track_id is an integer
        current_ids.add(track_id)
        
        # Get detection info if this track was detected in this frame
        was_detected = track_id in detected_ids
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
            if detected_conf > CONFIDENCE_THRESHOLD + 0.1:
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

        # === Improved plate crop with margin
        plate_h = (y2 - y1) // 3
        plate_w = (x2 - x1) // 2
        margin = 5
        plate_x1 = max(x1 + (x2 - x1 - plate_w) // 2 - margin, 0)
        plate_y1 = max(y2 - plate_h - margin, 0)
        plate_x2 = min(plate_x1 + plate_w + margin * 2, frame.shape[1])
        plate_y2 = min(plate_y1 + plate_h + margin * 2, frame.shape[0])

        # Draw plate region outline on display frame
        cv2.rectangle(display_frame, (plate_x1, plate_y1), (plate_x2, plate_y2), (0, 255, 255), 1)

        # Crop from the clean OCR frame without vehicle ID overlays
        cropped = ocr_frame[plate_y1:plate_y2, plate_x1:plate_x2]

        if cropped.size > 0 and cropped.shape[0] > 0 and cropped.shape[1] > 0:
            preprocessed = preprocess_plate(cropped)
            plate_texts = reader.readtext(preprocessed, detail=1, paragraph=False, min_size=5)

            if len(plate_texts) == 0:
                # fallback: try original if nothing detected
                plate_texts = reader.readtext(cropped, detail=1, paragraph=False, min_size=5)

            if track_id not in plate_history:
                plate_history[track_id] = []

            for result in plate_texts:
                text, confidence = result[1], result[2]
                text = text.strip().replace(" ", "")
                
                # Filter out vehicle ID text patterns
                if any(pattern in text.lower() for pattern in ["car", "truck", "bus", "van"]):
                    continue
                    
                # Filter out short IDs that might be just the track_id number
                if text.isdigit() and len(text) < 3:
                    continue
                    
                if len(text) >= 4 and confidence >= 0.4:
                    # Check if text contains only valid license plate characters
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
            # Validate the plate format to ensure it's not just "car 21" etc.
            is_valid_plate = True
            lower_plate = best_plate.lower()
            
            # Check for vehicle type words in the plate
            for vehicle_type in ["car", "truck", "bus", "van"]:
                if vehicle_type in lower_plate:
                    is_valid_plate = False
                    break
            
            if is_valid_plate:
                # Get the last character for odd/even determination
                last_char = best_plate.strip()[-1]
                plate_type = "Odd" if last_char.isdigit() and int(last_char) % 2 != 0 else "Even"
                
                # Draw plate text at bottom inside vehicle box
                label_box_height = 20
                label_box_width = x2 - x1
                cv2.rectangle(display_frame, (x1, y2 - label_box_height), (x1 + label_box_width, y2), (255, 255, 255), -1)
                cv2.putText(display_frame, f"{best_plate} ({plate_type})", (x1 + 2, y2 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                thumb_w = 120
                thumb_h = 40
                thumb_x = x2 - thumb_w - 5
                thumb_y = y1 + 5

                # Ensure the thumbnail fits within the vehicle box and display frame
                thumb_x = max(x1, thumb_x)  # Don't go left of vehicle box
                thumb_y = max(y1, thumb_y)   # Don't go above vehicle box
                
                # Adjust width if needed to fit in available space
                available_width = x2 - thumb_x - 5
                if available_width < thumb_w:
                    thumb_w = available_width
                
                # Adjust height if needed to fit in available space
                available_height = y2 - thumb_y - 5
                if available_height < thumb_h:
                    thumb_h = available_height
                
                # Only draw if we have reasonable dimensions
                if thumb_w > 10 and thumb_h > 10:
                    thumb_img = cv2.resize(enlarged, (thumb_w, thumb_h))
                    cv2.rectangle(display_frame, (thumb_x, thumb_y), (thumb_x + thumb_w, thumb_y + thumb_h), (0, 255, 255), 1)
                    
                    # Ensure the thumbnail fits in the display frame
                    if (thumb_x + thumb_w <= display_frame.shape[1] and 
                        thumb_y + thumb_h <= display_frame.shape[0]):
                        display_frame[thumb_y:thumb_y + thumb_h, thumb_x:thumb_x + thumb_w] = thumb_img

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