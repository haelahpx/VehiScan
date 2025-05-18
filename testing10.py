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
VIDEO_INPUT_PATH = os.path.join("assets", "video_testing4.mp4")
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
fps = cap.get(cv2.CAP_PROP_FPS)

# === ADAPTIVE COUNTING LINE SYSTEM ===
# Initialize with default values
LINE_POSITION = int(frame_height * 0.6)  # Default position at 60% of frame height
LINE_DIRECTION = "bottom"  # Direction of movement to count ("bottom", "top", "left", "right")
LINE_OFFSET = int(frame_height * 0.01)
MANUAL_LINE_SET = False  # Flag to indicate if user has manually set a line

vehicle_count = 0
counted_ids = set()
plate_history = {}

# === IMPROVED TRACKING DATA STRUCTURES ===
previous_positions = {}  # Track position history
track_labels = {}        # Final assigned labels
class_history = {}       # Class voting history
position_history = {}    # Position smoothing
track_quality = {}       # Track quality assessment
vehicle_directions = {}  # Track movement directions
counting_zones = []      # Multiple counting zones

# How many frames of class history to keep
CLASS_HISTORY_MAX = 10

# Confidence factor for class switching
CLASS_SWITCH_THRESHOLD = 3

# History window for direction determination
DIRECTION_HISTORY_SIZE = 10

label_colors = {
    'car': (0, 255, 0),
    'truck': (255, 0, 0),
    'bus': (0, 0, 255),
    'van': (255, 255, 0)
}

# === AUTO COUNTING LINE DETERMINATION ===
def determine_counting_line(frame_height, frame_width):
    """Determine the optimal counting line position based on frame dimensions"""
    # For now using a simple approach, can be enhanced with scene analysis
    horizontal_line_pos = int(frame_height * 0.6)  # Default position at 60% of frame height
    vertical_line_pos = int(frame_width * 0.5)     # Default position at 50% of frame width
    
    # Create two counting zones - one horizontal and one vertical
    zones = [
        {
            'line_start': (0, horizontal_line_pos),
            'line_end': (frame_width, horizontal_line_pos),
            'direction': 'bottom',  # Count vehicles moving downward
            'name': 'Horizontal Line',
            'count': 0,
            'color': (255, 255, 0),
            'counted_ids': set()
        },
        {
            'line_start': (vertical_line_pos, 0),
            'line_end': (vertical_line_pos, frame_height),
            'direction': 'right',   # Count vehicles moving right
            'name': 'Vertical Line',
            'count': 0,
            'color': (0, 255, 255),
            'counted_ids': set()
        }
    ]
    
    return zones

def preprocess_plate(cropped):
    """Preprocess plate image to improve OCR with minimal slowdown."""
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    
    # increase contrast
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
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

def determine_movement_direction(track_id, center_x, center_y):
    """Determine vehicle movement direction based on position history"""
    if track_id not in vehicle_directions:
        vehicle_directions[track_id] = {
            'positions': deque(maxlen=DIRECTION_HISTORY_SIZE),
            'direction': None
        }
    
    # Add current position to history
    vehicle_directions[track_id]['positions'].append((center_x, center_y))
    
    # Need at least 3 points to determine direction reliably
    if len(vehicle_directions[track_id]['positions']) < 3:
        return None
    
    # Use the first and last positions to determine overall direction
    first_x, first_y = vehicle_directions[track_id]['positions'][0]
    last_x, last_y = vehicle_directions[track_id]['positions'][-1]
    
    # Calculate displacement
    dx = last_x - first_x
    dy = last_y - first_y
    
    # Determine predominant direction
    if abs(dx) > abs(dy):  # Horizontal movement is stronger
        direction = "right" if dx > 0 else "left"
    else:  # Vertical movement is stronger
        direction = "bottom" if dy > 0 else "top"
    
    # Update the detected direction
    vehicle_directions[track_id]['direction'] = direction
    
    return direction

def check_line_crossing(track_id, center_x, center_y, zone, quality):
    """Check if a vehicle has crossed a counting line in a specific zone"""
    # Get previous position
    if track_id not in previous_positions:
        previous_positions[track_id] = (center_x, center_y)
        return False
    
    prev_x, prev_y = previous_positions[track_id]
    current_pos = (center_x, center_y)
    previous_positions[track_id] = current_pos
    
    # Line parameters
    x1, y1 = zone['line_start']
    x2, y2 = zone['line_end']
    
    # Skip if quality is too low
    if quality < 0.5:
        return False
    
    # Already counted this ID for this zone
    if track_id in zone['counted_ids']:
        return False
    
    # Check if line is horizontal or vertical
    if y1 == y2:  # Horizontal line
        # Check if vehicle crossed the line
        if (prev_y < y1 and center_y >= y1) or (prev_y > y1 and center_y <= y1):
            # Check direction
            crossed_direction = "bottom" if center_y > prev_y else "top"
            
            # If no specific direction required or correct direction
            if zone['direction'] == 'any' or zone['direction'] == crossed_direction:
                zone['counted_ids'].add(track_id)
                zone['count'] += 1
                return True
    
    elif x1 == x2:  # Vertical line
        # Check if vehicle crossed the line
        if (prev_x < x1 and center_x >= x1) or (prev_x > x1 and center_x <= x1):
            # Check direction
            crossed_direction = "right" if center_x > prev_x else "left"
            
            # If no specific direction required or correct direction
            if zone['direction'] == 'any' or zone['direction'] == crossed_direction:
                zone['counted_ids'].add(track_id)
                zone['count'] += 1
                return True
    
    return False

# Initialize counting zones
counting_zones = determine_counting_line(frame_height, frame_width)

# === PROCESSING VARIABLES ===
frame_skip = max(1, int(fps / 10))  # Process every Nth frame for optimization
auto_calibration_frames = min(100, int(total_frames * 0.1))  # Use first 10% frames for auto-calibration
calibration_data = []

# Create an initially empty display frame
display_frame = None

# === MAIN PROCESSING LOOP ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1
    print(f"Processing: {(frame_number / total_frames) * 100:.2f}%", end="\r")

    # Skip frames for performance if video is high frame rate
    if frame_number % frame_skip != 0 and frame_number > 10:
        continue
    
    # Auto-calibration: collect data about vehicle movements in early frames
    if frame_number <= auto_calibration_frames:
        # We'll use this phase to gather data about common vehicle movements
        # to automatically adjust counting zones if needed
        pass
    
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
        
        # Calculate center position for direction and crossing detection
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        # Determine movement direction
        direction = determine_movement_direction(track_id, center_x, center_y)
        
        # Check for line crossing in all counting zones
        for zone in counting_zones:
            check_line_crossing(track_id, center_x, center_y, zone, quality)
            
        # Get the current label for this track
        label = track_labels.get(track_id, "vehicle")
        box_color = label_colors.get(label, (255, 255, 255))

        # Draw vehicle bounding box on display frame
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)

        # Draw ID and direction text inside box top-left
        dir_text = f" {direction}" if direction else ""
        cv2.rectangle(display_frame, (x1, y1), (x1 + 140, y1 + 20), box_color, -1)
        cv2.putText(display_frame, f"{label} {track_id}{dir_text} Q:{quality:.2f}", (x1 + 2, y1 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # ===== LICENSE PLATE DETECTION =====
        # Extract vehicle region with a small margin for plate detection
        margin = 10
        vehicle_x1 = max(0, x1 - margin)
        vehicle_y1 = max(0, y1 - margin)
        vehicle_x2 = min(frame.shape[1], x2 + margin)
        vehicle_y2 = min(frame.shape[0], y2 + margin)
        
        # Crop the vehicle region from the OCR frame
        vehicle_crop = ocr_frame[vehicle_y1:vehicle_y2, vehicle_x1:vehicle_x2]
        
        # Only process if the crop is valid
        best_plate_box = None
        highest_conf = 0
        
        if vehicle_crop.size > 0 and vehicle_crop.shape[0] > 0 and vehicle_crop.shape[1] > 0:
            # Run plate detection model on the vehicle crop
            plate_results = plate_model.predict(source=vehicle_crop, conf=PLATE_CONFIDENCE_THRESHOLD, verbose=False)
            
            # Process plate detections
            for plate_r in plate_results:
                plate_boxes = plate_r.boxes
                for plate_box in plate_boxes:
                    px1, py1, px2, py2 = map(int, plate_box.xyxy[0])
                    plate_conf = float(plate_box.conf[0])
                    
                    # Convert coordinates back to original frame
                    plate_x1 = vehicle_x1 + px1
                    plate_y1 = vehicle_y1 + py1
                    plate_x2 = vehicle_x1 + px2
                    plate_y2 = vehicle_y1 + py2
                    
                    # Keep the highest confidence plate detection
                    if plate_conf > highest_conf:
                        highest_conf = plate_conf
                        best_plate_box = (plate_x1, plate_y1, plate_x2, plate_y2)
        
        # If no plate is detected by the model, fall back to the estimation method
        if best_plate_box is None:
            plate_h = (y2 - y1) // 4
            plate_w = (x2 - x1) // 2
            margin = 2
            plate_x1 = max(x1 + (x2 - x1 - plate_w) // 2 - margin, 0)
            plate_y1 = max(y2 - plate_h - margin, 0)
            plate_x2 = min(plate_x1 + plate_w + margin, frame.shape[1])
            plate_y2 = min(plate_y1 + plate_h + margin, frame.shape[0])
            best_plate_box = (plate_x1, plate_y1, plate_x2, plate_y2)
            
            # Draw estimated plate region with yellow outline (fallback)
            cv2.rectangle(display_frame, (plate_x1, plate_y1), (plate_x2, plate_y2), (0, 255, 255), 1)
        else:
            # Draw detected plate region with green outline
            cv2.rectangle(display_frame, best_plate_box[:2], best_plate_box[2:], (0, 255, 0), 2)
        
        # Extract coordinates for OCR processing
        plate_x1, plate_y1, plate_x2, plate_y2 = best_plate_box
        
        # Crop from the clean OCR frame
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
        if tid in vehicle_directions:
            del vehicle_directions[tid]

    # Draw all counting zones
    vehicle_count = sum(zone['count'] for zone in counting_zones)
    
    for i, zone in enumerate(counting_zones):
        # Draw line
        cv2.line(display_frame, zone['line_start'], zone['line_end'], zone['color'], 2)
        
        # Draw direction arrow
        start_x, start_y = zone['line_start']
        end_x, end_y = zone['line_end']
        mid_x, mid_y = (start_x + end_x) // 2, (start_y + end_y) // 2
        
        # Calculate arrow direction
        arrow_len = 20
        if zone['direction'] == 'bottom':
            arrow_x, arrow_y = mid_x, mid_y + arrow_len
        elif zone['direction'] == 'top':
            arrow_x, arrow_y = mid_x, mid_y - arrow_len
        elif zone['direction'] == 'right':
            arrow_x, arrow_y = mid_x + arrow_len, mid_y
        elif zone['direction'] == 'left':
            arrow_x, arrow_y = mid_x - arrow_len, mid_y
        else:
            arrow_x, arrow_y = mid_x, mid_y
        
        # Draw arrow
        cv2.arrowedLine(display_frame, (mid_x, mid_y), (arrow_x, arrow_y), zone['color'], 2)
        
        # Draw zone count
        label_y = 40 + i * 30
        cv2.putText(display_frame, f"{zone['name']}: {zone['count']}", (20, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(display_frame, f"{zone['name']}: {zone['count']}", (20, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, zone['color'], 2)

    # Draw total count
    cv2.putText(display_frame, f"Total Count: {vehicle_count}", (20, frame_height - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
    cv2.putText(display_frame, f"Total Count: {vehicle_count}", (20, frame_height - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Draw frame number and direction info
    cv2.putText(display_frame, f"Frame: {frame_number}/{total_frames}", (frame_width - 200, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(display_frame, f"Frame: {frame_number}/{total_frames}", (frame_width - 200, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Display the result
    cv2.imshow("Vehicle & Plate Detection", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === INTERACTIVE CONTROL FUNCTIONS ===
def on_mouse_click(event, x, y, flags, param):
    """Handle mouse clicks for manual line positioning"""
    global counting_zones, MANUAL_LINE_SET
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Left-click: Set horizontal counting line
        for zone in counting_zones:
            if zone['name'] == 'Horizontal Line':
                zone['line_start'] = (0, y)
                zone['line_end'] = (frame_width, y)
                MANUAL_LINE_SET = True
                break
    
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Right-click: Set vertical counting line
        for zone in counting_zones:
            if zone['name'] == 'Vertical Line':
                zone['line_start'] = (x, 0)
                zone['line_end'] = (x, frame_height)
                MANUAL_LINE_SET = True
                break

def add_keyboard_controls():
    """Add keyboard controls for counting system configuration"""
    global counting_zones, vehicle_count
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('r'):  # Reset counts
        vehicle_count = 0
        for zone in counting_zones:
            zone['count'] = 0
            zone['counted_ids'] = set()
    
    elif key == ord('d'):  # Cycle through direction options for horizontal line
        directions = ['bottom', 'top', 'any']
        for zone in counting_zones:
            if zone['name'] == 'Horizontal Line':
                current_idx = directions.index(zone['direction'])
                zone['direction'] = directions[(current_idx + 1) % len(directions)]
                break
    
    elif key == ord('v'):  # Cycle through direction options for vertical line
        directions = ['right', 'left', 'any']
        for zone in counting_zones:
            if zone['name'] == 'Vertical Line':
                current_idx = directions.index(zone['direction'])
                zone['direction'] = directions[(current_idx + 1) % len(directions)]
                break

# Create analysis report after processing
def create_analysis_report():
    """Create analysis report with vehicle counting statistics"""
    vehicle_types = {}
    plate_types = {'Odd': 0, 'Even': 0}
    direction_counts = {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
    
    # Count vehicles by type
    for track_id, label in track_labels.items():
        if label not in vehicle_types:
            vehicle_types[label] = 0
        vehicle_types[label] += 1
        
    # Count plate types
    for track_id, plates in plate_history.items():
        if plates:
            best_plate = sorted(plates, key=lambda x: x[1], reverse=True)[0][0]
            last_char = best_plate.strip()[-1]
            plate_type = "Odd" if last_char.isdigit() and int(last_char) % 2 != 0 else "Even"
            plate_types[plate_type] += 1
    
    # Count directions
    for track_id, data in vehicle_directions.items():
        direction = data.get('direction')
        if direction:
            direction_counts[direction] += 1
    
    print("\n===== ANALYSIS REPORT =====")
    print(f"Total vehicles counted: {vehicle_count}")
    print("\nVehicle Types:")
    for vehicle_type, count in vehicle_types.items():
        print(f"  {vehicle_type.capitalize()}: {count}")
    
    print("\nPlate Types:")
    for plate_type, count in plate_types.items():
        print(f"  {plate_type}: {count}")
    
    print("\nMovement Directions:")
    for direction, count in direction_counts.items():
        print(f"  {direction.capitalize()}: {count}")
    
    print("\nCounting Zones:")
    for zone in counting_zones:
        print(f"  {zone['name']}: {zone['count']} vehicles")
    
    print("==========================")

# Close video and perform cleanup
cap.release()
cv2.destroyAllWindows()
print("\nReal-time video processing finished.")

# Create analysis report
create_analysis_report()

# === USER INTERFACE AND CONFIGURATION ===
def main():
    """Main function to run the vehicle counting system with configuration options"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Vehicle Detection, Tracking and Counting System')
    parser.add_argument('--video', type=str, default=VIDEO_INPUT_PATH, help='Path to input video file')
    parser.add_argument('--car-model', type=str, default=CAR_MODEL_PATH, help='Path to YOLO car detection model')
    parser.add_argument('--plate-model', type=str, default=PLATE_MODEL_PATH, help='Path to YOLO plate detection model')
    parser.add_argument('--conf-thresh', type=float, default=CONFIDENCE_THRESHOLD, help='Confidence threshold for detections')
    parser.add_argument('--plate-thresh', type=float, default=PLATE_CONFIDENCE_THRESHOLD, help='Confidence threshold for plate detections')
    parser.add_argument('--skip-frames', type=int, default=0, help='Number of frames to skip (0 for auto)')
    parser.add_argument('--horizontal-line', type=float, default=0.6, help='Horizontal line position (0.0-1.0, default 0.6)')
    parser.add_argument('--vertical-line', type=float, default=0.5, help='Vertical line position (0.0-1.0, default 0.5)')
    parser.add_argument('--save-output', type=str, default='', help='Path to save output video')
    parser.add_argument('--show-ui', action='store_true', help='Show UI for configuration')
    
    args = parser.parse_args()
    
    print("Vehicle Detection and Counting System")
    print("-------------------------------------")
    print("Instructions:")
    print("  - Press 'q' to quit")
    print("  - Left-click to set horizontal counting line")
    print("  - Right-click to set vertical counting line")
    print("  - Press 'r' to reset counters")
    print("  - Press 'd' to change horizontal line direction")
    print("  - Press 'v' to change vertical line direction")
    
    # Run the main processing loop
    # (This is already implemented above)
    
    # For a complete application, you'd move the video processing loop here
    # and pass the arguments from the parser

if __name__ == "__main__":
    main()