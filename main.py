import flet as ft
import cv2
import numpy as np
import os
import base64
import ffmpeg
from io import BytesIO
from PIL import Image
import threading
import time
from ultralytics import YOLO
import torch
from sort.sort import Sort
import easyocr
from collections import Counter, deque
import datetime
import logging

# Add these imports at the top of your file
import boto3
from botocore.exceptions import ClientError
import tempfile
import json

# === CONFIG ===
CONFIG = {
    'CAR_MODEL_PATH': 'models/best.pt',
    'PLATE_MODEL_PATH': 'models/car_plat6.pt',
    'OUTPUT_VIDEO_PATH': 'output_video.mp4',
    'CONFIDENCE_THRESHOLD': 0.5,
    'PLATE_CONFIDENCE_THRESHOLD': 0.4,
    'DEFAULT_FRAME_WIDTH': 1280,
    'DEFAULT_FRAME_HEIGHT': 720,
    'VIDEO_URL': 'http://103.95.42.254:84/mjpg/video.mjpg',
    'FRAME_SKIP_FACTOR': 0,  # Process every frame  
    'MAX_FRAME_QUEUE': 2,    # Limit frame queue size to prevent memory buildup
    'TARGET_FPS': 15,        # Target frames per second
}

# Add this to your CONFIG section
AWS_CONFIG = {
    'AWS_ACCESS_KEY_ID': 'xxx',  # Should be set as environment variables
    'AWS_SECRET_ACCESS_KEY': 'xxx',
    'AWS_REGION': 'ap-southeast-3',  # Best for Indonesia (Jakarta)
    'S3_BUCKET_NAME': 'vehiscan',
    'S3_PROCESSED_VIDEOS_FOLDER': 'processed_videos/',
    'S3_PLATE_DATA_FOLDER': 'plate_data/',
    'S3_STATS_FOLDER': 'statistics/'
}   

# Initialize S3 client (add this after your other initializations)
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_CONFIG['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=AWS_CONFIG['AWS_SECRET_ACCESS_KEY'],
    region_name=AWS_CONFIG['AWS_REGION']
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {device}")
    
# Helper functions for S3 operations
def upload_to_s3(file_path, s3_key):
    """Upload a file to S3 bucket"""
    try:
        s3_client.upload_file(file_path, AWS_CONFIG['S3_BUCKET_NAME'], s3_key)
        logger.info(f"Successfully uploaded {file_path} to S3 as {s3_key}")
        return True
    except ClientError as e:
        logger.error(f"Failed to upload to S3: {str(e)}")
        return False

def upload_data_to_s3(data, s3_key):
    """Upload data (dict) as JSON to S3"""
    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
            json.dump(data, tmp_file)
            tmp_file_path = tmp_file.name
        
        return upload_to_s3(tmp_file_path, s3_key)
    except Exception as e:
        logger.error(f"Failed to upload data to S3: {str(e)}")
        return False

def generate_s3_key(folder, filename):
    """Generate S3 key with timestamp"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{folder}{timestamp}_{filename}"

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return YOLO(model_path).to(device)

try:
    car_model = load_model(CONFIG['CAR_MODEL_PATH'])
    plate_model = load_model(CONFIG['PLATE_MODEL_PATH'])
    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.25)
except Exception as e:
    logger.error(f"Failed to initialize models: {str(e)}")
    raise

# Global variables
cap = None
processing_thread = None
stop_processing = False
frame_width, frame_height = CONFIG['DEFAULT_FRAME_WIDTH'], CONFIG['DEFAULT_FRAME_HEIGHT']
label_colors = {
    'car': (0, 255, 0),
    'truck': (255, 0, 0),
    'bus': (0, 0, 255),
    'van': (255, 255, 0)
}

class ProcessingState:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.vehicle_count = 0
        self.counting_zones = []
        self.previous_positions = {}
        self.track_labels = {}
        self.class_history = {}
        self.position_history = {}
        self.track_quality = {}
        self.vehicle_directions = {}
        self.plate_history = {}
        self.global_counted_ids = set()  # Track all counted IDs across all zones
        self.vehicle_type_counts = {
            'car': 0,
            'truck': 0,
            'bus': 0,
            'van': 0
        }

realtime_state = ProcessingState()
upload_state = ProcessingState()

update_lock = threading.Lock()
frame_queue = []
frame_available = threading.Event()

def determine_counting_lines(frame_height, frame_width):
    """Create both horizontal and vertical counting lines"""
    return [
        # Horizontal lines
        {
            'line_start': (0, int(frame_height * 0.6)),
            'line_end': (frame_width, int(frame_height * 0.6)),
            'direction': 'bottom',
            'name': 'Horizontal Line Bottom',
            'color': (255, 255, 0),
            'counted_ids': set()
        },
        {
            'line_start': (0, int(frame_height * 0.4)),
            'line_end': (frame_width, int(frame_height * 0.4)),
            'direction': 'top',
            'name': 'Horizontal Line Top',
            'color': (255, 0, 255),
            'counted_ids': set()
        },
        # Vertical lines
        {
            'line_start': (int(frame_width * 0.3), 0),
            'line_end': (int(frame_width * 0.3), frame_height),
            'direction': 'left',
            'name': 'Vertical Line Left',
            'color': (0, 255, 255),
            'counted_ids': set()
        },
        {
            'line_start': (int(frame_width * 0.7), 0),
            'line_end': (int(frame_width * 0.7), frame_height),
            'direction': 'right',
            'name': 'Vertical Line Right',
            'color': (255, 0, 127),
            'counted_ids': set()
        }
    ]

def clean_plate_text(text):
    """Clean and validate plate text"""
    text = ''.join(c for c in text if c.isalnum() or c in '- ')
    false_positives = ['car', 'truck', 'bus', 'van', 'motor', 'vehicle']
    if any(fp in text.lower() for fp in false_positives):
        return None
    if len(text) < 4:
        return None
    return text.upper()

def preprocess_plate(cropped):
    """Improved plate preprocessing for better OCR results"""
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(morph, -1, kernel)
    return sharpened

def read_plate_with_retry(cropped):
    """Try multiple OCR approaches to read plate"""
    preprocessed = preprocess_plate(cropped)
    results = reader.readtext(preprocessed, detail=1, paragraph=False, min_size=10)
    if not results:
        results = reader.readtext(cropped, detail=1, paragraph=False, 
                                min_size=5, text_threshold=0.4, low_text=0.3)
    valid_results = []
    for result in results:
        text, confidence = result[1], result[2]
        cleaned_text = clean_plate_text(text)
        if cleaned_text and confidence > 0.4:
            valid_results.append((cleaned_text, confidence))
    return valid_results

def get_dominant_class(state, track_id, new_class=None, confidence=0):
    """Maintain class history and return dominant class using voting"""
    if track_id not in state.class_history:
        state.class_history[track_id] = []
    if new_class:
        state.class_history[track_id].append((new_class, confidence))
        if len(state.class_history[track_id]) > 10:
            state.class_history[track_id].pop(0)
    if not state.class_history[track_id]:
        return new_class
    votes = {}
    for cls, conf in state.class_history[track_id]:
        if cls not in votes:
            votes[cls] = 0
        votes[cls] += conf
    if votes:
        return max(votes.items(), key=lambda x: x[1])[0]
    return "vehicle"

def smooth_position(state, track_id, box, alpha=0.7):
    """Smooth bounding box position using exponential moving average"""
    if track_id not in state.position_history:
        state.position_history[track_id] = box
        return box
    x1, y1, x2, y2 = box
    px1, py1, px2, py2 = state.position_history[track_id]
    sx1 = int(alpha * px1 + (1 - alpha) * x1)
    sy1 = int(alpha * py1 + (1 - alpha) * y1)
    sx2 = int(alpha * px2 + (1 - alpha) * x2)
    sy2 = int(alpha * py2 + (1 - alpha) * y2)
    state.position_history[track_id] = (sx1, sy1, sx2, sy2)
    return (sx1, sy1, sx2, sy2)

def assess_track_quality(state, track_id, detection_count, is_detected):
    """Assess track quality based on detection consistency"""
    if track_id not in state.track_quality:
        state.track_quality[track_id] = {'count': 0, 'detections': 0}
    state.track_quality[track_id]['count'] += 1
    if is_detected:
        state.track_quality[track_id]['detections'] += 1
    quality = state.track_quality[track_id]['detections'] / state.track_quality[track_id]['count']
    return quality

def determine_movement_direction(state, track_id, center_x, center_y):
    """Determine vehicle movement direction based on position history"""
    if track_id not in state.vehicle_directions:
        state.vehicle_directions[track_id] = {'positions': deque(maxlen=10), 'direction': None}
    state.vehicle_directions[track_id]['positions'].append((center_x, center_y))
    
    if len(state.vehicle_directions[track_id]['positions']) < 3:
        return None
    
    # Calculate movement vector
    first_x, first_y = state.vehicle_directions[track_id]['positions'][0]
    last_x, last_y = state.vehicle_directions[track_id]['positions'][-1]
    dx = last_x - first_x
    dy = last_y - first_y
    
    # Determine primary direction
    if abs(dx) > abs(dy):
        direction = "right" if dx > 0 else "left"
    else:
        direction = "bottom" if dy > 0 else "top"
    
    state.vehicle_directions[track_id]['direction'] = direction
    return direction

def line_intersects_box(line_start, line_end, box):
    """Check if a line segment intersects with a bounding box"""
    x1, y1 = line_start
    x2, y2 = line_end
    bx1, by1, bx2, by2 = box

    # Helper function to check if a point is on a line segment
    def on_segment(p, q, r):
        return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))

    # Helper function to get orientation of triplet (p, q, r)
    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0
        return 1 if val > 0 else 2

    # Check if line segment (p1, p2) intersects with line segment (p3, p4)
    def do_intersect(p1, p2, p3, p4):
        o1 = orientation(p1, p2, p3)
        o2 = orientation(p1, p2, p4)
        o3 = orientation(p3, p4, p1)
        o4 = orientation(p3, p4, p2)

        if o1 != o2 and o3 != o4:
            return True

        if o1 == 0 and on_segment(p1, p3, p2):
            return True
        if o2 == 0 and on_segment(p1, p4, p2):
            return True
        if o3 == 0 and on_segment(p3, p1, p4):
            return True
        if o4 == 0 and on_segment(p3, p2, p4):
            return True

        return False

    # Define the four edges of the bounding box
    box_edges = [
        ((bx1, by1), (bx2, by1)),  # Top
        ((bx2, by1), (bx2, by2)),  # Right
        ((bx2, by2), (bx1, by2)),  # Bottom
        ((bx1, by2), (bx1, by1))   # Left
    ]

    # Check if the line intersects any edge of the bounding box
    for edge_start, edge_end in box_edges:
        if do_intersect(line_start, line_end, edge_start, edge_end):
            return True

    # Check if the line is completely inside the box
    if (min(x1, x2) >= bx1 and max(x1, x2) <= bx2 and
        min(y1, y2) >= by1 and max(y1, y2) <= by2):
        return True

    return False

def check_line_crossing(state, track_id, box, zone, quality):
    """Check if the vehicle's bounding box intersects the counting line"""
    if quality < 0.5:
        return False
    
    # Prevent counting if ID was already counted anywhere
    if track_id in state.global_counted_ids:
        return False

    x1, y1 = zone['line_start']
    x2, y2 = zone['line_end']
    bx1, by1, bx2, by2 = box

    # Check if the bounding box intersects the counting line
    if line_intersects_box((x1, y1), (x2, y2), (bx1, by1, bx2, by2)):
        # Get the vehicle's direction
        direction = state.vehicle_directions.get(track_id, {}).get('direction')
        
        # Only count if direction matches the line's expected direction
        if direction and direction == zone['direction']:
            zone['counted_ids'].add(track_id)
            state.global_counted_ids.add(track_id)
            
            # Get the vehicle type
            vehicle_type = state.track_labels.get(track_id, "car")
            if vehicle_type in state.vehicle_type_counts:
                state.vehicle_type_counts[vehicle_type] += 1
            
            return True

    return False

def process_frame_optimized(frame, state):
    """Optimized frame processing function"""
    try:
        display_frame = frame.copy()
        
        # Run car detection
        car_results = car_model.predict(source=frame, conf=CONFIG['CONFIDENCE_THRESHOLD'], 
                                      save=False, verbose=False, half=True)
        
        # Prepare detections for tracking
        detections = []
        detection_classes = []
        detection_confidences = []
        
        for r in car_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = car_model.names[cls].lower()
                if label in label_colors:
                    detections.append([x1, y1, x2, y2, conf])
                    detection_classes.append(label)
                    detection_confidences.append(conf)
                    
        # Update tracker
        dets_np = np.array(detections) if detections else np.empty((0, 5))
        tracks = tracker.update(dets_np)
        
        current_ids = set()
        detected_ids = set()
        detection_track_map = {}
        
        # Match detections with tracks
        if len(detections) > 0 and len(tracks) > 0:
            for det_idx, det in enumerate(detections):
                x1, y1, x2, y2, _ = det
                det_box = [x1, y1, x2, y2]
                best_iou = 0
                best_track_idx = -1
                
                for track_idx, track in enumerate(tracks):
                    tx1, ty1, tx2, ty2, track_id = map(int, track)
                    track_box = [tx1, ty1, tx2, ty2]
                    
                    # Calculate intersection over union
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
                    
        # Process each track
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
                
            quality = assess_track_quality(state, track_id, 1, was_detected)
            x1, y1, x2, y2 = smooth_position(state, track_id, (x1, y1, x2, y2))
            
            dominant_class = get_dominant_class(state, track_id, detected_class, detected_conf)
            
            if track_id not in state.track_labels:
                state.track_labels[track_id] = dominant_class
            elif was_detected and detected_class != state.track_labels[track_id]:
                if detected_conf > CONFIG['CONFIDENCE_THRESHOLD'] + 0.1:
                    class_counts = Counter([c for c, _ in state.class_history[track_id][-3:]])
                    if class_counts.get(dominant_class, 0) >= 3:
                        state.track_labels[track_id] = dominant_class
                        
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            direction = determine_movement_direction(state, track_id, center_x, center_y)
            
            # Check counting for this track's bounding box
            for zone in state.counting_zones:
                check_line_crossing(state, track_id, (x1, y1, x2, y2), zone, quality)
                
            label = state.track_labels.get(track_id, "vehicle")
            box_color = label_colors.get(label, (255, 255, 255))
            
            # Draw bounding box and label
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)
            dir_text = f" {direction}" if direction else ""
            cv2.rectangle(display_frame, (x1, y1), (x1 + 140, y1 + 20), box_color, -1)
            cv2.putText(display_frame, f"{label} {track_id}{dir_text}", (x1 + 2, y1 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Plate detection only for high-quality tracks
            if quality > 0.7 and was_detected:
                margin = 10
                vehicle_x1 = max(0, x1 - margin)
                vehicle_y1 = max(0, y1 - margin)
                vehicle_x2 = min(frame.shape[1], x2 + margin)
                vehicle_y2 = min(frame.shape[0], y2 + margin)
                
                vehicle_crop = frame[vehicle_y1:vehicle_y2, vehicle_x1:vehicle_x2]
                
                if vehicle_crop.size > 0 and vehicle_crop.shape[0] > 0 and vehicle_crop.shape[1] > 0:
                    plate_results = plate_model.predict(source=vehicle_crop, 
                                                      conf=CONFIG['PLATE_CONFIDENCE_THRESHOLD'], 
                                                      verbose=False, half=True)
                    
                    best_plate_box = None
                    highest_conf = 0
                    
                    for plate_r in plate_results:
                        for plate_box in plate_r.boxes:
                            px1, py1, px2, py2 = map(int, plate_box.xyxy[0])
                            plate_conf = float(plate_box.conf[0])
                            
                            if plate_conf > highest_conf:
                                highest_conf = plate_conf
                                best_plate_box = (vehicle_x1 + px1, vehicle_y1 + py1,
                                                 vehicle_x1 + px2, vehicle_y1 + py2)
                    
                    if best_plate_box is not None:
                        plate_x1, plate_y1, plate_x2, plate_y2 = best_plate_box
                        cv2.rectangle(display_frame, best_plate_box[:2], best_plate_box[2:], (0, 255, 0), 2)
                        
                        cropped = frame[plate_y1:plate_y2, plate_x1:plate_x2]
                        
                        if cropped.size > 0 and cropped.shape[0] > 0 and cropped.shape[1] > 0:
                            plate_texts = read_plate_with_retry(cropped)
                            
                            if track_id not in state.plate_history:
                                state.plate_history[track_id] = []
                            
                            for text, confidence in plate_texts:
                                state.plate_history[track_id].append((text, confidence))
                            
                            best_plate = None
                            if state.plate_history[track_id]:
                                sorted_plates = sorted(state.plate_history[track_id], key=lambda x: x[1], reverse=True)
                                best_plate = sorted_plates[0][0]
                            
                            if best_plate:
                                last_char = best_plate.strip()[-1]
                                plate_type = "Odd" if last_char.isdigit() and int(last_char) % 2 != 0 else "Even"
                                label_box_height = 20
                                label_box_width = x2 - x1
                                cv2.rectangle(display_frame, (x1, y2 - label_box_height), 
                                              (x1 + label_box_width, y2), (255, 255, 255), -1)
                                cv2.putText(display_frame, f"{best_plate} ({plate_type})", 
                                           (x1 + 2, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Clean up old tracks
        to_remove = [tid for tid in state.previous_positions if tid not in current_ids]
        for tid in to_remove:
            state.previous_positions.pop(tid, None)
            state.track_labels.pop(tid, None)
            state.class_history.pop(tid, None)
            state.position_history.pop(tid, None)
            state.track_quality.pop(tid, None)
            state.vehicle_directions.pop(tid, None)
            state.plate_history.pop(tid, None)
            
        # Draw counting lines
        for zone in state.counting_zones:
            cv2.line(display_frame, zone['line_start'], zone['line_end'], zone['color'], 2)
            cv2.putText(display_frame, zone['name'], 
                       (zone['line_start'][0] + 10, zone['line_start'][1] + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, zone['color'], 2)
            
        # Calculate total count
        total_count = len(state.global_counted_ids)
        
        # Display vehicle counts
        y_offset = 30
        cv2.putText(display_frame, f"Total Vehicles: {total_count}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(display_frame, f"Total Vehicles: {total_count}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        
        # Display vehicle type breakdown
        for vehicle_type, count in state.vehicle_type_counts.items():
            color = label_colors.get(vehicle_type, (255, 255, 255))
            cv2.putText(display_frame, f"{vehicle_type.capitalize()}: {count}", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(display_frame, f"{vehicle_type.capitalize()}: {count}", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 25
            
        return display_frame
                   
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        return frame

def frame_processor(page, image_ref, state):
    """Process frames from the queue"""
    while not stop_processing:
        if not frame_queue:
            time.sleep(0.01)
            continue
            
        frame = frame_queue.pop(0)
        processed_frame = process_frame_optimized(frame, state)
        
        # Encode and update UI
        _, buffer = cv2.imencode('.jpg', processed_frame)
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        with update_lock:
            if image_ref is not None:
                image_ref.src_base64 = img_str
                page.update()

def process_realtime_video_with_status(page, image_ref, video_url, conn_status, progress_bar, error_text, fps_text):
    global cap, stop_processing, frame_width, frame_height, frame_queue
    stop_processing = False
    realtime_state.reset()
    
    try:
        # Start frame processor thread
        processor_thread = threading.Thread(
            target=frame_processor,
            args=(page, image_ref, realtime_state),
            daemon=True
        )
        processor_thread.start()
        
        # Update connection status
        with update_lock:
            conn_status.value = "Connecting to video source..."
            conn_status.color = ft.Colors.YELLOW_400
            progress_bar.visible = True
            progress_bar.value = None  # Indeterminate progress
            error_text.visible = False
            page.update()
        
        # Open the video source
        try:
            cap = cv2.VideoCapture(video_url)
            if not cap.isOpened():
                raise ConnectionError(f"Could not open video stream: {video_url}")
        except Exception as e:
            with update_lock:
                conn_status.value = "Failed to connect to video source"
                conn_status.color = ft.Colors.RED_400
                error_text.value = str(e)
                error_text.visible = True
                page.update()
            return
            
        # Connection successful
        with update_lock:
            conn_status.value = "Connected - Processing frames"
            conn_status.color = ft.Colors.GREEN_400
            progress_bar.value = None  # Indeterminate progress
            page.update()
        
        try:
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            target_fps = min(fps, CONFIG['TARGET_FPS'])
            frame_interval = 1.0 / target_fps
            realtime_state.counting_zones = determine_counting_lines(frame_height, frame_width)
            
            frame_number = 0
            last_frame_time = time.time()
            fps_counter = 0
            last_fps_update = time.time()
            current_fps = 0
            
            while cap.isOpened() and not stop_processing:
                start_time = time.time()
                
                # Skip frames if we're falling behind
                if len(frame_queue) > CONFIG['MAX_FRAME_QUEUE']:
                    ret = cap.grab()  # Just grab the frame without decoding
                    if not ret:
                        break
                    continue
                    
                ret, frame = cap.read()
                if not ret:
                    with update_lock:
                        conn_status.value = "Video stream ended"
                        conn_status.color = ft.Colors.YELLOW_400
                        page.update()
                    break
                    
                frame_number += 1
                frame_queue.append(frame)
                fps_counter += 1
                
                # Update FPS counter every second
                if time.time() - last_fps_update >= 1.0:
                    current_fps = fps_counter
                    fps_counter = 0
                    last_fps_update = time.time()
                    with update_lock:
                        fps_text.value = f"FPS: {current_fps}"
                        page.update()
                
                # Maintain target frame rate
                processing_time = time.time() - start_time
                sleep_time = max(0, frame_interval - processing_time)
                time.sleep(sleep_time)
                
        except Exception as e:
            logger.error(f"Error during video processing: {str(e)}")
            with update_lock:
                conn_status.value = "Processing error"
                conn_status.color = ft.Colors.RED_400
                error_text.value = f"Error: {str(e)}"
                error_text.visible = True
                page.update()
                
    except Exception as e:
        logger.error(f"Error in real-time processing setup: {str(e)}")
        with update_lock:
            conn_status.value = "Setup error"
            conn_status.color = ft.Colors.RED_400
            error_text.value = f"Setup error: {str(e)}"
            error_text.visible = True
            page.update()
            
    finally:
        stop_processing = True
        if cap is not None:
            cap.release()
            cap = None
        with update_lock:
            conn_status.value = "Processing stopped"
            conn_status.color = ft.Colors.GREY_400
            progress_bar.visible = False
            page.update()
        logger.info("Real-time processing stopped")

def start_processing_wrapper(page, image_ref, video_url, conn_status, progress_bar, error_text, fps_text):
    global processing_thread
    with update_lock:
        conn_status.value = "Connecting to video source..."
        conn_status.color = ft.Colors.YELLOW_400
        progress_bar.visible = True
        progress_bar.value = None  # Indeterminate progress
        error_text.visible = False
        page.update()
    
    processing_thread = threading.Thread(
        target=process_realtime_video_with_status,
        args=(page, image_ref, video_url, conn_status, progress_bar, error_text, fps_text),
        daemon=True
    )
    processing_thread.start()

def stop_processing_wrapper(page, conn_status, progress_bar, error_text):
    global stop_processing
    stop_processing = True
    with update_lock:
        conn_status.value = "Stopping processing..."
        conn_status.color = ft.Colors.YELLOW_400
        progress_bar.value = 0
        page.update()

# [Rest of the code remains the same from the original, including main(), upload_processed_video(), process_uploaded_video(), etc.]

def main(page: ft.Page):
    page.title = "Vehiscan - Vehicle Detection System"
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 20
    page.window_width = 1200
    page.window_height = 800
    page.window_resizable = False
    page.fonts = {
        "Inter": "fonts/Inter-Regular.ttf",
        "InterBold": "fonts/Inter-Bold.ttf"
    }
    page.theme = ft.Theme(font_family="Inter")
    
    # Custom colors
    primary_color = ft.Colors.BLUE_400
    secondary_color = ft.Colors.GREY_800
    background_color = ft.Colors.GREY_900
    text_color = ft.Colors.WHITE
    
    # Styles
    title_style = ft.TextStyle(size=24, weight=ft.FontWeight.BOLD, font_family="InterBold")
    button_style = ft.ButtonStyle(
        shape=ft.RoundedRectangleBorder(radius=8),
        padding=15,
        side=ft.BorderSide(1, primary_color)
    )
    
    # Status controls
    connection_status = ft.Text(
        "Not connected",
        color=ft.Colors.GREY_400,
        size=14
    )
    
    processing_progress = ft.ProgressBar(
        width=800,
        height=10,
        value=0,
        color=ft.Colors.BLUE_400,
        bgcolor=ft.Colors.GREY_800,
        visible=False
    )
    
    error_display = ft.Text(
        "",
        color=ft.Colors.RED_400,
        size=14,
        visible=False
    )
    
    fps_counter = ft.Text(
        "FPS: 0",
        color=ft.Colors.GREY_400,
        size=14
    )

    # Loading controls
    realtime_loading = ft.ProgressRing(width=20, height=20, stroke_width=2, visible=False)
    upload_loading = ft.ProgressRing(width=20, height=20, stroke_width=2, visible=False)
    upload_progress = ft.ProgressBar(width=800, height=10, visible=False)
    
    # Realtime processing UI
    video_placeholder = ft.Text(
        "Live video stream will appear here",
        size=16,
        color=ft.Colors.GREY_500,
        text_align=ft.TextAlign.CENTER
    )
    
    realtime_image = ft.Image(
        src="https://www.shutterstock.com/shutterstock/videos/3599757819/thumb/11.jpg?ip=x480",
        width=800,
        height=450,
        fit=ft.ImageFit.CONTAIN,
        border_radius=10
    )
    
    realtime_controls = ft.Column(
        [
            ft.Row(
                [
                    ft.ElevatedButton(
                        "Start Processing",
                        icon=ft.Icons.PLAY_CIRCLE_OUTLINED,
                        on_click=lambda e: start_processing_wrapper(
                            page, realtime_image, CONFIG['VIDEO_URL'],
                            connection_status, processing_progress, error_display, fps_counter
                        ),
                        style=button_style,
                    ),
                    ft.ElevatedButton(
                        "Stop Processing",
                        icon=ft.Icons.STOP_CIRCLE_OUTLINED,
                        on_click=lambda e: stop_processing_wrapper(
                            page, connection_status, processing_progress, error_display
                        ),
                        style=button_style,
                    ),
                    realtime_loading
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=20
            ),
            connection_status,
            processing_progress,
            error_display,
            fps_counter
        ],
        spacing=10
    )
    
    # Upload processing UI
    upload_image = ft.Image(
        src="https://www.shutterstock.com/shutterstock/videos/3705982877/thumb/6.jpg?ip=x480",
        width=800,
        height=450,
        fit=ft.ImageFit.CONTAIN,
        border_radius=10
    )
    
    status_text = ft.Text(
        "No video uploaded",
        color=text_color,
        text_align=ft.TextAlign.CENTER
    )
    
    file_picker = ft.FilePicker(
        on_result=lambda e: (
            threading.Thread(
                target=process_uploaded_video,
                args=(page, e.files[0].path if e.files else None, 
                      upload_image, status_text, upload_button, upload_loading, upload_progress),
                daemon=True
            ).start() if e.files else None
        )
    )
    
    upload_button = ft.ElevatedButton(
        "Upload Video",
        icon=ft.Icons.UPLOAD_FILE,
        on_click=lambda e: file_picker.pick_files(
            allow_multiple=False, 
            allowed_extensions=["mp4", "avi", "mov"]
        ),
        style=button_style,
    )
    
    upload_to_s3_button = ft.ElevatedButton(
        "Upload to AWS",
        icon=ft.Icons.CLOUD_UPLOAD,
        on_click=lambda e: threading.Thread(
            target=upload_processed_video,
            args=(page, status_text, upload_to_s3_button, upload_loading),
            daemon=True
        ).start(),
        style=button_style,
        visible=False
    )
    
    upload_controls = ft.Column(
        [
            ft.Row([upload_button, upload_to_s3_button, upload_loading], 
                  alignment=ft.MainAxisAlignment.CENTER, spacing=20),
            upload_progress,
            status_text
        ],
        alignment=ft.MainAxisAlignment.CENTER,
        spacing=10
    )
    
    # Tab navigation
    def change_page(e):
        global stop_processing, cap
        page_index = e.control.selected_index
        stop_processing = True
        if cap is not None:
            cap.release()
            cap = None
        time.sleep(0.5)
        
        page.controls.clear()
        if page_index == 0:
            page.controls.append(
                ft.Column(
                    [
                        ft.Text("Realtime Processing", style=title_style),
                        ft.Container(
                            realtime_image,
                            border=ft.border.all(1, secondary_color),
                            border_radius=10,
                            padding=10,
                            alignment=ft.alignment.center
                        ),
                        realtime_controls
                    ],
                    alignment=ft.MainAxisAlignment.START,
                    spacing=20,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER
                )
            )
        else:
            page.controls.append(
                ft.Column(
                    [
                        ft.Text("Video Upload", style=title_style),
                        ft.Container(
                            upload_image,
                            border=ft.border.all(1, secondary_color),
                            border_radius=10,
                            padding=10,
                            alignment=ft.alignment.center
                        ),
                        upload_controls
                    ],
                    alignment=ft.MainAxisAlignment.START,
                    spacing=20,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER
                )
            )
        page.update()

    page.navigation_bar = ft.NavigationBar(
        destinations=[
            ft.NavigationBarDestination(
                icon=ft.Icons.VIDEOCAM,
                selected_icon=ft.Icons.VIDEOCAM,
                label="Realtime"
            ),
            ft.NavigationBarDestination(
                icon=ft.Icons.UPLOAD,
                selected_icon=ft.Icons.UPLOAD,
                label="Upload"
            )
        ],
        on_change=change_page,
        height=60,
        elevation=5,
        bgcolor=secondary_color
    )

    # Initial page
    page.controls.append(
        ft.Column(
            [
                ft.Text("Realtime Processing", style=title_style),
                ft.Container(
                    realtime_image,
                    border=ft.border.all(1, secondary_color),
                    border_radius=10,
                    padding=10,
                    alignment=ft.alignment.center
                ),
                realtime_controls
            ],
            alignment=ft.MainAxisAlignment.START,
            spacing=20,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER
        )
    )
    
    page.overlay.append(file_picker)
    page.update()

def upload_processed_video(page, status_text, upload_button, loading_control):
    """Handle the AWS upload process"""
    try:
        with update_lock:
            loading_control.visible = True
            upload_button.disabled = True
            status_text.value = "Uploading to AWS S3..."
            page.update()
        
        output_video_path = CONFIG['OUTPUT_VIDEO_PATH']
        if not os.path.exists(output_video_path):
            raise FileNotFoundError("Processed video file not found")
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_key = f"{AWS_CONFIG['S3_PROCESSED_VIDEOS_FOLDER']}{timestamp}_processed.mp4"
        
        if upload_to_s3(output_video_path, s3_key):
            stats_data = {
                'total_vehicles': len(upload_state.global_counted_ids),
                'vehicle_type_counts': upload_state.vehicle_type_counts,
                'timestamp': timestamp,
                'processed_video_key': s3_key
            }
            
            stats_key = f"{AWS_CONFIG['S3_STATS_FOLDER']}{timestamp}_stats.json"
            upload_data_to_s3(stats_data, stats_key)
            
            with update_lock:
                status_text.value = f"Upload complete! S3 Key: {s3_key}"
                page.snack_bar = ft.SnackBar(
                    ft.Text("Video successfully uploaded to S3!"),
                    bgcolor=ft.Colors.GREEN_400
                )
                page.snack_bar.open = True
        else:
            with update_lock:
                status_text.value = "Failed to upload to S3"
                page.snack_bar = ft.SnackBar(
                    ft.Text("Failed to upload to S3"),
                    bgcolor=ft.Colors.RED_400
                )
                page.snack_bar.open = True
                
    except Exception as e:
        logger.error(f"Error during upload: {str(e)}")
        with update_lock:
            status_text.value = f"Upload error: {str(e)}"
            page.snack_bar = ft.SnackBar(
                ft.Text(f"Upload error: {str(e)}"),
                bgcolor=ft.Colors.RED_400
            )
            page.snack_bar.open = True
            
    finally:
        with update_lock:
            loading_control.visible = False
            upload_button.disabled = False
            page.update()

def process_uploaded_video(page, video_path, image_ref, status_text, upload_button, loading_control, progress_bar):
    global cap, stop_processing, frame_width, frame_height
    stop_processing = False
    upload_state.reset()
    
    try:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        with update_lock:
            loading_control.visible = True
            upload_button.disabled = True
            status_text.value = "Initializing video processing..."
            page.update()
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Failed to open video file")
            
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_skip = max(1, int(fps * CONFIG['FRAME_SKIP_FACTOR']))
        upload_state.counting_zones = determine_counting_lines(frame_height, frame_width)
        
        output_video_path = CONFIG['OUTPUT_VIDEO_PATH']
        writer = cv2.VideoWriter(
            output_video_path, 
            cv2.VideoWriter_fourcc(*'mp4v'), 
            fps, 
            (frame_width, frame_height)
        )
        
        frame_number = 0
        
        with update_lock:
            progress_bar.visible = True
            page.update()
        
        while cap.isOpened() and not stop_processing:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of uploaded video")
                break
                
            frame_number += 1
            if frame_number % frame_skip != 0:
                continue
                
            processed_frame = process_frame_optimized(frame, upload_state)
            writer.write(processed_frame)
            
            # Update UI
            _, buffer = cv2.imencode('.jpg', processed_frame)
            img_str = base64.b64encode(buffer).decode('utf-8')
            
            with update_lock:
                progress = frame_number / total_frames
                progress_bar.value = progress
                status_text.value = f"Processing: {frame_number}/{total_frames} ({progress * 100:.1f}%)"
                if image_ref is not None:
                    image_ref.src_base64 = img_str
                page.update()
                
        with update_lock:
            status_text.value = "Processing complete! Ready to upload to AWS"
            upload_to_s3_button = page.controls[0].controls[2].controls[0].controls[1]  # Get reference to the upload button
            upload_to_s3_button.visible = True
            loading_control.visible = False
            progress_bar.visible = False
            page.update()
            
    except Exception as e:
        logger.error(f"Error processing uploaded video: {str(e)}")
        with update_lock:
            status_text.value = f"Error: {str(e)}"
            loading_control.visible = False
            progress_bar.visible = False
            page.update()
            
    finally:
        if writer is not None:
            writer.release()
        if cap is not None:
            cap.release()
            cap = None

if __name__ == "__main__":
    try:
        ft.app(target=main)
    except Exception as e:
        logger.error(f"Application failed: {str(e)}")