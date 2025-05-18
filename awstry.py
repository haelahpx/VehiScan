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
    'VIDEO_URL': 'https://pbcvideostreams1.pbc.gov/memfs/c4409ce1-c22e-4e4b-8205-d0da82b33165.m3u8',
    'FRAME_SKIP_FACTOR': 0,  # Process every frame  
}

# Add this to your CONFIG section
AWS_CONFIG = {
    'AWS_ACCESS_KEY_ID': 'AKIASEFYCFTAWGH7U4TL',  # Should be set as environment variables
    'AWS_SECRET_ACCESS_KEY': 'Yq/O/8J8aoXsLdSanQCPcvee/MzYYsUyHxB3sw6Z',
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


# Add these helper functions for S3 operations
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
        self.global_counted_ids = set()

realtime_state = ProcessingState()
upload_state = ProcessingState()

update_lock = threading.Lock()

def determine_counting_line(frame_height, frame_width):
    return [
        {
            'line_start': (0, int(frame_height * 0.6)),
            'line_end': (frame_width, int(frame_height * 0.6)),
            'direction': 'bottom',
            'name': 'Horizontal Line Bottom',
            'count': 0,
            'color': (255, 255, 0),
            'counted_ids': set()
        },
        {
            'line_start': (0, int(frame_height * 0.4)),
            'line_end': (frame_width, int(frame_height * 0.4)),
            'direction': 'top',
            'name': 'Horizontal Line Top',
            'count': 0,
            'color': (255, 0, 255),
            'counted_ids': set()
        },
        {
            'line_start': (int(frame_width * 0.6), 0),
            'line_end': (int(frame_width * 0.6), frame_height),
            'direction': 'right',
            'name': 'Vertical Line Right',
            'count': 0,
            'color': (0, 255, 255),
            'counted_ids': set()
        },
        {
            'line_start': (int(frame_width * 0.4), 0),
            'line_end': (int(frame_width * 0.4), frame_height),
            'direction': 'left',
            'name': 'Vertical Line Left',
            'count': 0,
            'color': (255, 0, 0),
            'counted_ids': set()
        }
    ]

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
    if track_id not in state.vehicle_directions:
        state.vehicle_directions[track_id] = {'positions': deque(maxlen=10), 'direction': None}
    state.vehicle_directions[track_id]['positions'].append((center_x, center_y))
    if len(state.vehicle_directions[track_id]['positions']) < 3:
        return None
    first_x, first_y = state.vehicle_directions[track_id]['positions'][0]
    last_x, last_y = state.vehicle_directions[track_id]['positions'][-1]
    dx = last_x - first_x
    dy = last_y - first_y
    
    # Determine primary direction based on movement
    if abs(dx) > abs(dy):  # Horizontal movement
        direction = "right" if dx > 0 else "left"
    else:  # Vertical movement
        direction = "bottom" if dy > 0 else "top"
    
    state.vehicle_directions[track_id]['direction'] = direction
    return direction

def check_line_crossing(state, track_id, center_x, center_y, zone, quality):
    if track_id not in state.previous_positions:
        state.previous_positions[track_id] = (center_x, center_y)
        return False
        
    prev_x, prev_y = state.previous_positions[track_id]
    state.previous_positions[track_id] = (center_x, center_y)
    
    x1, y1 = zone['line_start']
    x2, y2 = zone['line_end']
    
    if quality < 0.5:
        return False
        
    # Check if this ID has already been counted in any zone
    if track_id in state.global_counted_ids:
        return False
    
    crossed = False
    
    if y1 == y2:  # Horizontal line
        if (prev_y < y1 and center_y >= y1) or (prev_y > y1 and center_y <= y1):
            crossed_direction = "bottom" if center_y > prev_y else "top"
            if zone['direction'] == crossed_direction:
                crossed = True
    elif x1 == x2:  # Vertical line
        if (prev_x < x1 and center_x >= x1) or (prev_x > x1 and center_x <= x1):
            crossed_direction = "right" if center_x > prev_x else "left"
            if zone['direction'] == crossed_direction:
                crossed = True
    
    if crossed:
        zone['counted_ids'].add(track_id)
        state.global_counted_ids.add(track_id)
        zone['count'] += 1
        return True
        
    return False

def process_frame(frame, frame_number, total_frames, writer, page, image_ref, state):
    try:
        ocr_frame = frame.copy()
        display_frame = frame.copy()
        car_results = car_model.predict(source=frame, conf=CONFIG['CONFIDENCE_THRESHOLD'], save=False, verbose=False)
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
                
            quality = assess_track_quality(state, track_id, frame_number, was_detected)
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
            
            for zone in state.counting_zones:
                check_line_crossing(state, track_id, center_x, center_y, zone, quality)
                
            label = state.track_labels.get(track_id, "vehicle")
            box_color = label_colors.get(label, (255, 255, 255))
            
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)
            dir_text = f" {direction}" if direction else ""
            cv2.rectangle(display_frame, (x1, y1), (x1 + 140, y1 + 20), box_color, -1)
            cv2.putText(display_frame, f"{label} {track_id}{dir_text} Q:{quality:.2f}", (x1 + 2, y1 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # === PLATE DETECTION AND READING ===
            margin = 10
            vehicle_x1 = max(0, x1 - margin)
            vehicle_y1 = max(0, y1 - margin)
            vehicle_x2 = min(frame.shape[1], x2 + margin)
            vehicle_y2 = min(frame.shape[0], y2 + margin)
            
            vehicle_crop = ocr_frame[vehicle_y1:vehicle_y2, vehicle_x1:vehicle_x2]
            
            best_plate_box = None
            highest_conf = 0
            
            if vehicle_crop.size > 0 and vehicle_crop.shape[0] > 0 and vehicle_crop.shape[1] > 0:
                plate_results = plate_model.predict(source=vehicle_crop, conf=CONFIG['PLATE_CONFIDENCE_THRESHOLD'], verbose=False)
                
                for plate_r in plate_results:
                    for plate_box in plate_r.boxes:
                        px1, py1, px2, py2 = map(int, plate_box.xyxy[0])
                        plate_conf = float(plate_box.conf[0])
                        
                        plate_x1 = vehicle_x1 + px1
                        plate_y1 = vehicle_y1 + py1
                        plate_x2 = vehicle_x1 + px2
                        plate_y2 = vehicle_y1 + py2
                        
                        if plate_conf > highest_conf:
                            highest_conf = plate_conf
                            best_plate_box = (plate_x1, plate_y1, plate_x2, plate_y2)
            
            if best_plate_box is not None:
                plate_x1, plate_y1, plate_x2, plate_y2 = best_plate_box
                cv2.rectangle(display_frame, best_plate_box[:2], best_plate_box[2:], (0, 255, 0), 2)
                
                cropped = ocr_frame[plate_y1:plate_y2, plate_x1:plate_x2]

                if cropped.size > 0 and cropped.shape[0] > 0 and cropped.shape[1] > 0:
                    plate_texts = read_plate_with_retry(cropped)

                    if track_id not in state.plate_history:
                        state.plate_history[track_id] = []

                    for text, confidence in plate_texts:
                        state.plate_history[track_id].append((text, confidence))

                    # Get the best plate for this track
                    best_plate = None
                    if state.plate_history[track_id]:
                        sorted_plates = sorted(state.plate_history[track_id], key=lambda x: x[1], reverse=True)
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
        to_remove = [tid for tid in state.previous_positions if tid not in current_ids]
        for tid in to_remove:
            state.previous_positions.pop(tid, None)
            state.track_labels.pop(tid, None)
            state.class_history.pop(tid, None)
            state.position_history.pop(tid, None)
            state.track_quality.pop(tid, None)
            state.vehicle_directions.pop(tid, None)
            state.plate_history.pop(tid, None)
            
        state.vehicle_count = sum(zone['count'] for zone in state.counting_zones)
        
        for i, zone in enumerate(state.counting_zones):
            cv2.line(display_frame, zone['line_start'], zone['line_end'], zone['color'], 2)
            start_x, start_y = zone['line_start']
            end_x, end_y = zone['line_end']
            mid_x, mid_y = (start_x + end_x) // 2, (start_y + end_y) // 2
            arrow_len = 20
            arrow_x, arrow_y = {
                'bottom': (mid_x, mid_y + arrow_len),
                'top': (mid_x, mid_y - arrow_len),
                'right': (mid_x + arrow_len, mid_y),
                'left': (mid_x - arrow_len, mid_y),
                'any': (mid_x, mid_y)
            }.get(zone['direction'], (mid_x, mid_y))
            cv2.arrowedLine(display_frame, (mid_x, mid_y), (arrow_x, arrow_y), zone['color'], 2)
            label_y = 40 + i * 30
            cv2.putText(display_frame, f"{zone['name']}: {zone['count']}", (20, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(display_frame, f"{zone['name']}: {zone['count']}", (20, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, zone['color'], 2)
                       
        cv2.putText(display_frame, f"Total Count: {state.vehicle_count}", (20, frame_height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
        cv2.putText(display_frame, f"Total Count: {state.vehicle_count}", (20, frame_height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Frame: {frame_number}/{total_frames}", (frame_width - 200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(display_frame, f"Frame: {frame_number}/{total_frames}", (frame_width - 200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                   
        if writer is not None:
            writer.write(display_frame)
            
        _, buffer = cv2.imencode('.jpg', display_frame)
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        if image_ref is not None:
            with update_lock:
                image_ref.src_base64 = img_str
                page.update()
                
    except Exception as e:
        logger.error(f"Error processing frame {frame_number}: {str(e)}")

def process_realtime_video(page, image_ref, video_url):
    global cap, stop_processing, frame_width, frame_height
    stop_processing = False
    writer = None
    realtime_state.reset()
    
    try:
        cap = cv2.VideoCapture(video_url)
        if not cap.isOpened():
            fallback_url = 0  # Fallback to default camera
            cap = cv2.VideoCapture(fallback_url)
            if not cap.isOpened():
                with update_lock:
                    page.snack_bar = ft.SnackBar(ft.Text("Failed to open video stream or fallback camera"))
                    page.snack_bar.open = True
                    page.update()
                return
            logger.warning("Video URL failed, using fallback camera")
            
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_skip = max(1, int(fps * CONFIG['FRAME_SKIP_FACTOR']))
        realtime_state.counting_zones = determine_counting_line(frame_height, frame_width)
        
        writer = cv2.VideoWriter(CONFIG['OUTPUT_VIDEO_PATH'], cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        frame_number = 0
        
        while cap.isOpened() and not stop_processing:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video stream")
                break
                
            frame_number += 1
            if frame_number % frame_skip != 0:
                continue
                
            process_frame(frame, frame_number, -1, writer, page, image_ref, realtime_state)
            time.sleep(0.01)  # Prevent CPU overload
            
    except Exception as e:
        logger.error(f"Error in real-time processing: {str(e)}")
        with update_lock:
            page.snack_bar = ft.SnackBar(ft.Text(f"Error in real-time processing: {str(e)}"))
            page.snack_bar.open = True
            page.update()
            
    finally:
        if writer is not None:
            writer.release()
        if cap is not None:
            cap.release()
            cap = None
        logger.info("Real-time processing stopped")

def process_uploaded_video(page, video_path, image_ref, status_text, download_button, s3_upload_button):
    global cap, stop_processing, frame_width, frame_height
    stop_processing = False
    writer = None
    upload_state.reset()
    
    try:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        output_dir = os.path.dirname(CONFIG['OUTPUT_VIDEO_PATH'])
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Failed to open video file")
            
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_skip = max(1, int(fps * CONFIG['FRAME_SKIP_FACTOR']))
        upload_state.counting_zones = determine_counting_line(frame_height, frame_width)
        
        # Generate output filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"processed_{timestamp}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        writer = cv2.VideoWriter(
            output_path, 
            cv2.VideoWriter_fourcc(*'mp4v'), 
            fps, 
            (frame_width, frame_height)
        )
        
        frame_number = 0
        
        while cap.isOpened() and not stop_processing:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of uploaded video")
                break
                
            frame_number += 1
            if frame_number % frame_skip != 0:
                continue
                
            process_frame(frame, frame_number, total_frames, writer, page, image_ref, upload_state)
            
            with update_lock:
                status_text.value = f"Processing: {frame_number}/{total_frames} ({(frame_number / total_frames) * 100:.1f}%)"
                page.update()
                
        # After processing is complete
        if writer is not None:
            writer.release()
            
        with update_lock:
            status_text.value = "Processing complete!"
            download_button.disabled = False
            s3_upload_button.disabled = False
            page.update()
            
        return output_path
            
    except Exception as e:
        logger.error(f"Error processing uploaded video: {str(e)}")
        with update_lock:
            status_text.value = f"Error: {str(e)}"
            page.update()
        return None
            
    finally:
        if writer is not None:
            writer.release()
        if cap is not None:
            cap.release()
            cap = None
            
def main(page: ft.Page):
    global stop_processing, cap

    page.title = "Vehicle Detection and Counting System"
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 20

    processed_video_path = None
    video_url = CONFIG['VIDEO_URL']

    video_placeholder = ft.Text(
        "Live video stream (HLS not supported in Flet; processed output shown below)",
        size=16,
        color=ft.Colors.GREY
    )
    realtime_image = ft.Image(
        src_base64="",
        width=800,
        height=450,
        fit=ft.ImageFit.CONTAIN
    )

    def start_processing(_):
        threading.Thread(
            target=process_realtime_video,
            args=(page, realtime_image, video_url),
            daemon=True
        ).start()

    def stop_processing_func(_):
        global stop_processing
        stop_processing = True

    start_button = ft.ElevatedButton("Start Processing", on_click=start_processing)
    stop_button = ft.ElevatedButton("Stop Processing", on_click=stop_processing_func)

    upload_image = ft.Image(
        src_base64="",
        width=800,
        height=450,
        fit=ft.ImageFit.CONTAIN,
        border_radius=ft.border_radius.all(10)
    )
    status_text = ft.Text("No video uploaded")

    # Directory picker
    directory_picker = ft.FilePicker()
    loading_ring = ft.ProgressRing(visible=False)

    def handle_directory_pick(e):
        nonlocal processed_video_path
        if not e.path or not processed_video_path:
            page.snack_bar.content.value = "No directory selected or no processed video"
            page.snack_bar.open = True
            page.update()
            return

        filename = os.path.basename(processed_video_path)
        destination_path = os.path.join(e.path, filename)

        try:
            shutil.copy(processed_video_path, destination_path)
            page.snack_bar.content.value = f"Video saved to: {destination_path}"
        except Exception as err:
            page.snack_bar.content.value = f"Failed to save video: {str(err)}"

        page.snack_bar.open = True
        page.update()

    directory_picker.on_result = handle_directory_pick

    download_button = ft.ElevatedButton(
        "Download Processed Video",
        disabled=True,
        on_click=lambda _: directory_picker.get_directory_path()
    )

    def s3_upload_handler():
        nonlocal processed_video_path
        if not processed_video_path:
            page.snack_bar.content.value = "No processed video available to upload"
            page.snack_bar.open = True
            page.update()
            return

        # Show loading ring
        loading_ring.visible = True
        page.snack_bar.content.value = "Uploading to S3..."
        page.snack_bar.open = True
        page.update()

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_key = generate_s3_key(
            AWS_CONFIG['S3_PROCESSED_VIDEOS_FOLDER'],
            f"processed_{timestamp}.mp4"
        )

        try:
            if upload_to_s3(processed_video_path, s3_key):
                plate_data = []
                if hasattr(upload_state, 'plate_history'):
                    for track_id, plates in upload_state.plate_history.items():
                        for plate_text, confidence in plates:
                            plate_data.append({
                                'track_id': track_id,
                                'plate_text': plate_text,
                                'confidence': confidence,
                                'timestamp': datetime.datetime.now().isoformat()
                            })

                if plate_data:
                    plate_filename = f"plate_data_{timestamp}.json"
                    plate_s3_key = generate_s3_key(AWS_CONFIG['S3_PLATE_DATA_FOLDER'], plate_filename)
                    upload_data_to_s3({
                        'video_file': s3_key,
                        'plate_data': plate_data,
                        'vehicle_count': upload_state.vehicle_count,
                        'counting_zones': [
                            {k: v for k, v in zone.items() if k != 'counted_ids'} 
                            for zone in upload_state.counting_zones
                        ]
                    }, plate_s3_key)

                page.snack_bar.content.value = "Upload complete!"
            else:
                page.snack_bar.content.value = "Upload failed!"
        except Exception as e:
            page.snack_bar.content.value = f"Error: {str(e)}"

        # Hide loading ring
        loading_ring.visible = False
        page.snack_bar.open = True
        page.update()

    s3_upload_button = ft.ElevatedButton(
        "Save to S3",
        disabled=True,
        icon=ft.Icons.CLOUD_UPLOAD,
        on_click=lambda _: threading.Thread(target=s3_upload_handler, daemon=True).start()
    )

    def file_upload_handler(e):
        nonlocal processed_video_path
        if e.files:
            filepath = e.files[0].path
            status_text.value = "Processing video..."
            download_button.disabled = True
            s3_upload_button.disabled = True
            page.update()

            processed_video_path = process_uploaded_video(
                page,
                filepath,
                upload_image,
                status_text,
                download_button,
                s3_upload_button
            )

    file_picker = ft.FilePicker(on_result=file_upload_handler)

    upload_button = ft.ElevatedButton(
        "Upload Video",
        icon=ft.Icons.UPLOAD_FILE,
        on_click=lambda _: file_picker.pick_files(
            allow_multiple=False,
            allowed_extensions=["mp4", "avi", "mov"]
        )
    )

    page.snack_bar = ft.SnackBar(
        content=ft.Text(""),
        action="OK",
        on_action=lambda _: setattr(page.snack_bar, "open", False)
    )

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
                ft.Column([
                    ft.Text("Realtime Video Processing", size=24, weight=ft.FontWeight.BOLD),
                    video_placeholder,
                    ft.Row([start_button, stop_button], alignment=ft.MainAxisAlignment.CENTER),
                    realtime_image
                ])
            )
        else:
            page.controls.append(
                ft.Column([
                    ft.Text("Upload and Process Video", size=24, weight=ft.FontWeight.BOLD),
                    upload_button,
                    upload_image,
                    status_text,
                    ft.Row([download_button, s3_upload_button, loading_ring], spacing=10)
                ])
            )
        page.update()

    page.navigation_bar = ft.NavigationBar(
        destinations=[
            ft.NavigationBarDestination(icon=ft.Icons.VIDEOCAM, label="Realtime"),
            ft.NavigationBarDestination(icon=ft.Icons.UPLOAD, label="Upload")
        ],
        on_change=change_page
    )

    page.controls.append(
        ft.Column([
            ft.Text("Upload and Process Video", size=24, weight=ft.FontWeight.BOLD),
            upload_button,
            upload_image,
            status_text,
            ft.Row([download_button, s3_upload_button, loading_ring], spacing=10)
        ])
    )

    page.overlay.append(file_picker)
    page.overlay.append(directory_picker)
    page.update()
    


if __name__ == "__main__":
    try:
        ft.app(target=main)
    except Exception as e:
        logger.error(f"Application failed: {str(e)}")