import cv2
import os
import numpy as np
import easyocr
from sort.sort import *
import datetime
from ultralytics import YOLO
import torch
import re
import collections

class TrafficMonitoringSystem:
    """Improved traffic monitoring system with advanced vehicle counting and plate recognition"""
    
    def __init__(self, config=None):
        # Default configuration
        self.config = {
            "car_model_path": "models/vehicle_model(detrac)/weights/best.pt",
            "plate_model_path": "models/plat_model/weights/best.pt",
            "video_input_path": os.path.join("assets", "testing_video.mp4"),
            "confidence_threshold": 0.5,
            "plate_confidence_threshold": 0.7,
            "ocr_min_confidence": 0.6,
            "counting_line_position": 0.8,  # Relative position (0-1)
            "counting_line_offset": 0.01,   # Relative position (0-1)
            "plate_pattern": r"[A-Z0-9]{3,8}",  # Basic regex for plate format validation
            "min_vehicle_detections": 3,    # Minimum detections before counting a vehicle
            "frame_processing_interval": 1,  # Process every n frames
        }
        
        # Update with custom config if provided
        if config:
            self.config.update(config)
        
        # Create output paths based on timestamp
        self.setup_paths()
        
        # Load models
        self.setup_models()
        
        # Initialize tracking and counting variables
        self.tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
        self.vehicle_count = 0
        self.vehicle_types = collections.Counter()
        self.counted_ids = set()
        self.plate_history = {}  # track_id: list of (text, confidence)
        self.track_history = {}  # track_id: list of center positions (for smoother tracking)
        self.detection_counter = {}  # track_id: number of times detected
        
        # Label colors
        self.label_colors = {
            'car': (0, 255, 0),
            'truck': (255, 0, 0),
            'bus': (0, 0, 255),
            'van': (255, 255, 0)
        }
        
    def setup_paths(self):
        """Set up output paths with timestamps"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("outputs", exist_ok=True)
        
        output_video_filename = f"video_result_{timestamp}.mp4"
        self.video_output_path = os.path.join("outputs", output_video_filename)
        self.log_output_path = os.path.join("outputs", f"logs_{timestamp}.txt")
        self.csv_output_path = os.path.join("outputs", f"vehicle_data_{timestamp}.csv")
        
    def setup_models(self):
        """Initialize and configure the detection models"""
        # Set device based on availability
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load models
        print("Loading vehicle detection model...")
        self.car_model = YOLO(self.config["car_model_path"])
        self.car_model.to(self.device)
        
        print("Loading license plate detection model...")
        self.plate_model = YOLO(self.config["plate_model_path"])
        self.plate_model.to(self.device)
        
        # Initialize OCR
        print("Initializing OCR engine...")
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        
        # Print device info
        if torch.cuda.is_available():
            print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA is not available. Using CPU.")
            
    def preprocess_plate_text(self, text):
        """Clean and validate license plate text"""
        # Remove spaces and convert to uppercase
        cleaned_text = ''.join(text.upper().split())
        
        # Apply regex matching for valid plate format
        match = re.search(self.config["plate_pattern"], cleaned_text)
        if match:
            return match.group(0)
        
        return None
    
    def get_plate_type(self, plate_text):
        """Determine plate type (Odd/Even) based on last character"""
        if not plate_text:
            return "Unknown"
            
        last_char = plate_text.strip()[-1]
        if last_char.isdigit():
            return "Odd" if int(last_char) % 2 != 0 else "Even"
        return "Unknown"
    
    def update_plate_history(self, track_id, text, confidence):
        """Update the plate history with new detection"""
        if not text or len(text.strip()) < 3:
            return
            
        processed_text = self.preprocess_plate_text(text)
        if not processed_text:
            return
            
        if track_id not in self.plate_history:
            self.plate_history[track_id] = []
            
        # Only keep the top 5 most confident detections
        self.plate_history[track_id].append((processed_text, confidence))
        self.plate_history[track_id] = sorted(
            self.plate_history[track_id],
            key=lambda x: x[1],
            reverse=True
        )[:5]
    
    def get_best_plate(self, track_id):
        """Get the most confident plate for a track ID using frequency and confidence"""
        if track_id not in self.plate_history or not self.plate_history[track_id]:
            return None
            
        # Count occurrences of each plate text
        plates = [plate[0] for plate in self.plate_history[track_id]]
        plate_counter = collections.Counter(plates)
        
        # If we have a clear winner by frequency, use it
        most_common = plate_counter.most_common(1)[0]
        if most_common[1] >= 2 and len(self.plate_history[track_id]) >= 3:
            return most_common[0]
            
        # Otherwise use the highest confidence
        return self.plate_history[track_id][0][0]
    
    def update_tracking_history(self, track_id, center_x, center_y):
        """Update the tracking history for smoother path prediction"""
        if track_id not in self.track_history:
            self.track_history[track_id] = []
            
        self.track_history[track_id].append((center_x, center_y))
        
        # Keep only the most recent positions
        if len(self.track_history[track_id]) > 10:
            self.track_history[track_id].pop(0)
    
    def get_predicted_position(self, track_id):
        """Predict the next position based on tracking history"""
        if track_id not in self.track_history or len(self.track_history[track_id]) < 2:
            return None
            
        positions = self.track_history[track_id]
        if len(positions) >= 3:
            # Calculate velocity from last few positions
            dx = positions[-1][0] - positions[-3][0]
            dy = positions[-1][1] - positions[-3][1]
            
            # Predict next position
            next_x = positions[-1][0] + dx/2
            next_y = positions[-1][1] + dy/2
            
            return (next_x, next_y)
            
        return positions[-1]
    
    def count_vehicle(self, track_id, center_y, previous_y, vehicle_class):
        """Count a vehicle if it crosses the counting line"""
        # Get line position and offset
        line_position = self.line_position
        line_offset = self.line_offset
        
        # Check if vehicle has enough detections to be counted
        min_detections = self.config["min_vehicle_detections"]
        if self.detection_counter.get(track_id, 0) < min_detections:
            return False
            
        # Check if vehicle crosses the line
        if previous_y is not None and center_y > previous_y:
            if ((previous_y < line_position and center_y >= line_position) or
                (line_position - line_offset < center_y < line_position + line_offset)):
                
                if track_id not in self.counted_ids:
                    self.vehicle_count += 1
                    self.vehicle_types[vehicle_class] += 1
                    self.counted_ids.add(track_id)
                    return True
        
        return False
    
    def process_video(self):
        """Process the input video and generate output"""
        # Open input video
        cap = cv2.VideoCapture(self.config["video_input_path"])
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate line position
        self.line_position = int(frame_height * self.config["counting_line_position"])
        self.line_offset = int(frame_height * self.config["counting_line_offset"])
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.video_output_path, fourcc, fps, (frame_width, frame_height))
        
        # Prepare CSV header
        with open(self.csv_output_path, 'w') as csv_file:
            csv_file.write("timestamp,track_id,vehicle_type,license_plate,plate_type,x1,y1,x2,y2\n")
        
        # Process frames
        frame_number = 0
        processing_interval = self.config["frame_processing_interval"]
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_number += 1
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                
                # Print progress
                progress = (frame_number / total_frames) * 100
                print(f"Processing: {progress:.2f}% (Frame {frame_number}/{total_frames})", end="\r")
                
                # Process every nth frame (for speed)
                if frame_number % processing_interval != 0 and frame_number > 1:
                    out.write(frame)
                    continue
                
                # Process this frame
                processed_frame = self.process_frame(frame, frame_number, timestamp)
                out.write(processed_frame)
        
        except Exception as e:
            print(f"Error processing frame {frame_number}: {str(e)}")
            
        finally:
            # Cleanup
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            
            # Write final log
            self.write_log_file()
            
            print(f"\nProcessed video saved to {self.video_output_path}")
            print(f"Vehicle data saved to {self.csv_output_path}")
            print(f"Log file saved to {self.log_output_path}")
    
    def process_frame(self, frame, frame_number, timestamp):
        """Process a single frame"""
        # Make a copy of the frame
        result_frame = frame.copy()
        
        # Detect vehicles
        car_results = self.car_model.predict(
            source=frame, 
            conf=self.config["confidence_threshold"], 
            save=False, 
            verbose=False
        )
        
        # Extract detections
        detections = []
        class_names = []
        
        for r in car_results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = self.car_model.names[cls].lower()
                
                if label in self.label_colors:
                    detections.append([x1, y1, x2, y2, conf])
                    class_names.append(label)
        
        # Update tracker
        dets_np = np.array(detections) if detections else np.empty((0, 5))
        tracks = self.tracker.update(dets_np)
        
        # Track IDs in current frame
        current_ids = set()
        
        # Draw counting line
        cv2.line(result_frame, (0, self.line_position), 
                 (frame.shape[1], self.line_position), (255, 255, 0), 2)
        
        # Process each track
        for i, track in enumerate(tracks):
            x1, y1, x2, y2, track_id = map(int, track)
            track_id = int(track_id)
            current_ids.add(track_id)
            
            # Update detection counter
            self.detection_counter[track_id] = self.detection_counter.get(track_id, 0) + 1
            
            # Calculate center position
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # Update tracking history
            self.update_tracking_history(track_id, center_x, center_y)
            
            # Get previous position
            previous_pos = None
            if track_id in self.track_history and len(self.track_history[track_id]) > 1:
                previous_pos = self.track_history[track_id][-2][1]  # Y-coordinate
            
            # Get vehicle class
            vehicle_class = class_names[i] if i < len(class_names) else "vehicle"
            
            # Count vehicle if crossing line
            vehicle_counted = self.count_vehicle(track_id, center_y, previous_pos, vehicle_class)
            
            # Draw bounding box and label
            box_color = self.label_colors.get(vehicle_class, (255, 255, 255))
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), box_color, 2)
            
            # Get license plate for this vehicle
            plate_text = None
            plate_type = "Unknown"
            
            # Only process license plate every few frames to save processing time
            if frame_number % 3 == 0:
                self.process_license_plate(result_frame, x1, y1, x2, y2, track_id)
            
            # Get best plate estimation
            best_plate = self.get_best_plate(track_id)
            if best_plate:
                plate_text = best_plate
                plate_type = self.get_plate_type(best_plate)
                
                # Draw plate information
                self.draw_plate_info(result_frame, x1, y1, best_plate, plate_type)
            
            # Write vehicle info to CSV
            with open(self.csv_output_path, 'a') as csv_file:
                plate_str = plate_text if plate_text else "Unknown"
                csv_file.write(f"{timestamp:.3f},{track_id},{vehicle_class},{plate_str},{plate_type},{x1},{y1},{x2},{y2}\n")
            
            # Draw vehicle info on frame
            status = "COUNTED" if track_id in self.counted_ids else ""
            cv2.putText(result_frame, f"{vehicle_class.capitalize()} {track_id} {status}", 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            
            # Draw tracking path
            if track_id in self.track_history and len(self.track_history[track_id]) > 1:
                points = np.array(self.track_history[track_id], dtype=np.int32)
                cv2.polylines(result_frame, [points], False, box_color, 2)
        
        # Clean up old tracks
        to_remove = [tid for tid in self.track_history if tid not in current_ids]
        for tid in to_remove:
            if tid in self.track_history:
                del self.track_history[tid]
        
        # Draw counter
        self.draw_counter(result_frame)
        
        return result_frame
    
    def process_license_plate(self, frame, x1, y1, x2, y2, track_id):
        """Process and recognize license plate in the given region"""
        # Define plate region (bottom part of vehicle)
        plate_h = (y2 - y1) // 3
        plate_w = (x2 - x1) // 2
        plate_x1 = x1 + (x2 - x1 - plate_w) // 2
        plate_y1 = y2 - plate_h
        plate_x2 = plate_x1 + plate_w
        plate_y2 = y2
        
        # Ensure we're not out of bounds
        plate_x1 = max(0, plate_x1)
        plate_y1 = max(0, plate_y1)
        plate_x2 = min(frame.shape[1], plate_x2)
        plate_y2 = min(frame.shape[0], plate_y2)
        
        # Skip if region is too small
        if plate_x2 - plate_x1 < 20 or plate_y2 - plate_y1 < 10:
            return
        
        # Crop plate region
        cropped = frame[plate_y1:plate_y2, plate_x1:plate_x2]
        
        if cropped.size > 0 and cropped.shape[0] > 0 and cropped.shape[1] > 0:
            # Detect plates using YOLO
            plate_results = self.plate_model.predict(
                source=cropped, 
                conf=self.config["plate_confidence_threshold"], 
                save=False, 
                verbose=False
            )
            
            # If we have plate detections, focus OCR on those regions
            plate_found = False
            
            for p in plate_results:
                boxes = p.boxes
                for box in boxes:
                    px1, py1, px2, py2 = map(int, box.xyxy[0])
                    
                    # Ensure bounds
                    px1 = max(0, px1)
                    py1 = max(0, py1)
                    px2 = min(cropped.shape[1], px2)
                    py2 = min(cropped.shape[0], py2)
                    
                    if px2 - px1 < 10 or py2 - py1 < 5:
                        continue
                    
                    # Crop to detected plate
                    plate_region = cropped[py1:py2, px1:px2]
                    
                    # Apply pre-processing for better OCR
                    plate_region = self.preprocess_plate_image(plate_region)
                    
                    # OCR on plate region
                    plate_texts = self.reader.readtext(plate_region)
                    
                    # Process OCR results
                    for result in plate_texts:
                        text, confidence = result[1], result[2]
                        if confidence >= self.config["ocr_min_confidence"]:
                            self.update_plate_history(track_id, text, confidence)
                            plate_found = True
            
            # If no plates detected or no good OCR results, try on the whole cropped area
            if not plate_found:
                # Apply pre-processing
                processed_crop = self.preprocess_plate_image(cropped)
                
                # OCR on whole cropped area
                plate_texts = self.reader.readtext(processed_crop)
                
                # Process OCR results
                for result in plate_texts:
                    text, confidence = result[1], result[2]
                    if confidence >= self.config["ocr_min_confidence"]:
                        self.update_plate_history(track_id, text, confidence)
    
    def preprocess_plate_image(self, img):
        """Preprocess plate image for better OCR results"""
        # Resize to reasonable OCR size if too small
        h, w = img.shape[:2]
        if w < 100:
            scale_factor = 100 / w
            img = cv2.resize(img, (100, int(h * scale_factor)))
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to remove noise while keeping edges
        blur = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply morphological operations
        kernel = np.ones((3, 3), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        return morph
    
    def draw_plate_info(self, frame, x1, y1, plate_text, plate_type):
        """Draw plate information on the frame"""
        # Create label image with plate info
        label_img = np.ones((60, 180, 3), dtype=np.uint8) * 255
        
        # Draw plate text and type
        cv2.putText(label_img, f"{plate_text}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(label_img, f"({plate_type})", (10, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Find position to overlay the label
        label_x = max(0, x1)
        label_y = max(0, y1 - 70)
        
        # Ensure it fits within frame
        h, w = label_img.shape[:2]
        if label_y + h >= frame.shape[0]:
            label_y = max(0, frame.shape[0] - h)
        if label_x + w >= frame.shape[1]:
            label_x = max(0, frame.shape[1] - w)
        
        # Create a semi-transparent overlay for better readability
        alpha = 0.7
        overlay = frame[label_y:label_y + h, label_x:label_x + w].copy()
        cv2.addWeighted(label_img, alpha, overlay, 1 - alpha, 0, 
                       frame[label_y:label_y + h, label_x:label_x + w])
    
    def draw_counter(self, frame):
        """Draw counter and statistics on the frame"""
        # Draw main counter
        cv2.putText(frame, f"Total Count: {self.vehicle_count}", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
        cv2.putText(frame, f"Total Count: {self.vehicle_count}", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        
        # Draw vehicle type counts
        y_offset = 80
        for i, (vehicle_type, count) in enumerate(self.vehicle_types.most_common()):
            text = f"{vehicle_type.capitalize()}: {count}"
            cv2.putText(frame, text, (20, y_offset + i * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
            cv2.putText(frame, text, (20, y_offset + i * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    def write_log_file(self):
        """Write detailed log file with results"""
        with open(self.log_output_path, "w") as log_file:
            log_file.write(f"=== Traffic Monitoring Report ===\n")
            log_file.write(f"Processed at: {datetime.datetime.now()}\n")
            log_file.write(f"Input Video: {self.config['video_input_path']}\n")
            log_file.write(f"Output Video: {self.video_output_path}\n")
            log_file.write(f"CSV Data: {self.csv_output_path}\n\n")
            
            log_file.write(f"=== Vehicle Statistics ===\n")
            log_file.write(f"Total Vehicles Counted: {self.vehicle_count}\n")
            
            log_file.write("\nVehicle Types:\n")
            for vehicle_type, count in self.vehicle_types.most_common():
                percentage = (count / self.vehicle_count * 100) if self.vehicle_count > 0 else 0
                log_file.write(f"  {vehicle_type.capitalize()}: {count} ({percentage:.1f}%)\n")
            
            log_file.write("\n=== Detected License Plates ===\n")
            for track_id in sorted(self.plate_history.keys()):
                if track_id in self.counted_ids and self.plate_history[track_id]:
                    best_plate = self.get_best_plate(track_id)
                    if best_plate:
                        plate_type = self.get_plate_type(best_plate)
                        log_file.write(f"  Vehicle ID {track_id}: {best_plate} ({plate_type})\n")
            
            log_file.write("\n=== Configuration ===\n")
            for key, value in self.config.items():
                log_file.write(f"  {key}: {value}\n")
            
            log_file.write("\n" + "="*40 + "\n")


# Usage example
if __name__ == "__main__":
    # Define custom configuration (optional)
    custom_config = {
        "video_input_path": os.path.join("assets", "testing_video.mp4"),
        "confidence_threshold": 0.4,
        "plate_confidence_threshold": 0.7,
        "ocr_min_confidence": 0.65,
        "plate_pattern": r"[A-Z0-9]{3,8}",  # Adjust according to your region's plate format
        "min_vehicle_detections": 5,
        "frame_processing_interval": 2,  # Process every second frame for speed
    }
    
    # Initialize the system
    traffic_system = TrafficMonitoringSystem(custom_config)
        
    # Process the video
    traffic_system.process_video()