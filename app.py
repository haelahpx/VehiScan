from flask import Flask, render_template, Response, send_file
import cv2
import os
import numpy as np
import easyocr
from sort.sort import *
import datetime
from ultralytics import YOLO
import torch
from collections import Counter, deque
import io
from threading import Thread

app = Flask(__name__)

# === CONFIG ===
CAR_MODEL_PATH = "models/testv3.pt"
PLATE_MODEL_PATH = "models/plat_model/weights/best.pt"
VIDEO_INPUT_PATH = os.path.join("assets", "video_testing4.mp4")
CONFIDENCE_THRESHOLD = 0.5

# === LOAD MODELS AND OCR ===
car_model = YOLO(CAR_MODEL_PATH)
plate_model = YOLO(PLATE_MODEL_PATH)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
car_model.to(device)
plate_model.to(device)

reader = easyocr.Reader(['en'], gpu=True)
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.25)

# === GLOBAL VARIABLES ===
vehicle_count = 0
counted_ids = set()
plate_history = {}
previous_positions = {}
track_labels = {}
class_history = {}
position_history = {}
track_quality = {}
CLASS_HISTORY_MAX = 10
CLASS_SWITCH_THRESHOLD = 3

label_colors = {
    'car': (0, 255, 0),
    'truck': (255, 0, 0),
    'bus': (0, 0, 255),
    'van': (255, 255, 0)
}

def preprocess_plate(cropped):
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return morph

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

def process_and_save():
    global vehicle_count, counted_ids, plate_history
    vehicle_count = 0
    counted_ids = set()
    plate_history = {}

    cap = cv2.VideoCapture(VIDEO_INPUT_PATH)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    LINE_POSITION = int(frame_height * 0.8)
    LINE_OFFSET = int(frame_height * 0.01)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video_filename = f"video_result_{timestamp}.mp4"
    VIDEO_OUTPUT_PATH = os.path.join("outputs", output_video_filename)
    LOG_OUTPUT_PATH = os.path.join("outputs", "logs.txt")

    os.makedirs("outputs", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))

    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1
        print(f"Saving: {(frame_number / total_frames) * 100:.2f}%", end="\r")

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

            plate_h = (y2 - y1) // 3
            plate_w = (x2 - x1) // 2
            margin = 5
            plate_x1 = max(x1 + (x2 - x1 - plate_w) // 2 - margin, 0)
            plate_y1 = max(y2 - plate_h - margin, 0)
            plate_x2 = min(plate_x1 + plate_w + margin * 2, frame.shape[1])
            plate_y2 = min(plate_y1 + plate_h + margin * 2, frame.shape[0])

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
                    if thumb_x >= x1 and thumb_y + thumb_h <= y2:
                        thumb_img = cv2.resize(enlarged, (thumb_w, thumb_h))
                        cv2.rectangle(display_frame, (thumb_x, thumb_y), (thumb_x + thumb_w, thumb_y + thumb_h), (0, 255, 255), 1)
                        h, w = thumb_img.shape[:2]
                        display_frame[thumb_y:thumb_y + h, thumb_x:thumb_x + w] = thumb_img

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

        cv2.line(display_frame, (0, LINE_POSITION), (frame_width, LINE_POSITION), (255, 255, 0), 2)
        cv2.putText(display_frame, f"Count: {vehicle_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
        cv2.putText(display_frame, f"Count: {vehicle_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

        out.write(display_frame)

    cap.release()
    out.release()

    with open(LOG_OUTPUT_PATH, "a") as log_file:
        log_file.write(f"\n=== Processed at {datetime.datetime.now()} ===\n")
        log_file.write(f"Processed Video: {VIDEO_OUTPUT_PATH}\n")
        log_file.write(f"Total Vehicles Counted: {vehicle_count}\n")
        log_file.write("Detected Plates:\n")
        for track_id, plates in plate_history.items():
            if plates:
                sorted_plates = sorted(plates, key=lambda x: x[1], reverse=True)
                best_plate = sorted_plates[0][0]
                log_file.write(f"  ID {track_id}: {best_plate}\n")
        log_file.write("="*40 + "\n")

    return VIDEO_OUTPUT_PATH, LOG_OUTPUT_PATH

def generate_frames():
    global vehicle_count, counted_ids, plate_history
    vehicle_count = 0
    counted_ids = set()
    plate_history = {}

    cap = cv2.VideoCapture(VIDEO_INPUT_PATH)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    LINE_POSITION = int(frame_height * 0.8)
    LINE_OFFSET = int(frame_height * 0.01)

    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1
        print(f"Streaming: {(frame_number / total_frames) * 100:.2f}%", end="\r")

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

            plate_h = (y2 - y1) // 3
            plate_w = (x2 - x1) // 2
            margin = 5
            plate_x1 = max(x1 + (x2 - x1 - plate_w) // 2 - margin, 0)
            plate_y1 = max(y2 - plate_h - margin, 0)
            plate_x2 = min(plate_x1 + plate_w + margin * 2, frame.shape[1])
            plate_y2 = min(plate_y1 + plate_h + margin * 2, frame.shape[0])

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
                    if thumb_x >= x1 and thumb_y + thumb_h <= y2:
                        thumb_img = cv2.resize(enlarged, (thumb_w, thumb_h))
                        cv2.rectangle(display_frame, (thumb_x, thumb_y), (thumb_x + thumb_w, thumb_y + thumb_h), (0, 255, 255), 1)
                        h, w = thumb_img.shape[:2]
                        display_frame[thumb_y:thumb_y + h, thumb_x:thumb_x + w] = thumb_img

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

        cv2.line(display_frame, (0, LINE_POSITION), (frame_width, LINE_POSITION), (255, 255, 0), 2)
        cv2.putText(display_frame, f"Count: {vehicle_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
        cv2.putText(display_frame, f"Count: {vehicle_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', display_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/save')
def save():
    def run_save():
        video_path, log_path = process_and_save()
        return video_path, log_path
    video_path, log_path = run_save()
    return render_template('save.html', video_path=video_path, log_path=log_path)

@app.route('/realtime')
def realtime():
    return render_template('realtime.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/download/<path:filename>')
def download_file(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)