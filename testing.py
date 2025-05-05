from ultralytics import YOLO
import cv2
import os
import easyocr
import matplotlib.pyplot as plt

# === CONFIG ===
MODEL_PATH = "vehicle_detection_project/vehicle_model/weights/best.pt"
VIDEO_INPUT_PATH = os.path.join("assets", "video_testing3.mp4")
CONFIDENCE_THRESHOLD = 0.25

# === LOAD MODEL AND OCR ===
model = YOLO(MODEL_PATH)
reader = easyocr.Reader(['en'])

# === OPEN VIDEO ===qqqqqqqqqqqqq
cap = cv2.VideoCapture(VIDEO_INPUT_PATH)

# === PROCESS AND DISPLAY FRAMES IN REALTIME ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=CONFIDENCE_THRESHOLD, save=False, verbose=False)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            # Crop the detected area for OCR
            cropped = frame[y1:y2, x1:x2]
            plate_texts = reader.readtext(cropped)

            box_color = (0, 255, 0)  # Default: green (even)
            plate_number = None

            if plate_texts:
                plate_number = plate_texts[0][1]  # Get the first OCR result text
                digits = ''.join(filter(str.isdigit, plate_number))
                if digits:
                    last_digit = int(digits[-1])
                    if last_digit % 2 == 0:
                        box_color = (0, 255, 0)  # Green for even
                        classification = "Even"
                    else:
                        box_color = (0, 0, 255)  # Red for odd
                        classification = "Odd"

                    # Add classification label
                    cv2.putText(frame, f"{plate_number} ({classification})", (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

            # Draw bounding box and object label
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    # === SHOW FRAME using Matplotlib ===
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Hide axes
    plt.show(block=False)
    plt.pause(0.001)  # Pause to allow the frame to update

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === CLEANUP ===
cap.release()
cv2.destroyAllWindows()
