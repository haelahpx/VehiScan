from ultralytics import YOLO
import cv2
import pytesseract

# Load two different models
vehicle_model = YOLO("models/vehicle_model(detrac-16)/weights/best.pt")
plate_model = YOLO("models/plat_model/weights/best.pt")

cap = cv2.VideoCapture("assets/testing_video2.mp4")

vehicle_count = 0
counted_centers = []
line_y = 500
offset = 10

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Vehicle Detection
    vehicle_results = vehicle_model(frame)[0]

    # Plate Detection
    plate_results = plate_model(frame)[0]

    # Draw counting line
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 255), 2)

    # Process vehicle detections
    for box in vehicle_results.boxes:
        cls = int(box.cls[0])
        label = vehicle_model.names[cls]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        if label == "car":
            if (line_y - offset) < center_y < (line_y + offset):
                if not any(abs(center_x - cx) < 30 and abs(center_y - cy) < 30 for cx, cy in counted_centers):
                    vehicle_count += 1
                    counted_centers.append((center_x, center_y))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, "Car", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Process license plate detections
    for box in plate_results.boxes:
        cls = int(box.cls[0])
        label = plate_model.names[cls]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if label == "plate_license":
            plate_roi = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

            text = pytesseract.image_to_string(thresh, config='--psm 6')
            text = ''.join(filter(str.isalnum, text))

            digits = [char for char in text if char.isdigit()]
            if digits:
                last_digit = int(digits[-1])
                odd_even = "Even" if last_digit % 2 == 0 else "Odd"
                color = (0, 255, 0) if odd_even == "Even" else (0, 0, 255)
            else:
                odd_even = "N/A"
                color = (255, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{text} - {odd_even}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Show vehicle count
    cv2.putText(frame, f"Vehicle Count: {vehicle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Car & Plate Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
