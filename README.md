# Vehiscan - Vehicle Detection and License Plate Recognition System
## lapis project final

![Vehiscan Demo] ![image](https://github.com/user-attachments/assets/0fa5a425-683e-4b47-bc41-34040e3328e2)


Vehiscan is an advanced vehicle detection and license plate recognition system that utilizes computer vision and deep learning to monitor traffic, count vehicles, and identify license plates in real-time or from uploaded videos.

## Features

- 🚗 **Vehicle Detection**: Identifies cars, trucks, buses, and vans using YOLO object detection
- 🔢 **Vehicle Counting**: Tracks vehicles across counting lines/zones with direction detection
- 🚘 **License Plate Recognition**: Extracts and reads license plates using OCR (Optical Character Recognition)
- 📊 **Statistics Collection**: Maintains counts by vehicle type and plates detected
- ☁️ **Cloud Integration**: Uploads processed videos and data to AWS S3
- 🎥 **Dual Processing Modes**:
  - Real-time video stream processing
  - Uploaded video file processing

## Technologies Used

- **Computer Vision**: OpenCV
- **Deep Learning**: 
  - YOLO (You Only Look Once) for object detection
  - SORT (Simple Online and Realtime Tracking) for vehicle tracking
- **OCR**: EasyOCR for license plate reading
- **GUI**: Flet (Python framework for building interactive UIs)
- **Cloud**: AWS S3 for data storage
- **Video Processing**: FFmpeg
