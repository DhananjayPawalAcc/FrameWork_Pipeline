# yoloModel.py
import cv2
from ultralytics import YOLO

class YOLOModel:
    def __init__(self, model_path):
        # Load the YOLO model
        self.model = YOLO(model_path)

    def process_frame(self, frame):
        # Perform inference
        results = self.model(frame)

        # Extract results
        detections = []
        for result in results:
            for detection in result.boxes:
                # Get bounding box coordinates and confidence
                x1, y1, x2, y2 = detection.xyxy[0]  # Extract box coordinates
                confidence = detection.conf[0].item()  # Confidence score
                class_id = int(detection.cls[0])  # Class ID
                
                detections.append((class_id, confidence, (int(x1), int(y1), int(x2 - x1), int(y2 - y1))))

        return detections
