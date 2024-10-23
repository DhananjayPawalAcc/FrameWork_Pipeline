# visualizer.py
import cv2

class Visualizer:
    def __init__(self, class_names):
        self.class_names = class_names

    def visualize(self, frame, results):
        for class_id, confidence, bbox in results:
            x, y, w, h = bbox
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else "Unknown"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame, f"{class_name}: {confidence:.2f}", 
                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )
        return frame
