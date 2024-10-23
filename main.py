import cv2
import threading
import queue
from visualizer import Visualizer
from camera_reader import CameraReader
from modelProcessor import YOLOModel
import time

def load_class_names(file_path):
    with open(file_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

def capture_frames(camera_reader, frame_queue):
    while True:
        frame = camera_reader.get_frame()
        if frame is not None:
            if not frame_queue.full():  # Avoid adding if queue is full
                frame_queue.put(frame)
        time.sleep(0.01)  # Small delay to prevent thread from hogging CPU

def process_frames(frame_queue, yolo_model, result_queue, skip_frames=2):
    frame_cnt = 0
    while True:
        try:
            frame = frame_queue.get(timeout=1)  # Timeout to avoid indefinite blocking
            frame_cnt += 1
            if frame_cnt % skip_frames == 0:
                resized_frame = cv2.resize(frame, (640, 640))  # Resize for YOLO
                results = yolo_model.process_frame(resized_frame)
                if not result_queue.full():  # Avoid adding if queue is full
                    result_queue.put((frame, results))
        except queue.Empty:
            continue  # If no frames are available, keep looping
        time.sleep(0.01)  # Small delay to avoid high CPU usage

def main():
    camera_reader = CameraReader()
    yolo_model = YOLOModel("yolov8n.pt")
    class_names = load_class_names('coco.names.txt')
    visualizer = Visualizer(class_names)

    # Queues with a max size to avoid too much backlog
    frame_queue = queue.Queue(maxsize=10)
    result_queue = queue.Queue(maxsize=10)

    camera_reader.start()

    # Start threads
    capture_thread = threading.Thread(target=capture_frames, args=(camera_reader, frame_queue), daemon=True)
    process_thread = threading.Thread(target=process_frames, args=(frame_queue, yolo_model, result_queue), daemon=True)

    capture_thread.start()
    process_thread.start()

    # Main thread for visualization
    while True:
        try:
            frame, results = result_queue.get(timeout=1)  # Timeout to avoid indefinite blocking
            visualized_frame = visualizer.visualize(frame, results)
            cv2.imshow("YOLO Detection", visualized_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except queue.Empty:
            continue  # If no results are available, keep looping

    camera_reader.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
