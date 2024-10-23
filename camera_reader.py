# camera_reader.py
import cv2
import threading
import queue

class CameraReader:
    def __init__(self, source=0):
        self.capture = cv2.VideoCapture(source)
        self.frames = queue.Queue()
        self.running = True

    def start(self):
        threading.Thread(target=self.read_frames, daemon=True).start()

    def read_frames(self):
        while self.running:
            ret, frame = self.capture.read()
            if ret:
                self.frames.put(frame)

    def get_frame(self):
        return self.frames.get() if not self.frames.empty() else None

    def stop(self):
        self.running = False
        self.capture.release()
