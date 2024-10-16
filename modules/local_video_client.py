import cv2
import logging

class LocalVideoClient:
    def __init__(self, video_path):
        self.video_path = video_path
        self.video_stream = None

    def get_frame(self):
        if self.video_stream is None:
            self.video_stream = cv2.VideoCapture(self.video_path)
            if not self.video_stream.isOpened():
                logging.error(f"Failed to open video file: {self.video_path}")
                return None

        ret, frame = self.video_stream.read()
        if not ret:
            logging.error("Failed to retrieve frame from video or end of video reached.")
            return None

        return frame

    def release(self):
        if self.video_stream is not None:
            self.video_stream.release()
