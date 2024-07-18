import numpy as np
import cv2
import sort
import pandas as pd
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO
import yt_dlp
import time
import logging
import urllib.request
import os
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



class YOLOv8_ObjectDetector:
    def __init__(self, model_file='yolov8m.pt', labels=None, classes=None, conf=0.25, iou=0.45):
        self.classes = classes
        self.conf = conf
        self.iou = iou
        self.model = YOLO(model_file)
        self.model_name = model_file.split('.')[0]
        self.results = None

        if labels is None:
            self.labels = self.model.names
        else:
            self.labels = labels

    def predict_img(self, img, verbose=True):
        results = self.model(img, classes=self.classes, conf=self.conf, iou=self.iou, verbose=verbose)
        self.orig_img = img
        self.results = results[0]
        return results[0]

class YOLOv8_ObjectCounter(YOLOv8_ObjectDetector):
    def __init__(self, model_file='yolov8m.pt', labels=None, classes=[0, 1, 2, 3, 5, 7], conf=0.60, iou=0.45, track_max_age=45, track_min_hits=15, track_iou_threshold=0.3):
        super().__init__(model_file, labels, classes, conf, iou)
        self.track_max_age = track_max_age
        self.track_min_hits = track_min_hits
        self.track_iou_threshold = track_iou_threshold
        self.class_counts = defaultdict(lambda: defaultdict(set))

    def predict_video(self, video_source, output_file_path, frame_skip=5, update_interval=10, verbose=True):
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            logging.error("Error opening video stream or file")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        tracker = sort.Sort(max_age=self.track_max_age, min_hits=self.track_min_hits, iou_threshold=self.track_iou_threshold)
        totalCount = set()
        start_time = time.time()

        def process_frame(frame):
            frame_resized = cv2.resize(frame, (640, 360))  # Resize frame for faster processing
            results = self.predict_img(frame_resized, verbose=False)
            if results is None:
                return None, None

            detections = np.empty((0, 5))
            for box in results.boxes:
                score = box.conf.item() * 100
                class_id = int(box.cls.item())
                x1, y1, x2, y2 = np.squeeze(box.xyxy.numpy()).astype(int)
                currentArray = np.array([x1, y1, x2, y2, score])
                detections = np.vstack((detections, currentArray))

            resultsTracker = tracker.update(detections)
            for result in resultsTracker:
                x1, y1, x2, y2, id = result
                x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
                if id not in totalCount:
                    totalCount.add(id)
                    for box in results.boxes:
                        if int(box.cls.item()) == class_id:
                            self.class_counts[hour][int(box.cls.item())].add(id)
            return results, resultsTracker

        while cap.isOpened():
            try:
                ret, frame = cap.read()
                if not ret:
                    logging.warning("Failed to capture frame. Retrying...")
                    time.sleep(1)
                    continue

                current_time = datetime.now()
                hour = current_time.replace(second=0, microsecond=0)

                if frame_count % frame_skip == 0:
                    with ThreadPoolExecutor() as executor:
                        results, resultsTracker = executor.submit(process_frame, frame).result()
                        if results is None or resultsTracker is None:
                            continue

                if frame_count % (update_interval * fps) == 0:
                    self.save_count_to_csv(output_file_path)

                frame_count += 1

            except Exception as e:
                logging.error(f"Error processing frame: {e}")
                time.sleep(1)
                continue

        cap.release()
        self.save_count_to_csv(output_file_path)
        logging.info(f"Total processing time: {time.time() - start_time} seconds")

    def save_count_to_csv(self, output_file_path):
        rows = []
        for hour, class_data in self.class_counts.items():
            hour_str = hour.strftime('%Y-%m-%d %H:%M')
            for cls, ids in class_data.items():
                rows.append({
                    "Timestamp": hour_str,
                    "Classe": self.labels[cls],
                    "Count": len(ids)
                })

        if not rows:
            logging.info("No data to save.")
            return

        df = pd.DataFrame(rows)
        if df.empty:
            logging.info("DataFrame is empty. No data to save.")
            return

        df.to_csv(output_file_path, index=False)
        logging.info(f"Rows: {rows}")
        logging.info(f"DataFrame:\n{df}")
        logging.info(f"Data written and saved to {output_file_path}")

    def print_counts(self, total_count):
        logging.info(f'Total count of detected objects: {len(total_count)}')
        for hour, class_counts in self.class_counts.items():
            logging.info(f'Hour: {hour}')
            for cls, ids in class_counts.items():
                logging.info(f'  Class {self.labels[cls]}: {len(ids)} objects')

def get_youtube_stream_url(youtube_url):
    ydl_opts = {'format': 'best'}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info['url']

# Example usage
if __name__ == '__main__':
    from datetime import datetime
    import os

  

    youtube_url = 'https://www.youtube.com/live/1fiF7B6VkCk?si=IK09UXRxvUqT9XV-'
    stream_url = get_youtube_stream_url(youtube_url)

    counter = YOLOv8_ObjectCounter(model_file='yolov8m.pt', conf=0.60, iou=0.60)

    current_date = datetime.now().strftime('%Y-%m-%d')
    output_file_path = f'/content/test-{current_date}.csv'

    counter.predict_video(stream_url, output_file_path, frame_skip=5, update_interval=2)