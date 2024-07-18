import numpy as np
import cv2
import sort
import pandas as pd
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO

class YOLOv8_ObjectDetector:
    def __init__(self, model_file='yolov8n.pt', labels=None, classes=None, conf=0.25, iou=0.45):
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
    def __init__(self,
                model_file='yolov8s.pt',
                labels=None,
                classes=[0, 1, 2, 3, 5, 7],
                conf=0.60,
                iou=0.45,
                track_max_age=45,
                track_min_hits=15,
                track_iou_threshold=0.3):

        super().__init__(model_file, labels, classes, conf, iou)
        self.track_max_age = track_max_age
        self.track_min_hits = track_min_hits
        self.track_iou_threshold = track_iou_threshold
        self.class_counts = defaultdict(lambda: defaultdict(set))  # Initialize nested dictionary to store unique IDs by hour and class

    def predict_video(self, video_source, output_file_path, frame_skip=1, update_interval=30, verbose=True):
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print("Error opening video stream or file")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second of the video
        frame_count = 0
        start_time = datetime.now()  # Record the start time of video processing

        tracker = sort.Sort(max_age=self.track_max_age, min_hits=self.track_min_hits, iou_threshold=self.track_iou_threshold)
        totalCount = set()  # Using a set to track unique object IDs

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print("Failed to capture frame. Exiting...")
                break

            # Calculate the timestamp for the current frame
            current_time = start_time + pd.to_timedelta(frame_count / fps, unit='s')
            hour = current_time.replace(minute=0, second=0, microsecond=0)

            # Skip frames based on frame_skip value
            if frame_count % frame_skip == 0:
                results = self.predict_img(frame, verbose=False)
                if results is None:
                    continue

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
                        # Update class counts for unique ID only once
                        for box in results.boxes:
                            if int(box.cls.item()) == class_id:
                                self.class_counts[hour][int(box.cls.item())].add(id)

            # Periodically save the results to CSV
            if frame_count % (update_interval * fps) == 0:
                self.save_count_to_csv(output_file_path)

            frame_count += 1

        cap.release()
        self.save_count_to_csv(output_file_path)  # Final save at the end of video processing

        if verbose:
            self.print_counts(totalCount)

    def save_count_to_csv(self, output_file_path):
        # Convert class_counts to DataFrame
        rows = []
        for hour, class_data in self.class_counts.items():
            hour_str = hour.strftime('%Hh')  # Extract hour as string formatted as 'Hh'
            for cls, ids in class_data.items():
                rows.append({
                    "Hour": hour_str,
                    "Classe": self.labels[cls],
                    "Count": len(ids)
                })

        df = pd.DataFrame(rows)

        # Pivot the DataFrame to have hours as columns and classes as rows
        df_pivot = df.pivot(index='Classe', columns='Hour', values='Count').fillna(0).reset_index()

        # Add 'Classe / Hour' to the index column name
        df_pivot.columns.name = None
        df_pivot.rename(columns={'Classe': 'Classe / Hour'}, inplace=True)

        # Write data to CSV
        df_pivot.to_csv(output_file_path, index=False)
        print(f"Data written and saved to {output_file_path}")

    def print_counts(self, total_count):
        print(f'Total count of detected objects: {len(total_count)}')
        for hour, class_counts in self.class_counts.items():
            print(f'Hour: {hour}')
            for cls, ids in class_counts.items():
                print(f'  Class {self.labels[cls]}: {len(ids)} objects')

