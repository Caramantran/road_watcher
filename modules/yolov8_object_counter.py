import numpy as np
import cv2
import modules.sort as sort
import pandas as pd
from datetime import datetime
from collections import defaultdict
import logging
from concurrent.futures import ThreadPoolExecutor
import time
import os
from modules.yolov8_object_detector import YOLOv8_ObjectDetector
from modules.drive_upload import upload_to_drive, append_data_to_existing_file

class YOLOv8_ObjectCounter(YOLOv8_ObjectDetector):
    def __init__(self, model_file='yolov8m.pt', labels=None, classes=[0, 1, 2, 3, 5, 7], conf=0.60, iou=0.45, track_max_age=45, track_min_hits=15, track_iou_threshold=0.3):
        super().__init__(model_file, labels, classes, conf, iou)
        self.track_max_age = track_max_age
        self.track_min_hits = track_min_hits
        self.track_iou_threshold = track_iou_threshold
        self.class_counts = defaultdict(lambda: defaultdict(set))

    def predict_video(self, frame_provider, output_file_path, frame_skip=2, update_interval=2, verbose=True):
        frame_count = 0
        tracker = sort.Sort(max_age=self.track_max_age, min_hits=self.track_min_hits, iou_threshold=self.track_iou_threshold)
        totalCount = set()
        start_time = time.time()
        logging.info("Starting video prediction...")

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

        while True:
            try:
                frame = frame_provider.get_frame()
                if frame is None:
                    logging.warning("Failed to capture frame. Retrying...")
                    time.sleep(1)
                    continue

                current_time = datetime.now()
                hour = current_time.replace(second=0, microsecond=0)

                if frame_count % frame_skip == 0:
                    with ThreadPoolExecutor() as executor:
                        results, resultsTracker = executor.submit(process_frame, frame).result()
                        if results is None or resultsTracker is None:
                            logging.info(f"No results for frame at {current_time}")
                            continue

                if frame_count % (update_interval * 30) == 0:  # Assuming 30 FPS
                    logging.info("Saving counts to CSV")
                    self.save_count_to_csv(output_file_path)

                frame_count += 1
                logging.info(f"Processed frame {frame_count} at {current_time}")

            except Exception as e:
                logging.error(f"Error processing frame: {e}")
                time.sleep(1)
                continue

        logging.info("Final save of counts to CSV")
        self.save_count_to_csv(output_file_path)
        logging.info(f"Total processing time: {time.time() - start_time} seconds")

    def save_count_to_csv(self, output_file_path):
        rows = []
        for hour, class_data in self.class_counts.items():
            hour_str = hour.strftime('%Y-%m-%d %H:%M')
            for cls, ids in class_data.items():
                rows.append({
                    "Date": hour.date().strftime('%Y-%m-%d'),
                    "Time": hour.time().strftime('%H:%M:%S'),
                    "Class": self.labels[cls],
                    "Count": len(ids)
                })

        if not rows:
            logging.info("No data to save.")
            return

        # Append the new data to the existing file
        append_data_to_existing_file(output_file_path, rows)
        logging.info(f"Data written and saved to {output_file_path}")

        # Print the new data to console for verification
        print(pd.DataFrame(rows))

        # Upload to Google Drive
        google_drive_folder_id = os.getenv('GOOGLE_DRIVE_FOLDER_ID')
        if google_drive_folder_id:
            upload_to_drive(output_file_path, google_drive_folder_id)
            logging.info(f"Data uploaded to Google Drive: {output_file_path}")
        else:
            logging.warning("Google Drive folder ID not found. Skipping upload.")

def append_data_to_existing_file(existing_file_path, new_data):
    # Load the existing data
    if os.path.exists(existing_file_path):
        existing_df = pd.read_csv(existing_file_path)
    else:
        existing_df = pd.DataFrame()

    # Convert new data to DataFrame
    new_df = pd.DataFrame(new_data)

    # Remove duplicate rows
    combined_df = pd.concat([existing_df, new_df]).drop_duplicates()

    # Save the combined data back to the file
    combined_df.to_csv(existing_file_path, index=False)

    # Print the combined data to console for verification
    print(combined_df)
