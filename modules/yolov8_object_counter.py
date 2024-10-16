import numpy as np
import cv2
import modules.sort as sort
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
import logging
import time
import os
from modules.yolov8_object_detector import YOLOv8_ObjectDetector
from modules.drive_upload import upload_to_drive

class YOLOv8_ObjectCounter(YOLOv8_ObjectDetector):
    def __init__(self, model_file='yolov8l.pt', labels=None, classes=[0, 1, 2, 3, 5, 7], conf=0.5, iou=0.5, track_max_age=45, track_min_hits=15, track_iou_threshold=0.3):
        super().__init__(model_file, labels, classes, conf, iou)
        self.labels = labels or {
            0: 'person',
            1: 'bicycle',
            2: 'car',
            3: 'motorbike',
            5: 'bus',
            7: 'truck'
        }
        self.coefficients = {  
            'person': 1,
            'bicycle': 1,
            'car': 2.5,
            'motorbike': 1.5,
            'bus': 25,
            'truck': 1.5
        }

        self.track_max_age = track_max_age
        self.track_min_hits = track_min_hits
        self.track_iou_threshold = track_iou_threshold

        
        self.counted_ids = defaultdict(set) 
        self.class_counts = defaultdict(int)

    def reset_tracker(self):
        """Reset the SORT tracker after each upload to avoid ID conflicts between intervals."""
        self.tracker = sort.Sort(
            max_age=self.track_max_age,
            min_hits=self.track_min_hits,
            iou_threshold=self.track_iou_threshold
        )

    def predict_video(self, frame_provider, output_file_path, frame_interval=0.5, verbose=True):
        self.reset_tracker()  
        logging.info("Starting video prediction...")

        last_upload_time = time.time()
        last_frame_time = time.time() - frame_interval

        def process_frame(frame):
            results = self.predict_img(frame, verbose=False)
            if results is None:
                logging.info("No results from YOLO model")
                return None, None

            detections = np.empty((0, 6))
            motorbike_detected_but_not_tracked = False
            for box in results.boxes:
                score = box.conf.item()
                class_id = int(box.cls.item())
                if class_id not in self.classes or score < self.conf:
                    continue
                x1, y1, x2, y2 = np.squeeze(box.xyxy.numpy()).astype(int)
                currentArray = np.array([x1, y1, x2, y2, score, class_id])
                detections = np.vstack((detections, currentArray))
                class_name = self.labels.get(class_id, f"Class {class_id}")
                logging.info(f"Detection: {class_name}, Score {score * 100:.2f}%, Box ({x1}, {y1}, {x2}, {y2})")

                if class_name == 'motorbike':
                    motorbike_detected_but_not_tracked = True

            tracked_objects = self.tracker.update(detections)
            logging.info(f"Tracked objects: {len(tracked_objects)}")

        
            now = datetime.now()
            rounded_minute = (now.minute // 2) * 2
            current_time_str = now.replace(minute=rounded_minute, second=0, microsecond=0).strftime("%H:%M")

        
            motorbike_tracked = False
            for trk in tracked_objects:
                x1, y1, x2, y2, track_id, class_id = trk.astype(int)
                class_name = self.labels.get(class_id, f"Class {class_id}")
                if class_name == 'motorbike':
                    motorbike_tracked = True

                
                if track_id not in self.counted_ids[current_time_str]:
                    self.counted_ids[current_time_str].add(track_id)
                    self.class_counts[class_name] += 1
                    logging.info(f"New object counted: ID {track_id}, Class {class_name}")

            if motorbike_detected_but_not_tracked and not motorbike_tracked:
                self.class_counts['motorbike'] += 1

            return results, tracked_objects

        while True:
            try:
                current_time = time.time()
                if current_time - last_frame_time >= frame_interval:
                    last_frame_time = current_time
                    frame = frame_provider.get_frame()
                    if frame is None:
                        logging.warning("Failed to capture frame. Retrying...")
                        time.sleep(1)
                        continue
                    logging.info("Frame captured successfully")

                    process_frame(frame)

                
                if current_time - last_upload_time >= 120:  
                    logging.info("Two minutes passed. Saving counts to Excel and uploading to Google Drive.")
                    self.save_count_to_csv(output_file_path)
                    last_upload_time = current_time

                
                    self.reset_tracker()  
                    self.counted_ids.clear()  

            except Exception as e:
                logging.error(f"Error processing frame: {e}")
                time.sleep(1)
                continue

        logging.info("Final save of counts to Excel")
        self.save_count_to_csv(output_file_path)
        logging.info(f"Total processing time: {time.time() - last_upload_time} seconds")

    def save_count_to_csv(self, output_file_path):
        aggregated_counts = self.class_counts
        self.save_aggregated_data_to_excel(aggregated_counts, output_file_path)
        logging.info(f"Data written and saved to {output_file_path}")

        google_drive_folder_id = os.getenv('GOOGLE_DRIVE_FOLDER_ID')
        if google_drive_folder_id:
            upload_to_drive(output_file_path, google_drive_folder_id)
            logging.info(f"Data uploaded to Google Drive: {output_file_path}")
        else:
            logging.warning("Google Drive folder ID not found. Skipping upload.")

        
        self.counted_ids.clear()
        self.class_counts.clear()

    def save_aggregated_data_to_excel(self, aggregated_counts, output_file_path):
        rows = defaultdict(lambda: defaultdict(int))
        City = os.getenv('CITY')
        code_panel = os.getenv('CODE_PANEL')

        now = datetime.now()

        for class_name, count in aggregated_counts.items():
            rounded_minute = (now.minute // 2) * 2
            hour_minute_str = now.replace(minute=rounded_minute).strftime("%H:%M")
            key = (City, code_panel, class_name, now.date(), hour_minute_str)
            
            rows[key]['count'] += count
            coefficient = self.coefficients.get(class_name, 1)
            rows[key]['audience'] = rows[key]['count'] * coefficient

        data = [
            {
                "City": key[0],
                "Panel": key[1],
                "Type": key[2],
                "Date": key[3].strftime('%Y-%m-%d'),
                "Time": key[4],
                "Count": value['count'],
                "Audience": value['audience']
            }
            for key, value in rows.items()
        ]
        df = pd.DataFrame(data)

        if df.empty:
            logging.warning("DataFrame is empty. No data to save.")
            return

        df = df.sort_values(by=['Date', 'Time'])

        if not os.path.exists(output_file_path):
            logging.info(f"Creating file: {output_file_path}")
            with pd.ExcelWriter(output_file_path, engine='openpyxl', mode='w') as writer:
                df.to_excel(writer, sheet_name=now.strftime("%B %Y"), index=False)
        else:
            with pd.ExcelWriter(output_file_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                month_name = now.strftime("%B %Y")
                if month_name in writer.book.sheetnames:
                    existing_df = pd.read_excel(output_file_path, sheet_name=month_name)
                    df = pd.concat([existing_df, df]).groupby(["City", "Panel", "Type", "Date", "Time"], as_index=False).sum()
                    df = df.sort_values(by=['Date', 'Time'])
                    df.to_excel(writer, sheet_name=month_name, index=False)
                else:
                    df.to_excel(writer, sheet_name=month_name, index=False)

        logging.info(f"Data written and saved to {output_file_path}")
