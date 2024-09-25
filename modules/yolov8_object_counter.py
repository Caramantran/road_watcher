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
    def __init__(self, model_file='yolov8l.pt', labels=None, classes=[0, 1, 2, 3, 5, 7], conf=0.45, iou=0.5, track_max_age=45, track_min_hits=1, track_iou_threshold=0.4):
        super().__init__(model_file, labels, classes, conf, iou)
        self.labels = labels or {
            0: 'person',
            1: 'bicycle',
            2: 'car',
            3: 'motorbike',
            5: 'bus',
            7: 'truck'
        }
        self.coefficients = {  # Coefficients based on class
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

        # Initialize counted IDs
        self.counted_ids = set()
        # Initialize a dictionary to keep counts per class
        self.class_counts = defaultdict(int)

    def predict_video(self, frame_provider, output_file_path, frame_interval=5.0, verbose=True):
        tracker = sort.Sort(
            max_age=self.track_max_age,
            min_hits=self.track_min_hits,
            iou_threshold=self.track_iou_threshold
        )
        logging.info("Starting video prediction...")

        # Initialize last_frame_time
        last_frame_time = time.time() - frame_interval  # Initialize to capture first frame immediately

        def process_frame(frame):
            frame_resized = cv2.resize(frame, (640, 360))  # Resize frame for faster processing
            results = self.predict_img(frame_resized, verbose=False)
            if results is None:
                logging.info("No results from YOLO model")
                return None, None

            detections = np.empty((0, 6))  # Now includes class_id
            for box in results.boxes:
                score = box.conf.item()  # Confidence score between 0 and 1
                class_id = int(box.cls.item())
                if class_id not in self.classes or score < self.conf:
                    continue  # Skip classes not of interest or low confidence
                x1, y1, x2, y2 = np.squeeze(box.xyxy.numpy()).astype(int)
                currentArray = np.array([x1, y1, x2, y2, score, class_id])
                detections = np.vstack((detections, currentArray))
                class_name = self.labels.get(class_id, f"Class {class_id}")
                logging.info(f"Detection: {class_name}, Score {score * 100:.2f}%, Box ({x1}, {y1}, {x2}, {y2})")

            # Update tracker with detections
            tracked_objects = tracker.update(detections)
            logging.info(f"Tracked objects: {len(tracked_objects)}")

            # Process tracked objects or fallback to detection counts
            if len(tracked_objects) == 0:  # Fallback if tracking fails
                for box in results.boxes:
                    class_id = int(box.cls.item())
                    class_name = self.labels.get(class_id, f"Class {class_id}")
                    self.class_counts[class_name] += 1  # Count the detection even if tracking fails
                    logging.info(f"New object detected (no tracking): Class {class_name}")
            else:
                # Process tracked objects
                for trk in tracked_objects:
                    x1, y1, x2, y2, track_id, class_id = trk.astype(int)
                    class_name = self.labels.get(class_id, f"Class {class_id}")
                    if track_id not in self.counted_ids:
                        self.counted_ids.add(track_id)
                        self.class_counts[class_name] += 1
                        logging.info(f"New object counted: ID {track_id}, Class {class_name}")

            return results, tracked_objects

        def get_time_until_next_hour():
            """ Returns the number of seconds until the next full hour. """
            now = datetime.now()
            next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
            return (next_hour - now).total_seconds()

        while True:
            try:
                current_time = time.time()

                # Process frames at the defined frame interval
                if current_time - last_frame_time >= frame_interval:
                    last_frame_time = current_time

                    frame = frame_provider.get_frame()
                    if frame is None:
                        logging.warning("Failed to capture frame. Retrying...")
                        time.sleep(1)
                        continue
                    logging.info("Frame captured successfully")

                    # Process the frame
                    process_frame(frame)

                else:
                    time.sleep(1)  # Sleep briefly to prevent high CPU usage

                # Calculate time remaining until the next full hour
                time_until_next_hour = get_time_until_next_hour()

                # If time_until_next_hour is less than frame_interval, save and upload at the start of the next hour
                if time_until_next_hour <= frame_interval:
                    logging.info(f"{datetime.now().strftime('%H:%M')} - Saving counts for the past hour and uploading data.")
                    
                    # Save the counts for the previous hour
                    self.save_count_to_csv(output_file_path)
                    
                    # Sleep until the next hour
                    time.sleep(time_until_next_hour)

            except Exception as e:
                logging.error(f"Error processing frame: {e}")
                time.sleep(2)
                continue

        # Final save if needed (this part will never be reached due to the infinite loop)
        logging.info("Final save of counts to excel")
        self.save_count_to_csv(output_file_path)
        logging.info(f"Total processing time: {time.time() - start_time} seconds")

    def save_count_to_csv(self, output_file_path):
        # Use self.class_counts instead of reading detections.txt
        aggregated_counts = self.class_counts
        self.save_aggregated_data_to_excel(aggregated_counts, output_file_path)
        logging.info(f"Data written and saved to {output_file_path}")

        # Upload to Google Drive
        google_drive_folder_id = os.getenv('GOOGLE_DRIVE_FOLDER_ID')
        if google_drive_folder_id:
            upload_to_drive(output_file_path, google_drive_folder_id)
            logging.info(f"Data uploaded to Google Drive: {output_file_path}")
        else:
            logging.warning("Google Drive folder ID not found. Skipping upload.")

        # Clear counted IDs and counts after saving
        self.counted_ids.clear()
        self.class_counts.clear()

#################################################################################################
    
    @staticmethod
    def read_and_format_detections(file_path):
        # Create a dictionary to store aggregated counts
        data = defaultdict(lambda: defaultdict(int))
        # Read the detections.txt file
        if not os.path.exists(file_path):
            logging.warning(f"{file_path} does not exist.")
            return data
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split(', ')
                if len(parts) < 4:
                    continue
                dt_str = parts[0]
                cls_str = parts[1]
                class_name = cls_str.split(' ')[-1]
                # Convert string to datetime
                dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S.%f')
                # Aggregate by hour
                hour = dt.replace(minute=0, second=0, microsecond=0)
                # Add to dictionary
                data[hour][class_name] += 1

        return data
################################################################################################# 


    def save_aggregated_data_to_excel(self, aggregated_counts, output_file_path):
        """
        Saves aggregated detection counts to an Excel file, one row per hour for each class.
        """
        # Dictionary to store aggregated rows
        rows = defaultdict(lambda: defaultdict(int))  # Use a defaultdict to accumulate counts
        City = os.getenv('CITY')
        code_panel = os.getenv('CODE_PANEL')
        now = datetime.now()
        hour = now.replace(minute=0, second=0, microsecond=0)
        hour_str = f"{hour.hour}h"

        for class_name, count in aggregated_counts.items():
            key = (City, code_panel, class_name, hour.date(), hour_str)
            rows[key]['count'] += count
            coefficient = self.coefficients.get(class_name, 1)  # Get the coefficient based on class
            rows[key]['audience'] = rows[key]['count'] * coefficient  # Calculate Audience

        # Convert rows to a DataFrame
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
        
        # Ensure file exists, create it if not
        if not os.path.exists(output_file_path):
            logging.info(f"Creating file: {output_file_path}")
            with pd.ExcelWriter(output_file_path, engine='openpyxl', mode='w') as writer:
                df.to_excel(writer, sheet_name=now.strftime("%B %Y"), index=False)
        else:
            # Write the DataFrame to an Excel file, each month in a separate sheet
            with pd.ExcelWriter(output_file_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                month_name = now.strftime("%B %Y")
                if month_name in writer.book.sheetnames:
                    existing_df = pd.read_excel(output_file_path, sheet_name=month_name)
                    df = df.groupby(["City", "Panel", "Type", "Date", "Time"], as_index=False).sum()  # Aggregate counts by hour
                    existing_df = pd.concat([existing_df, df]).groupby(["City", "Panel", "Type", "Date", "Time"], as_index=False).sum()
                    existing_df.to_excel(writer, sheet_name=month_name, index=False)
                else:
                    df.to_excel(writer, sheet_name=month_name, index=False)

        logging.info(f"Data written and saved to {output_file_path}")
