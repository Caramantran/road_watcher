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
from modules.drive_upload import upload_to_drive

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class YOLOv8_ObjectCounter(YOLOv8_ObjectDetector):
    def __init__(self, model_file='yolov8m.pt', labels=None, classes=[0, 1, 2, 3, 5, 7], conf=0.50, iou=0.45, track_max_age=45, track_min_hits=15, track_iou_threshold=0.3):
        super().__init__(model_file, labels, classes, conf, iou)
        self.labels = labels or {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        self.track_max_age = track_max_age
        self.track_min_hits = track_min_hits
        self.track_iou_threshold = track_iou_threshold

    def predict_video(self, frame_provider, output_file_path, frame_skip=2, update_interval=5, verbose=True):
        frame_count = 0
        tracker = sort.Sort(max_age=self.track_max_age, min_hits=self.track_min_hits, iou_threshold=self.track_iou_threshold)
        start_time = time.time()
        logging.info("Starting video prediction...")

        def process_frame(frame):
            frame_resized = cv2.resize(frame, (640, 360))  # Resize frame for faster processing
            results = self.predict_img(frame_resized, verbose=False)
            if results is None:
                logging.info("No results from YOLO model")
                return None, None

            detections = np.empty((0, 5))
            for box in results.boxes:
                score = box.conf.item() * 100
                class_id = int(box.cls.item())
                if class_id not in self.classes:
                    continue  # Skip classes not of interest
                x1, y1, x2, y2 = np.squeeze(box.xyxy.numpy()).astype(int)
                currentArray = np.array([x1, y1, x2, y2, score])
                detections = np.vstack((detections, currentArray))
                logging.info(f"Detection: Class {class_id}, Score {score}, Box ({x1}, {y1}, {x2}, {y2})")
                with open('detections.txt', 'a') as f:
                    f.write(f"{datetime.now()}, Class {class_id}, Score {score}, Box ({x1}, {y1}, {x2}, {y2})\n")

            resultsTracker = tracker.update(detections)
            logging.info(f"Tracked objects: {len(resultsTracker)}")

            return results, resultsTracker

        while True:
            try:
                frame = frame_provider.get_frame()
                if frame is None:
                    logging.warning("Failed to capture frame. Retrying...")
                    time.sleep(1)
                    continue
                logging.info("Frame captured successfully")
                
                if frame_count % frame_skip == 0:
                    with ThreadPoolExecutor() as executor:
                        results, resultsTracker = executor.submit(process_frame, frame).result()
                        if results is None or resultsTracker is None:
                            continue

                if frame_count % (update_interval * 30) == 0:  # Assuming 30 FPS
                    logging.info("Saving counts to sheets")
                    self.save_count_to_csv(output_file_path)

                frame_count += 1

            except Exception as e:
                logging.error(f"Error processing frame: {e}")
                time.sleep(1)
                continue

            logging.info("Final save of counts to sheets")
            self.save_count_to_csv(output_file_path)
            logging.info(f"Total processing time: {time.time() - start_time} seconds")

    def save_count_to_csv(self, output_file_path):
        aggregated_counts = self.read_and_format_detections('detections.txt')
        self.save_aggregated_data_to_excel(aggregated_counts, output_file_path, self.labels)
        logging.info(f"Data written and saved to {output_file_path}")

        # Upload to Google Drive
        google_drive_folder_id = os.getenv('GOOGLE_DRIVE_FOLDER_ID')
        if google_drive_folder_id:
            upload_to_drive(output_file_path, google_drive_folder_id)
            logging.info(f"Data uploaded to Google Drive: {output_file_path}")
        else:
            logging.warning("Google Drive folder ID not found. Skipping upload.")

    @staticmethod
    def read_and_format_detections(file_path):
        # Créer un dictionnaire pour stocker les comptes agrégés
        data = defaultdict(lambda: defaultdict(int))
        city = os.getenv('CITY')
        code_panel = os.getenv('CODE_PANEL')


        # Lire le fichier detections.txt
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split(', ')
                if len(parts) < 4:
                    continue

                dt_str = parts[0]
                cls_str = parts[1]
                class_id = int(cls_str.split(' ')[-1])

                # Convertir la chaîne de caractères en datetime
                dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S.%f')
                # Agréger par minute
                minute = dt.replace(second=0, microsecond=0)

                # Ajouter au dictionnaire
                data[minute][class_id] += 1

        return data

 #   @staticmethod
  #  def save_aggregated_data_to_excel(aggregated_data, output_file_path, labels):
   #     # Créer une liste pour stocker les lignes du DataFrame
    #    rows = []
     #   city = os.getenv('CITY')
      #  code_panel = os.getenv('CODE_PANEL')

       # for minute, class_data in aggregated_data.items():
        #    for cls, count in class_data.items():
         #       row = {
          #          "city": city,
           #         "Source": code_panel,
            #        "Class": labels.get(cls, f"Class {cls}"),
             #       "Date": minute.date().strftime('%Y-%m-%d'),
              #      "Time": minute.time().strftime('%H:%M'),
               #     "Count": count
             #   }
                #rows.append(row)

        # Convertir la liste en DataFrame
        #df = pd.DataFrame(rows)

        # Écrire le DataFrame dans un fichier Excel
        #with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
         #   for date, date_df in df.groupby('Date'):
         #      date_df.to_excel(writer, sheet_name=date, index=False)

        #logging.info(f"Data written and saved to {output_file_path}")  

    @staticmethod
    def save_aggregated_data_to_excel(aggregated_data, output_file_path, labels):
        """
        Saves aggregated detection counts to an Excel file, one row per hour for each class.
        """
        # Dictionary to store aggregated rows
        rows = defaultdict(lambda: defaultdict(int))  # Use a defaultdict to accumulate counts
        city = os.getenv('CITY')
        code_panel = os.getenv('CODE_PANEL')

        for minute, class_data in aggregated_data.items():
            # Use only the hour part (e.g., '17h')
            hour_str = f"{minute.hour}h"
            for cls, count in class_data.items():
                key = (city, code_panel, labels.get(cls, f"Class {cls}"), minute.date(), hour_str)
                rows[key]['count'] += count


        # Convert rows to a DataFrame
        data = [
            {"city": key[0], "Source": key[1], "Class": key[2], "Date": key[3].strftime('%Y-%m-%d'), "Time": key[4], "Count": value['count']}
            for key, value in rows.items()
    ]
        df = pd.DataFrame(data)

        # Write the DataFrame to an Excel file, each date in a separate sheet
        with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
            for date, date_df in df.groupby('Date'):
                date_df.to_excel(writer, sheet_name=date, index=False)

        logging.info(f"Data written and saved to {output_file_path}")