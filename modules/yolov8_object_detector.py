import numpy as np
from ultralytics import YOLO
import logging

class YOLOv8_ObjectDetector:
    def __init__(self, model_file='yolov8l.pt', labels=None, classes=None, conf=0.4, iou=0.45):
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

        logging.info(f"YOLOv8 model '{self.model_name}' initialized with confidence threshold {self.conf} and IOU threshold {self.iou}.")

    def predict_img(self, img, verbose=True):
        logging.info("Starting image prediction...")
        try:
            results = self.model(img, classes=self.classes, conf=self.conf, iou=self.iou, verbose=verbose)
            self.orig_img = img
            self.results = results[0]
            logging.info("Image prediction completed successfully.")
            return results[0]
        except Exception as e:
            logging.error(f"Error during image prediction: {e}")
            return None

