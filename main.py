import os
import logging
import time
from datetime import datetime
from dotenv import load_dotenv
from modules.hikvision_client import HikvisionClient
from modules.yolov8_object_counter import YOLOv8_ObjectCounter

# Load environment variables from .env file
load_dotenv()

# Fetch the camera details from environment variables
cam_ip = os.getenv('CAM_IP')
cam_user = os.getenv('CAM_USER')
cam_password = os.getenv('CAM_PASSWORD')
cam_name = os.getenv('CAM_NAME')

# Fetch the storage volume path from environment variables
storage_volume_path = os.getenv('STORAGE_VOLUME_PATH')

# Ensure the storage directory exists
if not os.path.exists(storage_volume_path):
    os.makedirs(storage_volume_path)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == '__main__':
    cam = HikvisionClient(cam_ip, cam_user, cam_password)
    counter = YOLOv8_ObjectCounter(model_file='yolov8m.pt', conf=0.60, iou=0.60)

    current_date = datetime.now().strftime('%Y-%m-%d')
    output_file_path = os.path.join(storage_volume_path, f'{cam_name}-{current_date}.csv')

    counter.predict_video(cam, output_file_path, frame_skip=5, update_interval=2)
