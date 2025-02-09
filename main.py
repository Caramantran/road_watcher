import os
import logging
import time
from datetime import datetime
from dotenv import load_dotenv
from modules.hikvision_client import HikvisionClient
from modules.yolov8_object_counter import YOLOv8_ObjectCounter
from modules.drive_upload import upload_to_drive


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

# Retrieve camera details from environment variables
cam_ip = os.getenv('CAM_IP')
cam_user = os.getenv('CAM_USER')
cam_password = os.getenv('CAM_PASSWORD')
code_panel = os.getenv('CODE_PANEL')

# Retrieve storage volume path from environment variables
storage_volume_path = os.getenv('STORAGE_VOLUME_PATH')
city = os.getenv('CITY')

# Ensure storage directory exists
if not os.path.exists(storage_volume_path):
    os.makedirs(storage_volume_path)
    logging.info(f'Répertoire créé : {storage_volume_path}')
else:
    logging.info(f'Répertoire déjà existant : {storage_volume_path}')

if __name__ == '__main__':
    cam = HikvisionClient(cam_ip, cam_user, cam_password)

    counter = YOLOv8_ObjectCounter(
        model_file='yolov8l.pt',
        conf=0.4,
        iou=0.5,
        track_max_age=45,   
        track_min_hits=15
    )

    current_date = datetime.now().strftime('%Y-%m-%d')
    output_file_path = os.path.join(storage_volume_path, f'{code_panel}.xlsx')


    logging.info(f'Chemin de sauvegarde du fichier : {output_file_path}')

    try:
        counter.predict_video(cam, output_file_path, frame_interval=0.1)
    except Exception as e:
        logging.error(f'Erreur lors de la prédiction vidéo : {e}')
    