import os
import logging
import time
from datetime import datetime
from dotenv import load_dotenv
from modules.hikvision_client import HikvisionClient
from modules.yolov8_object_counter import YOLOv8_ObjectCounter
from modules.drive_upload import upload_to_drive

# Configurer la journalisation
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Charger les variables d'environnement du fichier .env
load_dotenv()

# Récupérer les détails de la caméra à partir des variables d'environnement
cam_ip = os.getenv('CAM_IP')
cam_user = os.getenv('CAM_USER')
cam_password = os.getenv('CAM_PASSWORD')
cam_name = os.getenv('CAM_NAME')

# Récupérer le chemin du volume de stockage à partir des variables d'environnement
storage_volume_path = os.getenv('STORAGE_VOLUME_PATH')

# Assurer que le répertoire de stockage existe
if not os.path.exists(storage_volume_path):
    os.makedirs(storage_volume_path)
    logging.info(f'Répertoire créé : {storage_volume_path}')
else:
    logging.info(f'Répertoire déjà existant : {storage_volume_path}')

if __name__ == '__main__':
    cam = HikvisionClient(cam_ip, cam_user, cam_password)
    counter = YOLOv8_ObjectCounter(model_file='yolov8m.pt', conf=0.60, iou=0.60)

    current_date = datetime.now().strftime('%Y-%m-%d')
    output_file_path = os.path.join(storage_volume_path, f'{cam_name}-{current_date}.csv')
    
    # Ajouter un message de débogage pour vérifier le chemin
    logging.info(f'Chemin de sauvegarde du fichier CSV : {output_file_path}')

    counter.predict_video(cam, output_file_path, frame_skip=2, update_interval=2)

    # Vérifier si le fichier existe avant de tenter l'upload
    if os.path.exists(output_file_path):
        logging.info(f'Le fichier {output_file_path} a été créé avec succès.')
        # Télécharger le fichier CSV sur Google Drive
        google_drive_folder_id = os.getenv('GOOGLE_DRIVE_FOLDER_ID')
        upload_to_drive(output_file_path, google_drive_folder_id)
    else:
        logging.error(f'Le fichier {output_file_path} n\'a pas été créé.')
