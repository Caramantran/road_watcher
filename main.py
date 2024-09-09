import os
import logging
import time
from datetime import datetime
from dotenv import load_dotenv
from modules.hikvision_client import HikvisionClient
from modules.yolov8_object_counter import YOLOv8_ObjectCounter
from modules.drive_upload import upload_to_drive
from modules.data_aggregator import aggregate_data_to_global, initialize_service

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from the .env file
load_dotenv()

# Retrieve camera details from environment variables
cam_ip = os.getenv('CAM_IP')
cam_user = os.getenv('CAM_USER')
cam_password = os.getenv('CAM_PASSWORD')
code_panel = os.getenv('CODE_PANEL')

# Retrieve storage volume path from environment variables
storage_volume_path = os.getenv('STORAGE_VOLUME_PATH')
city = os.getenv('CITY')

# Ensure the storage directory exists
if not os.path.exists(storage_volume_path):
    os.makedirs(storage_volume_path)
    logging.info(f'Created directory: {storage_volume_path}')
else:
    logging.info(f'Directory already exists: {storage_volume_path}')

if __name__ == '__main__':
    # Initialize camera client
    cam = HikvisionClient(cam_ip, cam_user, cam_password)
    counter = YOLOv8_ObjectCounter(model_file='yolov8m.pt', conf=0.50, iou=0.45)

    # Generate file path based on date and code panel
    current_date = datetime.now().strftime('%Y-%m-%d')
    output_file_path = os.path.join(storage_volume_path, f'{code_panel}')
    
    logging.info(f'Save path for file: {output_file_path}')

    try:
        # Run object detection and tracking on video feed
        counter.predict_video(cam, output_file_path, frame_skip=2, update_interval=2)
    except Exception as e:
        logging.error(f'Error during video prediction: {e}')
        
    # Aggregate the detection results and save to Excel file
    aggregated_data = counter.read_and_format_detections('detections.txt')
    counter.save_aggregated_data_to_excel(aggregated_data, output_file_path, counter.labels)

    # Check if file exists before attempting to upload
    if os.path.exists(output_file_path):
        logging.info(f'File {output_file_path} successfully created.')
        
        # Upload the file to Google Drive
        google_drive_folder_id = os.getenv('GOOGLE_DRIVE_FOLDER_ID')
        if google_drive_folder_id:
            try:
                upload_to_drive(output_file_path, google_drive_folder_id)
                logging.info(f'File {output_file_path} uploaded to Google Drive.')
            except Exception as e:
                logging.error(f'Error during Google Drive upload: {e}')
        else:
            logging.warning('GOOGLE_DRIVE_FOLDER_ID not found. Skipping upload.')
    else:
        logging.error(f'File {output_file_path} was not created.')

    # Initialize Google services (Drive & Sheets)
    try:
        drive_service, sheets_service = initialize_service()
        logging.info("Google Drive and Sheets services initialized.")
        
        # Aggregate data from all files in subfolders to the global sheet
        aggregate_data_to_global(drive_service, sheets_service, google_drive_folder_id, os.getenv('GLOBAL_SHEET_ID'))
        logging.info(f'Global Sheet ID loaded: {os.getenv("GLOBAL_SHEET_ID")}')
    except Exception as e:
        logging.error(f'Error during global data aggregation: {e}')
