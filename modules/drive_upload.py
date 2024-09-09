from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import os
import io
import logging
import pandas as pd
from datetime import datetime

def get_drive_service():
    SCOPES = ['https://www.googleapis.com/auth/drive']
    creds = service_account.Credentials.from_service_account_file('credentials.json', scopes=SCOPES)
    return build('drive', 'v3', credentials=creds)

def file_exists_on_drive(service, folder_id, file_name):
    query = f"'{folder_id}' in parents and name='{file_name}' and trashed=false"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get('files', [])
    if files:
        return files[0]['id']
    return None

def download_file_from_drive(service, file_id, local_path):
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(local_path, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        logging.info(f"Download {int(status.progress() * 100)}%.")
    fh.close()

def update_google_sheet(service, file_id, local_path):
    media = MediaFileUpload(local_path, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', resumable=True)
    updated_file = service.files().update(fileId=file_id, media_body=media).execute()
    logging.info(f"Updated file ID: {updated_file.get('id')}")

def get_or_create_folder(service, folder_name, parent_folder_id):
    query = f"'{parent_folder_id}' in parents and name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    folders = results.get('files', [])
    if folders:
        return folders[0]['id']
    else:
        folder_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [parent_folder_id]
        }
        folder = service.files().create(body=folder_metadata, fields='id').execute()
        return folder.get('id')

def upload_to_drive(file_path, folder_id):
    try:
        service = get_drive_service()
        file_name = os.path.basename(file_path)
        city = os.getenv('CITY')
        city_folder_id = get_or_create_folder(service, city, folder_id)

        logging.info(f'Téléchargement du fichier : {file_name} dans le dossier : {city}')

        # Check if file already exists
        file_id = file_exists_on_drive(service, city_folder_id, file_name)
        if file_id:
            logging.info(f"File {file_name} exists on Google Drive. Updating it.")
            update_google_sheet(service, file_id, file_path)
        else:
            logging.info(f"File {file_name} does not exist on Google Drive. Creating a new one.")
            file_metadata = {
                'name': file_name,
                'parents': [city_folder_id],
                'mimeType': 'application/vnd.google-apps.spreadsheet'  # Convert to Google Sheets
            }
            media = MediaFileUpload(file_path, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', resumable=True)
            created_file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
            logging.info(f'File ID: {created_file.get("id")}')
    except Exception as e:
        logging.error(f'Erreur lors du téléchargement sur Google Drive : {e}')
        if hasattr(e, 'args') and len(e.args) > 0:
            logging.error(f'Détails de l\'erreur : {e.args[0]}')

def append_data_to_existing_file(existing_file_path, new_data):
    try:
        # Load the existing data
        if os.path.exists(existing_file_path):
            existing_df = pd.read_excel(existing_file_path, sheet_name=None)
            logging.info(f"Loaded existing data from {existing_file_path}")
        else:
            existing_df = {}
            logging.info(f"No existing data found at {existing_file_path}. Creating new file.")

        # Convert new data to DataFrame
        new_df = pd.DataFrame(new_data)
        logging.info("New data converted to DataFrame.")

        # Get current date
        current_date = datetime.now().strftime('%Y-%m-%d')

        if current_date in existing_df:
            existing_df[current_date] = pd.concat([existing_df[current_date], new_df]).drop_duplicates()
        else:
            existing_df[current_date] = new_df

        # Save the combined data back to the file
        with pd.ExcelWriter(existing_file_path, engine='openpyxl') as writer:
            for sheet_name, df in existing_df.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        logging.info(f"Data written and saved to {existing_file_path}")
        
        # Print the combined data to console for verification
        print(new_df)
    except Exception as e:
        logging.error(f"Error appending data to existing file: {e}")
        if hasattr(e, 'args') and len(e.args) > 0:
            logging.error(f"Error details: {e.args[0]}")
