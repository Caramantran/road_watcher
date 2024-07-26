from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import os
import logging

def upload_to_drive(file_path, folder_id):
    try:
        # Charger les identifiants du fichier 'credentials.json'
        SCOPES = ['https://www.googleapis.com/auth/drive.file']
        creds = service_account.Credentials.from_service_account_file('credentials.json', scopes=SCOPES)
        service = build('drive', 'v3', credentials=creds)

        # Obtenir le nom du fichier
        file_name = os.path.basename(file_path)
        logging.info(f'Téléchargement du fichier : {file_name}')

        # Créer un objet de téléchargement de fichier média
        media = MediaFileUpload(file_path, resumable=True)

        # Définir les métadonnées pour le fichier
        file_metadata = {
            'name': file_name,
            'parents': [folder_id]  # ID du dossier où vous voulez télécharger le fichier
        }

        # Télécharger le fichier
        file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        logging.info(f'File ID: {file.get("id")}')
    except Exception as e:
        logging.error(f'Erreur lors du téléchargement sur Google Drive : {e}')
