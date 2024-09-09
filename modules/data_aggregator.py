import os
import logging
import pandas as pd
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
from modules.drive_upload import get_drive_service

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
GOOGLE_CREDENTIALS_PATH = 'credentials.json'
PARENT_FOLDER_ID = os.getenv('GOOGLE_DRIVE_FOLDER_ID')
GLOBAL_SHEET_ID = os.getenv('GLOBAL_SHEET_ID')

# Initialize Google Sheets and Google Drive API
SCOPES = ['https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/spreadsheets']
credentials = Credentials.from_service_account_file(GOOGLE_CREDENTIALS_PATH, scopes=SCOPES)

def initialize_service():
    try:
        logging.info("Initializing Google Drive and Google Sheets API services...")
        drive_service = build('drive', 'v3', credentials=credentials)
        sheets_service = build('sheets', 'v4', credentials=credentials)
        logging.info("Services initialized successfully.")
        return drive_service, sheets_service
    except Exception as e:
        logging.error(f"Failed to initialize services: {e}")
        raise

# Get all files from subfolders (cities) in Google Drive
def get_files_from_folders(drive_service, parent_folder_id):
    try:
        logging.info(f"Fetching subfolders from Google Drive parent folder ID: {parent_folder_id}...")
        query = f"'{parent_folder_id}' in parents and mimeType = 'application/vnd.google-apps.folder'"
        results = drive_service.files().list(q=query, fields="files(id, name)").execute()
        folders = results.get('files', [])
        
        if not folders:
            logging.warning(f"No subfolders found in folder ID: {parent_folder_id}")
        
        files_to_aggregate = []
        for folder in folders:
            folder_id = folder['id']
            logging.info(f"Fetching files from subfolder: {folder['name']} (ID: {folder_id})...")
            sub_query = f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.spreadsheet'"
            sub_results = drive_service.files().list(q=sub_query, fields="files(id, name)").execute()
            files = sub_results.get('files', [])
            
            if files:
                logging.info(f"Found {len(files)} files in subfolder {folder['name']}.")
            else:
                logging.warning(f"No Google Sheets files found in subfolder: {folder['name']}")

            files_to_aggregate.extend(files)

        return files_to_aggregate
    except Exception as e:
        logging.error(f"Error fetching files from folders: {e}")
        raise

# Read all sheets from a Google Sheets file
def read_all_sheets(sheets_service, sheet_id):
    try:
        logging.info(f"Fetching sheet metadata for sheet ID: {sheet_id}...")
        sheet_metadata = sheets_service.spreadsheets().get(spreadsheetId=sheet_id).execute()
        sheet_names = [s['properties']['title'] for s in sheet_metadata.get('sheets', [])]
        logging.info(f"Found sheets: {sheet_names}")

        all_data = pd.DataFrame()

        for sheet_name in sheet_names:
            logging.info(f"Reading data from sheet: {sheet_name}...")
            result = sheets_service.spreadsheets().values().get(
                spreadsheetId=sheet_id, range=sheet_name
            ).execute()
            values = result.get('values', [])
            
            if values:
                df = pd.DataFrame(values[1:], columns=values[0])  # Assuming first row is header
                df['SheetName'] = sheet_name  # Add column to track sheet name (date)
                all_data = pd.concat([all_data, df])
                logging.info(f"Data from sheet '{sheet_name}' read successfully. Rows: {len(df)}.")
            else:
                logging.warning(f"No data found in sheet: {sheet_name}")
        
        logging.info(f"Total rows aggregated from all sheets: {len(all_data)}.")
        return all_data
    except Exception as e:
        logging.error(f"Error reading sheets from file: {e}")
        raise

# Append data to the global sheet
def append_data_to_global_sheet(sheets_service, global_sheet_id, data):
    try:
        logging.info(f"Appending data to global sheet ID: {global_sheet_id}...")
        sheet = sheets_service.spreadsheets()
        body = {
            'values': data.values.tolist()
        }
        result = sheet.values().append(
            spreadsheetId=global_sheet_id, range='A1',
            valueInputOption="RAW", body=body
        ).execute()
        logging.info(f"Data appended successfully. Updated rows: {result.get('updates').get('updatedRows', 'N/A')}.")
    except Exception as e:
        logging.error(f"Error appending data to global sheet: {e}")
        raise

# Main function to aggregate data from subfolder files into the global sheet
def aggregate_data_to_global(drive_service, sheets_service, parent_folder_id, global_sheet_id):
    try:
        logging.info("Starting data aggregation process...")
        # Get all files from the subfolders (cities)
        files_to_aggregate = get_files_from_folders(drive_service, parent_folder_id)

        if not files_to_aggregate:
            logging.warning("No files to aggregate.")
            return

        for file in files_to_aggregate:
            file_id = file['id']
            logging.info(f"Processing file: {file['name']} (ID: {file_id})")

            # Read data from all sheets within each Google Sheet file
            file_data = read_all_sheets(sheets_service, file_id)
            
            if not file_data.empty:
                # Append the read data to the global sheet
                append_data_to_global_sheet(sheets_service, global_sheet_id, file_data)
            else:
                logging.warning(f"No data found in {file['name']}")
        
        logging.info("Data aggregation completed successfully.")
    except Exception as e:
        logging.error(f"Error during data aggregation: {e}")
        raise

if __name__ == '__main__':
    try:
        # Initialize services
        logging.info("Initializing services...")
        drive_service, sheets_service = initialize_service()

        # Aggregate data from subfolders to the global sheet
        aggregate_data_to_global(drive_service, sheets_service, PARENT_FOLDER_ID, GLOBAL_SHEET_ID)
    except Exception as e:
        logging.error(f"Fatal error during script execution: {e}")
