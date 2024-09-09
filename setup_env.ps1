#RUN THE SCRIPT WITH THIS CML ./setup_env.ps1

# PowerShell script to create .env file with user input
$CAM_IP = Read-Host "Enter the Camera IP"
$CAM_USER = Read-Host "Enter the Camera User"
$CAM_PASSWORD = Read-Host "Enter the Camera Password"
$CODE_PANEL = Read-Host "Enter the Code Panel"
$GOOGLE_DRIVE_FOLDER_ID = Read-Host "Enter the Google Drive Folder ID"
$CITY = Read-Host "Enter the City"

# Create the .env file content
$envContent = @"
CAM_IP=$CAM_IP
CAM_USER=$CAM_USER
CAM_PASSWORD=$CAM_PASSWORD
CODE_PANEL=$CODE_PANEL
STORAGE_VOLUME_PATH=./Output_data
GOOGLE_DRIVE_FOLDER_ID=$GOOGLE_DRIVE_FOLDER_ID
CITY=$CITY
"@

# Write the content to a .env file
$envContent | Out-File -FilePath ".env" -Encoding ascii

# Confirm successful creation
Write-Host ".env file created successfully!"
