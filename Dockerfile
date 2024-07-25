# Use the official miniconda image as the base image
FROM continuumio/miniconda3

# Set the working directory
WORKDIR /road_watcher

# Install system dependencies for OpenCV, OpenGL, and UVC support
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libegl1-mesa \
    libglib2.0-0 \
    libglvnd-dev \
    libgl1-mesa-dri \
    v4l-utils \
    ffmpeg \
    usbutils \
    udev \
    python3-opencv

# Copy the environment.yml file into the container
COPY environment.yml .

# Create the conda environment from the environment.yml file
RUN conda env create -f environment.yml

# Activate the environment for subsequent commands
SHELL ["conda", "run", "-n", "yolo_env", "/bin/bash", "-c"]

# Install additional pip packages separately to handle any issues
RUN conda run -n yolo_env pip install ultralytics hikvisionapi python-dotenv

# Copy the rest of your application code into the container
COPY . .

# Copy the .env file into the container
COPY .env .

# Ensure the conda environment is active when running the application
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "yolo_env", "python", "main.py"]
