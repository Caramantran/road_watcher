# Use Miniconda (smaller than Anaconda) as the base image
FROM continuumio/miniconda3

# Set the working directory inside the container
WORKDIR /usr/src/app

# Install system dependencies (OpenCV, timezone data)
RUN apt-get update && apt-get install -y \
    python3-opencv \
    tzdata \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the timezone
ENV TZ=Africa/Casablanca

# Link timezone configuration
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Copy the environment.yml file into the container
COPY environment.yml .

# Create the conda environment from the environment.yml file
RUN conda env create -f environment.yml \
    && conda clean -afy

# Install additional pip packages that might not be in conda
RUN conda run -n yolo_env pip install ultralytics hikvisionapi python-dotenv

# Copy the project files into the container
COPY . .

# Adjust permissions to ensure the app directory is writable
RUN chmod -R 777 /usr/src/app

# Ensure the conda environment is active when running the application
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "yolo_env", "python", "main.py"]
