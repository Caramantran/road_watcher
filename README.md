# Road Watcher - Intelligent Traffic Monitoring System

Welcome to **Road Watcher**, the intelligent solution for real-time traffic monitoring and analysis using advanced object detection and counting techniques. This system leverages cutting-edge AI technology to provide accurate and actionable insights for urban traffic management.

## Overview

Road Watcher utilizes state-of-the-art YOLOv8 object detection models integrated with Hikvision cameras to monitor traffic in real-time. This powerful combination allows for seamless detection, tracking, and counting of vehicles, providing invaluable data for improving traffic flow, enhancing safety, and optimizing infrastructure planning.

## Key Features

### Real-Time Traffic Monitoring
- **Continuous Surveillance**: Monitor traffic 24/7 using high-definition Hikvision cameras.
### Advanced Object Detection
- **Accurate Vehicle Detection**: Identify and count various types of vehicles including cars, trucks, motorcycles, and buses with high precision.
- **Dynamic Tracking**: Track vehicle movement across multiple frames for detailed traffic analysis.

### Comprehensive Data Analytics
- **Detailed Reports**: Generate comprehensive reports on traffic density, vehicle types, and peak hours.

### Integration and Flexibility
- **Seamless Integration**: Easily integrate with existing traffic management systems and city infrastructure.
- **Scalable Solution**: Expand monitoring capabilities to cover more areas as needed.

## Business Benefits

### Enhance Traffic Management
- **Reduce Congestion**: Optimize traffic light timings and road usage to minimize traffic jams.
- **Improve Safety**: Detect and respond to traffic incidents quickly, reducing the risk of accidents.

### Optimize Infrastructure Planning
- **Data-Driven Decisions**: Use accurate traffic data to plan new roads, expand existing ones, and improve public transportation routes.
- **Resource Allocation**: Allocate resources more effectively based on real-time traffic data and trends.

### Boost Operational Efficiency
- **Automated Monitoring**: Eliminate the need for manual traffic counting and monitoring, freeing up resources for other tasks.
- **Real-Time Insights**: Gain immediate access to traffic data, enabling quick decision-making and response.

## Getting Started

### Prerequisites
- **Hikvision Camera**: Ensure you have a Hikvision camera set up and connected to your network.
- **Conda**: Install Conda to manage the project environment.

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/road-watcher.git
   cd road-watcher

2. **Create and Activate the Environment**:

   ```bash
   conda env create -f environment.yml
   conda activate yolo_hikvision_env
 
 
3. **Run the Main Script**:

   ```bash
   python main.py

## Usage
1. **Configure the Camera**:
Update the IP address, username, and password of your Hikvision camera in the main.py file.

2. **Start Monitoring**:
Run the main script to start real-time traffic monitoring and data collection.

3. **Access Reports**:
Reports are generated in CSV format and saved in the specified directory. Use these reports for further analysis and decision-making.

## Project Structure

**modules/:** Contains the core modules for camera integration, object detection, and data processing.

**notebooks/:** Jupyter notebooks for development and testing.

**env/:** Conda environment files.

**yolov8m.pt:** Pre-trained YOLOv8 model file.

**coco.names:** List of object names used by the YOLO model.

**Dockerfile:** Docker configuration for containerized deployment.

**environment.yml:** Conda environment configuration file.

**main.py:** Entry point of the application.

**README.md:** Project documentation.

## Contributing
We welcome contributions to enhance the capabilities and features of Road Watcher. Please feel free to submit pull requests or report issues.

## License
This project is licensed under the MIT License.
