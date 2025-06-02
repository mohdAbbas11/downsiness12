# Driver Drowsiness Detection System

A real-time driver drowsiness detection system that can be deployed as a web service. The system uses computer vision techniques to detect signs of drowsiness in drivers, including eye closure, yawning, and head position.

## Features

- Real-time drowsiness detection using webcam
- REST API for image analysis
- Video streaming capability
- FastAPI implementation with built-in documentation
- Docker support for easy deployment
- Arduino integration for hardware alerts (optional)

## Prerequisites

- Python 3.7+
- Webcam
- dlib's facial landmark predictor file (`shape_predictor_68_face_landmarks.dat`)
- Arduino (optional, for hardware alerts)

## Quick Start

### Local Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download the shape predictor model:
   ```
   # Option 1: Using gdown
   pip install gdown
   gdown --id 1Xg08olCgQ_wKhfGr9I_0_X7CPXS8q0qf
   
   # Option 2: Download manually from dlib website
   # Place the file in the project root directory
   ```

### Running the Application

```
python app_fastapi.py
```

Access the web interface at: http://localhost:8000
API documentation: http://localhost:8000/docs

### Docker Deployment

Using Docker Compose:

```
docker-compose up --build
```

Or run the container directly:

```
docker build -t drowsiness-detection -f Dockerfile.fastapi .
docker run -p 8000:8000 drowsiness-detection
```

## API Usage

### Detect Drowsiness in an Image

```python
import requests
import base64
import cv2

# Load an image
image = cv2.imread('test_image.jpg')

# Convert to base64
_, buffer = cv2.imencode('.jpg', image)
img_base64 = base64.b64encode(buffer).decode('utf-8')

# Send request to API
response = requests.post(
    'http://localhost:8000/detect',
    json={'image': img_base64}
)

# Print results
print(response.json())
```

### Embedding Video Feed in HTML

```html
<img src="http://localhost:8000/video_feed" width="640" height="480" />
```

## Deployment Options

For detailed deployment instructions, see [deployment_guide.md](deployment_guide.md).

## License

This project is licensed under the MIT License - see the LICENSE file for details. 