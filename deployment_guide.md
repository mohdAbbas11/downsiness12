# Drowsiness Detection System - Deployment Guide

This guide explains how to deploy the drowsiness detection system as a website that can be shared with anyone.

## Prerequisites

- Python 3.7+
- OpenCV
- dlib
- PyTorch
- FastAPI
- Docker (optional, for containerized deployment)

## Option 1: Local Deployment

### Using FastAPI

1. Navigate to your project directory:
   ```
   cd path/to/project/d2
   ```

2. Install required packages:
   ```
   pip install fastapi uvicorn opencv-python dlib torch torchvision numpy serial
   ```

3. Run the FastAPI app:
   ```
   python app_fastapi.py
   ```

4. Access the website locally at:
   ```
   http://localhost:8000
   ```
   
5. Access API documentation at:
   ```
   http://localhost:8000/docs
   ```

## Option 2: Docker Deployment

Docker provides a consistent environment for deployment.

### FastAPI Docker Deployment

1. Create a Dockerfile in your project directory:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for OpenCV and dlib
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model
COPY . .

# Download shape predictor model if not present
RUN if [ ! -f shape_predictor_68_face_landmarks.dat ]; then \
    pip install gdown && \
    gdown --id 1Xg08olCgQ_wKhfGr9I_0_X7CPXS8q0qf && \
    unzip shape_predictor_68_face_landmarks.dat.zip && \
    rm shape_predictor_68_face_landmarks.dat.zip; \
    fi

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
```

2. Create a requirements.txt file:

```
fastapi==0.68.0
uvicorn==0.15.0
opencv-python==4.5.3.56
dlib==19.22.1
torch==1.9.0
torchvision==0.10.0
numpy==1.21.2
pyserial==3.5
python-multipart==0.0.5
pydantic==1.8.2
gdown==4.4.0
```

3. Build and run the Docker container:
   ```
   docker build -t drowsiness-detection -f Dockerfile.fastapi .
   docker run -p 8000:8000 drowsiness-detection
   ```

4. Access the website at:
   ```
   http://localhost:8000
   ```

## Option 3: Cloud Deployment

### Deploying to Heroku

1. Create a Procfile for Heroku:
   ```
   web: uvicorn app_fastapi:app --host=0.0.0.0 --port=${PORT:-8000}
   ```

2. Deploy using the Heroku CLI:
   ```
   heroku create your-app-name
   git push heroku main
   ```

### Deploying to AWS EC2

1. Launch an EC2 instance with Ubuntu.
2. SSH into your instance.
3. Install Docker.
4. Clone your repository.
5. Build and run your Docker container.
6. Configure security groups to allow traffic on port 8000.

### Deploying to Google Cloud Run

1. Install Google Cloud SDK.
2. Build and push your Docker image:
   ```
   gcloud builds submit --tag gcr.io/your-project/drowsiness-detection
   ```
3. Deploy to Cloud Run:
   ```
   gcloud run deploy --image gcr.io/your-project/drowsiness-detection --platform managed
   ```

## Important Notes for Production Deployment

1. **Camera Access**: Browser-based camera access requires HTTPS for security reasons when deployed online.

2. **Performance**: The face detection and landmark detection are computationally intensive. Consider:
   - Using a server with GPU support
   - Optimizing frame rate and resolution
   - Using WebRTC for better video streaming

3. **Security**: Implement proper authentication if this is a sensitive application.

4. **Scaling**: For multiple users, implement proper scaling strategies using load balancers.

5. **Model Deployment**: Ensure the shape predictor model file is included in your deployment.

## Troubleshooting

- If the camera doesn't work in the browser, check that you're using HTTPS or localhost.
- If face detection is slow, consider reducing the resolution or frame rate.
- For Docker deployment issues, check that all dependencies are properly installed. 