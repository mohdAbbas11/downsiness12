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