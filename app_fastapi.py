from fastapi import FastAPI, HTTPException, Response, File, UploadFile, Form
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import cv2
import numpy as np
import base64
import uvicorn
import asyncio
import time
import io
import os
import serial
import serial.tools.list_ports
from typing import Optional, List
from drowsiness_detection import DrowsinessDetector

app = FastAPI(
    title="Drowsiness Detection API",
    description="API for detecting driver drowsiness from images and video streams",
    version="1.0.0"
)

# Initialize the detector
detector = None
arduino = None

def get_detector(arduino_port=None, disable_arduino=True):
    global detector
    if detector is None:
        try:
            # Initialize with arduino_port and disable_arduino parameters
            detector = DrowsinessDetector(arduino_port=arduino_port, disable_arduino=disable_arduino)
        except Exception as e:
            print(f"Error initializing detector: {str(e)}")
            # If initialization fails, try again with minimal options
            detector = DrowsinessDetector(arduino_port=None, disable_arduino=True)
    return detector

class ImageRequest(BaseModel):
    image: str  # Base64 encoded image
    
class DetectionResponse(BaseModel):
    face_detected: bool
    drowsy: Optional[bool] = None
    eyes_closed: Optional[bool] = None
    yawning: Optional[bool] = None
    head_down: Optional[bool] = None
    ear: Optional[float] = None
    head_angle: Optional[float] = None
    message: Optional[str] = None

class ArduinoResponse(BaseModel):
    success: bool
    message: str
    port: Optional[str] = None
    connected: bool = False

@app.get("/", response_class=FileResponse)
async def read_root():
    return FileResponse("index.html")

@app.get("/arduino/ports", response_model=List[str])
async def get_arduino_ports():
    """Get list of available Arduino ports"""
    ports = [port.device for port in serial.tools.list_ports.comports()]
    return ports

@app.post("/arduino/connect", response_model=ArduinoResponse)
async def connect_arduino(port: str = Form(...), baud_rate: int = Form(9600)):
    """Connect to Arduino on specified port"""
    global arduino, detector
    
    try:
        # Close existing connection if any
        if arduino is not None and arduino.is_open:
            arduino.close()
            
        # Try to connect to Arduino
        arduino = serial.Serial(port, baud_rate, timeout=1)
        
        # Update detector with new Arduino connection
        if detector is not None:
            detector.arduino = arduino
            detector.arduino_connected = True
            detector.disable_arduino = False
        else:
            # Initialize detector with Arduino
            detector = get_detector(arduino_port=port, disable_arduino=False)
            
        return ArduinoResponse(
            success=True,
            message=f"Successfully connected to Arduino on {port}",
            port=port,
            connected=True
        )
    except Exception as e:
        return ArduinoResponse(
            success=False,
            message=f"Failed to connect to Arduino: {str(e)}",
            connected=False
        )

@app.post("/arduino/disconnect", response_model=ArduinoResponse)
async def disconnect_arduino():
    """Disconnect from Arduino"""
    global arduino, detector
    
    if arduino is None or not arduino.is_open:
        return ArduinoResponse(
            success=True,
            message="No Arduino connection to disconnect",
            connected=False
        )
    
    try:
        arduino.close()
        
        # Update detector
        if detector is not None:
            detector.arduino_connected = False
            detector.disable_arduino = True
            detector.arduino = None
            
        return ArduinoResponse(
            success=True,
            message="Successfully disconnected from Arduino",
            connected=False
        )
    except Exception as e:
        return ArduinoResponse(
            success=False,
            message=f"Error disconnecting from Arduino: {str(e)}",
            connected=arduino.is_open if arduino else False
        )

@app.get("/arduino/status", response_model=ArduinoResponse)
async def arduino_status():
    """Get Arduino connection status"""
    global arduino
    
    if arduino is None:
        return ArduinoResponse(
            success=True,
            message="Arduino not connected",
            connected=False
        )
    
    try:
        is_connected = arduino.is_open
        port = arduino.port if is_connected else None
        
        return ArduinoResponse(
            success=True,
            message=f"Arduino {'connected' if is_connected else 'not connected'}",
            port=port,
            connected=is_connected
        )
    except Exception as e:
        return ArduinoResponse(
            success=False,
            message=f"Error checking Arduino status: {str(e)}",
            connected=False
        )

@app.post("/detect", response_model=DetectionResponse)
async def detect_drowsiness(request: ImageRequest):
    # Get the detector instance
    detector_instance = get_detector()
    
    try:
        # Decode the base64 image
        image_data = base64.b64decode(request.image)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = detector_instance.detector(gray, 0)
        
        if len(faces) == 0:
            return DetectionResponse(
                face_detected=False,
                message="No face detected in the image"
            )
        
        # Process the first face
        try:
            detector_instance.process_landmarks(image, faces[0])
        except Exception as e:
            print(f"Error processing landmarks: {str(e)}")
            # Return partial results if available
            return DetectionResponse(
                face_detected=True,
                message=f"Error processing landmarks: {str(e)}"
            )
        
        # Return the detection results
        return DetectionResponse(
            face_detected=True,
            drowsy=detector_instance.drowsy,
            eyes_closed=detector_instance.eyes_closed,
            yawning=detector_instance.yawning,
            head_down=detector_instance.head_down,
            ear=float(detector_instance.current_ear),
            head_angle=float(detector_instance.head_metrics['angle'])
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def generate_frames():
    """Generate frames from webcam with drowsiness detection"""
    # Get the detector instance
    detector_instance = get_detector()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        # Return a black frame
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', black_frame)
        io_buf = io.BytesIO(buffer)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + io_buf.getvalue() + b'\r\n')
        return
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    try:
        while True:
            # Read frame
            success, frame = cap.read()
            if not success:
                break
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = detector_instance.detector(gray, 0)
            
            if len(faces) > 0:
                # Process the first face
                try:
                    frame = detector_instance.process_landmarks(frame, faces[0])
                except Exception as e:
                    print(f"Error processing landmarks: {str(e)}")
                    cv2.putText(frame, f"Error: {str(e)[:30]}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            else:
                # No face detected
                cv2.putText(frame, "No Face Detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add Arduino connection status
            global arduino
            if arduino and arduino.is_open:
                cv2.putText(frame, f"Arduino: Connected ({arduino.port})", (10, frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                cv2.putText(frame, "Arduino: Not Connected", (10, frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Encode the frame in JPEG format
            _, buffer = cv2.imencode('.jpg', frame)
            io_buf = io.BytesIO(buffer)
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + io_buf.getvalue() + b'\r\n')
            
            # Sleep to reduce CPU usage
            await asyncio.sleep(0.03)
    except Exception as e:
        print(f"Error in video stream: {str(e)}")
    finally:
        cap.release()

@app.get("/video_feed")
async def video_feed():
    """Video streaming route"""
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=8000, reload=True) 