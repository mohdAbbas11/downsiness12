import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dlib
import time
import serial
import argparse
from torchvision import transforms
from threading import Thread

# Constants
EYE_AR_THRESH = 0.25  # Lowered threshold to better detect eye closure
YAWN_THRESH = 20
EYE_AR_CONSEC_FRAMES = 30  # Reduced from 48 to detect drowsiness faster
HEAD_DOWN_THRESH = 20  # Increased from 15 to reduce false positives
HEAD_DOWN_ANGLE_THRESH = 25  # Angle threshold in degrees
HEAD_DOWN_CONSEC_FRAMES = 25  # Increased from 20 to reduce false positives
EYES_CLOSED_THRESH = 0.22  # Threshold for completely closed eyes
EYES_CLOSED_CONSEC_FRAMES = 15
ARDUINO_PORT = 'COM3'  # Change this to match your Arduino port
ARDUINO_BAUD_RATE = 9600

class DrowsinessDetectionModel(nn.Module):
    def __init__(self):
        super(DrowsinessDetectionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 2)  # 2 classes: drowsy or not drowsy
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class DrowsinessDetector:
    def __init__(self, arduino_port=ARDUINO_PORT, arduino_baud_rate=ARDUINO_BAUD_RATE, disable_arduino=False):
        # Initialize face detector and landmark predictor
        print("Loading facial landmark predictor...")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        # Initialize PyTorch model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = DrowsinessDetectionModel().to(self.device)
        
        # Initialize eye and mouth indices for facial landmarks
        self.left_eye_indices = list(range(36, 42))
        self.right_eye_indices = list(range(42, 48))
        self.mouth_indices = list(range(48, 68))
        
        # Initialize counters
        self.eye_closed_counter = 0
        self.eyes_completely_closed_counter = 0
        self.yawn_counter = 0
        self.head_down_counter = 0
        
        # Initialize flags
        self.drowsy = False
        self.yawning = False
        self.head_down = False
        self.eyes_closed = False
        
        # Initialize head metrics
        self.head_metrics = {'angle': 0, 'vertical_ratio': 0, 'absolute_distance': 0}
        
        # Initialize image transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Setup Arduino communication
        self.arduino_connected = False
        self.arduino = None
        self.disable_arduino = disable_arduino
        
        if not disable_arduino:
            try:
                self.arduino = serial.Serial(arduino_port, arduino_baud_rate, timeout=1)
                print(f"Connected to Arduino on {arduino_port}")
                self.arduino_connected = True
            except:
                print("Failed to connect to Arduino. Running without hardware alerts.")
                self.arduino_connected = False

    def send_alert_to_arduino(self, alert_type):
        if self.disable_arduino:
            return
            
        if self.arduino_connected and self.arduino:
            try:
                if alert_type == "drowsy":
                    self.arduino.write(b'1')
                elif alert_type == "yawn":
                    self.arduino.write(b'2')
                elif alert_type == "head_down":
                    self.arduino.write(b'3')
                elif alert_type == "eyes_closed":
                    self.arduino.write(b'4')
                else:
                    self.arduino.write(b'0')  # Normal state
            except Exception as e:
                print(f"Arduino communication error: {str(e)}")
                self.arduino_connected = False

    def calculate_eye_aspect_ratio(self, eye_landmarks):
        # Compute the euclidean distances between the vertical eye landmarks
        A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        
        # Compute the euclidean distance between the horizontal eye landmarks
        C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        # Compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear

    def detect_yawn(self, mouth_landmarks):
        # Calculate the distance between top and bottom lip
        top_lip = mouth_landmarks[13]
        bottom_lip = mouth_landmarks[19]
        
        mouth_open = np.linalg.norm(top_lip - bottom_lip)
        return mouth_open > YAWN_THRESH

    def detect_head_down(self, face_landmarks):
        """
        Enhanced head pose estimation based on facial landmarks
        Using multiple reference points and angles for more accurate detection
        """
        # Get key facial landmarks
        nose_tip = face_landmarks[30]
        nose_bridge = face_landmarks[27]
        chin = face_landmarks[8]
        left_eye_center = np.mean([face_landmarks[i] for i in self.left_eye_indices], axis=0)
        right_eye_center = np.mean([face_landmarks[i] for i in self.right_eye_indices], axis=0)
        eye_center = np.mean([left_eye_center, right_eye_center], axis=0)
        forehead = face_landmarks[27] - (face_landmarks[8] - face_landmarks[27])  # Estimate forehead position
        
        # Calculate vertical distance ratio (normalized by face size)
        face_height = np.linalg.norm(chin - forehead)
        vertical_ratio = (nose_tip[1] - eye_center[1]) / max(1, face_height) * 100
        
        # Calculate angle between vertical line and line from eyes to nose
        eye_to_nose = nose_tip - eye_center
        vertical_vector = np.array([0, 1])
        
        # Normalize vectors
        eye_to_nose_normalized = eye_to_nose / np.linalg.norm(eye_to_nose)
        
        # Calculate angle in degrees
        dot_product = np.dot(eye_to_nose_normalized, vertical_vector)
        angle = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
        
        # Calculate absolute vertical distance (for compatibility with old method)
        absolute_distance = nose_tip[1] - eye_center[1]
        
        # Store values for display
        self.head_metrics = {
            'angle': angle,
            'vertical_ratio': vertical_ratio,
            'absolute_distance': absolute_distance
        }
        
        # Return true if head is tilted down based on multiple criteria
        # Must meet at least 2 of the 3 conditions to reduce false positives
        conditions_met = 0
        if absolute_distance > HEAD_DOWN_THRESH:
            conditions_met += 1
        if vertical_ratio > 30:  # Percentage of face height
            conditions_met += 1
        if angle > HEAD_DOWN_ANGLE_THRESH:
            conditions_met += 1
            
        return conditions_met >= 2

    def process_landmarks(self, frame, face_rect):
        # Get facial landmarks
        shape = self.predictor(frame, face_rect)
        landmarks = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])
        
        # Get eye landmarks
        left_eye = np.array([landmarks[i] for i in self.left_eye_indices])
        right_eye = np.array([landmarks[i] for i in self.right_eye_indices])
        
        # Calculate eye aspect ratio
        left_ear = self.calculate_eye_aspect_ratio(left_eye)
        right_ear = self.calculate_eye_aspect_ratio(right_eye)
        
        # Average the eye aspect ratio
        ear = (left_ear + right_ear) / 2.0
        
        # Store current ear for external access
        self.current_ear = ear
        
        # Check if eyes are closed (partially)
        if ear < EYE_AR_THRESH:
            self.eye_closed_counter += 1
        else:
            self.eye_closed_counter = 0
            
        # Check if eyes are completely closed
        if ear < EYES_CLOSED_THRESH:
            self.eyes_completely_closed_counter += 1
            if self.eyes_completely_closed_counter >= EYES_CLOSED_CONSEC_FRAMES:
                self.eyes_closed = True
        else:
            self.eyes_completely_closed_counter = 0
            self.eyes_closed = False
        
        # Check for yawning
        mouth = np.array([landmarks[i] for i in self.mouth_indices])
        if self.detect_yawn(mouth):
            self.yawn_counter += 1
            self.yawning = True
        else:
            self.yawn_counter = 0
            self.yawning = False
        
        # Check for head down
        if self.detect_head_down(landmarks):
            self.head_down_counter += 1
            if self.head_down_counter >= HEAD_DOWN_CONSEC_FRAMES:
                self.head_down = True
        else:
            self.head_down_counter = 0
            self.head_down = False
        
        # Determine if drowsy (prioritize alerts)
        if self.eye_closed_counter >= EYE_AR_CONSEC_FRAMES:
            self.drowsy = True
            self.send_alert_to_arduino("drowsy")
        elif self.eyes_closed:
            self.send_alert_to_arduino("eyes_closed")
        elif self.head_down:
            self.send_alert_to_arduino("head_down")
        elif self.yawning:
            self.send_alert_to_arduino("yawn")
        else:
            self.drowsy = False
            self.send_alert_to_arduino("normal")
        
        # Draw landmarks and status on frame
        self.draw_landmarks(frame, left_eye, right_eye, mouth)
        self.display_status(frame, ear)
        
        return frame

    def draw_landmarks(self, frame, left_eye, right_eye, mouth):
        # Draw eyes with color based on status
        left_ear = self.calculate_eye_aspect_ratio(left_eye)
        right_ear = self.calculate_eye_aspect_ratio(right_eye)
        
        # Determine eye colors based on openness
        def get_eye_color(ear_value):
            if ear_value < EYES_CLOSED_THRESH:
                return (0, 0, 255)  # Red for closed
            elif ear_value < EYE_AR_THRESH:
                return (0, 165, 255)  # Orange for partially closed
            else:
                return (0, 255, 0)  # Green for open
        
        left_eye_color = get_eye_color(left_ear)
        right_eye_color = get_eye_color(right_ear)
        
        # Draw eyes with color indicator
        cv2.polylines(frame, [left_eye.astype(np.int32)], True, left_eye_color, 2)
        cv2.polylines(frame, [right_eye.astype(np.int32)], True, right_eye_color, 2)
        
        # Label the eye areas
        left_center = np.mean(left_eye, axis=0).astype(np.int32)
        right_center = np.mean(right_eye, axis=0).astype(np.int32)
        
        cv2.putText(frame, "L", (left_center[0]-10, left_center[1]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, left_eye_color, 1)
        cv2.putText(frame, "R", (right_center[0]-10, right_center[1]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, right_eye_color, 1)
        
        # Draw mouth
        cv2.polylines(frame, [mouth.astype(np.int32)], True, (0, 255, 0), 1)

    def display_status(self, frame, ear):
        # Display EAR value with clear label
        cv2.putText(frame, f"Eye Aspect Ratio: {ear:.2f}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Show eye status visually
        status_text = "EYES: "
        if ear < EYES_CLOSED_THRESH:
            status_text += "CLOSED"
            status_color = (0, 0, 255)  # Red
        elif ear < EYE_AR_THRESH:
            status_text += "PARTIALLY CLOSED"
            status_color = (0, 165, 255)  # Orange
        else:
            status_text += "OPEN"
            status_color = (0, 255, 0)  # Green
            
        cv2.putText(frame, status_text, (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Display head position metrics if available
        if hasattr(self, 'head_metrics'):
            metrics = self.head_metrics
            cv2.putText(frame, f"Head Angle: {metrics['angle']:.1f}°", (10, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(frame, f"Vert Ratio: {metrics['vertical_ratio']:.1f}%", (10, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Display drowsiness status
        if self.drowsy:
            cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display eyes closed status
        if self.eyes_closed:
            cv2.putText(frame, "EYES CLOSED ALERT!", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display yawning status
        if self.yawning:
            cv2.putText(frame, "YAWNING DETECTED", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display head down status
        if self.head_down:
            cv2.putText(frame, "HEAD DOWN DETECTED", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def run(self):
        # Start video capture
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open video stream.")
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture video frame.")
                break
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.detector(gray, 0)
            
            if len(faces) > 0:
                # Process the first face detected
                processed_frame = self.process_landmarks(frame, faces[0])
            else:
                # No face detected
                cv2.putText(frame, "No Face Detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                processed_frame = frame
            
            # Display the frame
            cv2.imshow('Drowsiness Detection', processed_frame)
            
            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        if self.arduino_connected:
            self.arduino.close()

def main():
    parser = argparse.ArgumentParser(description="Drowsiness Detection System")
    parser.add_argument('--port', type=str, default=ARDUINO_PORT, 
                        help='Arduino serial port (default: COM3)')
    parser.add_argument('--baud', type=int, default=ARDUINO_BAUD_RATE, 
                        help='Arduino baud rate (default: 9600)')
    args = parser.parse_args()
    
    detector = DrowsinessDetector(arduino_port=args.port, arduino_baud_rate=args.baud)
    detector.run()

if __name__ == "__main__":
    main() 