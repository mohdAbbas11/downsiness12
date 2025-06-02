#!/usr/bin/env python3
import subprocess
import sys
import os
import platform

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)

def install_requirements():
    """Install required packages from requirements.txt."""
    print("Installing required packages...")
    
    # Special handling for dlib on Windows
    is_windows = platform.system() == "Windows"
    
    try:
        # First install packages except dlib
        if is_windows:
            print("Windows detected, using special installation method for dlib...")
            # Install everything except dlib first
            with open("requirements.txt", "r") as f:
                requirements = f.read().splitlines()
            
            non_dlib_requirements = [req for req in requirements if not req.strip().startswith("dlib")]
            if non_dlib_requirements:
                subprocess.check_call([sys.executable, "-m", "pip", "install"] + non_dlib_requirements)
            
            # Install dlib from a pre-built wheel
            print("Installing pre-built dlib wheel...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "https://github.com/jloh02/dlib/releases/download/v19.22/dlib-19.22.99-cp310-cp310-win_amd64.whl"
            ])
        else:
            # For non-Windows systems, install normally
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        print("Successfully installed required packages")
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to install required packages: {str(e)}")
        sys.exit(1)

def download_model():
    """Download the facial landmark model."""
    print("Downloading facial landmark model...")
    try:
        subprocess.check_call([sys.executable, "download_model.py"])
        print("Successfully downloaded the facial landmark model")
    except subprocess.CalledProcessError:
        print("Error: Failed to download the facial landmark model")
        sys.exit(1)

def print_arduino_instructions():
    """Print instructions for Arduino setup."""
    print("\n" + "="*80)
    print("Arduino Setup Instructions:")
    print("="*80)
    print("1. Open the Arduino IDE")
    print("2. Open the file 'arduino_drowsiness_alert.ino'")
    print("3. Connect your Arduino Uno to your computer")
    print("4. Select the correct board and port in the Arduino IDE")
    print("5. Upload the sketch to your Arduino")
    print("6. Connect the components according to the circuit diagram in 'arduino_circuit_setup.txt'")
    
    # Show the detected COM ports
    print("\nDetected serial ports:")
    try:
        import serial.tools.list_ports
        ports = list(serial.tools.list_ports.comports())
        for port in ports:
            print(f"  - {port.device}: {port.description}")
    except:
        print("  (Unable to detect serial ports)")
    
    print("\nNote the COM port of your Arduino for use with the drowsiness detection script")

def main():
    """Main setup function."""
    print("Drowsiness Detection System Setup")
    print("="*40)
    
    # Check Python version
    check_python_version()
    
    # Install requirements
    install_requirements()
    
    # Download the facial landmark model
    download_model()
    
    # Print Arduino setup instructions
    print_arduino_instructions()
    
    print("\nSetup complete! You can now run the drowsiness detection system:")
    print("python drowsiness_detection.py --port YOUR_COM_PORT --baud 9600")

if __name__ == "__main__":
    main() 