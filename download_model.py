#!/usr/bin/env python3
"""
Download script for the dlib facial landmark predictor model.
This script downloads the pre-trained model file required for facial landmark detection.
"""

import os
import sys
import requests
import bz2
import gdown

def download_with_gdown():
    """Download the model using gdown (Google Drive)"""
    print("Downloading shape_predictor_68_face_landmarks.dat using gdown...")
    
    # Google Drive file ID for the model
    file_id = "1Xg08olCgQ_wKhfGr9I_0_X7CPXS8q0qf"
    
    try:
        gdown.download(id=file_id, output="shape_predictor_68_face_landmarks.dat.zip", quiet=False)
        print("Download complete. Extracting...")
        
        import zipfile
        with zipfile.ZipFile("shape_predictor_68_face_landmarks.dat.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
        
        # Remove the zip file
        os.remove("shape_predictor_68_face_landmarks.dat.zip")
        print("Extraction complete. Model is ready to use.")
        return True
    except Exception as e:
        print(f"Error downloading with gdown: {e}")
        return False

def download_from_dlib():
    """Download the model from dlib's GitHub repository"""
    print("Downloading shape_predictor_68_face_landmarks.dat from dlib repository...")
    
    url = "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2"
    
    try:
        # Download the compressed file
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Save the compressed file
        with open("shape_predictor_68_face_landmarks.dat.bz2", "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print("Download complete. Extracting...")
        
        # Extract the bz2 file
        with open("shape_predictor_68_face_landmarks.dat", "wb") as new_file:
            with open("shape_predictor_68_face_landmarks.dat.bz2", "rb") as file:
                decompressor = bz2.BZ2Decompressor()
                for data in iter(lambda: file.read(100 * 1024), b''):
                    new_file.write(decompressor.decompress(data))
        
        # Remove the compressed file
        os.remove("shape_predictor_68_face_landmarks.dat.bz2")
        print("Extraction complete. Model is ready to use.")
        return True
    except Exception as e:
        print(f"Error downloading from dlib: {e}")
        return False

def main():
    """Main function to download the model"""
    # Check if the model already exists
    if os.path.exists("shape_predictor_68_face_landmarks.dat"):
        print("Model file already exists. Skipping download.")
        return
    
    # Try downloading with gdown first
    try:
        import gdown
        if download_with_gdown():
            return
    except ImportError:
        print("gdown not installed. Trying alternative download method...")
    
    # If gdown fails or is not installed, try downloading from dlib
    if download_from_dlib():
        return
    
    # If all methods fail
    print("Failed to download the model. Please download it manually from:")
    print("https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2")
    print("Extract it and place it in the same directory as this script.")
    sys.exit(1)

if __name__ == "__main__":
    main() 