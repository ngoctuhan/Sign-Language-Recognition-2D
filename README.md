# Sign-Language-Recognition-2D

A real-time sign language recognition system that detects and interprets hand gestures for deaf and mute individuals.

## Overview

This project implements a computer vision system that recognizes hand gestures and translates them into sign language alphabet characters. The system uses a combination of image processing techniques and deep learning to achieve accurate recognition in real-time.

## Technical Implementation

### Key Features

- Region of Interest (ROI) detection for hand area isolation
- Advanced image preprocessing pipeline including:
  - Gaussian noise filtering
  - Image smoothing
  - Automatic Otsu thresholding
- Binary image classification using a pre-trained VGG16 model
- Real-time mapping of classifications to alphabet characters

### Project Structure

- `static/`: Frontend assets (HTML, CSS, JavaScript files)
- `models/`: Pre-trained models (Transfer learning from VGG16)
- `webstreaming.py`: Main application file for web interface implementation

### Technologies Used

- Frontend: HTML, CSS, JavaScript
- Backend: Flask
- Machine Learning: TensorFlow 1.14
- Computer Vision: OpenCV 4.1
- Data Processing: NumPy 1.16

## Usage Instructions

1. Run the Python script and wait for model loading and parameter initialization
2. Access the application interface at `127.0.0.2:5000`
3. Wait a few seconds for the system to capture the background (used for background removal)
4. Perform hand gestures within the designated ROI area
5. Click "Predict" to get word predictions based on recognized alphabet characters

Note: Characters that are consistently recognized for 2 seconds are considered valid inputs.

## Demo

Watch the demonstration video: [SignLanguageRecognition2D Demo](https://www.youtube.com/watch?v=DvuglmDpWIY)
