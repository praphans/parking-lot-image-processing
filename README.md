
# Parking Lot Image Processing

## Overview

The **Parking Lot Image Processing** project utilizes YOLOv8s with Python and OpenCV to enhance image quality and improve vehicle detection in challenging weather conditions, such as heavy rain. This project focuses on preprocessing techniques to optimize image clarity and accuracy for better detection of vehicles in parking lots.

## Features

- **Brightness Adjustment**: This technique modifies the overall brightness of the image to enhance visibility in low-light conditions.
- **Contrast Adjustment**: Increases the contrast between pixels to make objects more distinguishable and improve overall image clarity.
- **Noise Reduction**: Removes unwanted noise from the image, which is crucial for maintaining image quality and ensuring accurate detection even in adverse weather conditions.

## Libraries Required

- `cv2` (OpenCV): Used for various image processing operations.
- `ultralytics` (YOLOv8s): Utilized for vehicle detection and counting.

## Installation

To get started, you need to install the required libraries. You can install them using pip:

```bash
pip install opencv-python ultralytics
```

## Usage

1. **Image Preprocessing**: Apply the brightness and contrast adjustments and noise reduction techniques to your images to prepare them for detection.
2. **Vehicle Detection**: Use YOLOv8s to analyze the preprocessed images and detect vehicles.

This project is designed to handle the complexities of image processing in adverse weather, ensuring reliable vehicle detection and counting in parking lots.

# Demo Screenshots
![alt text](https://github.com/praphanth/parking-lot-image-processing/blob/master/img-demo.png?raw=true)