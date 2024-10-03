import cv2
import cvzone
from ultralytics import YOLO
from skimage.metrics import structural_similarity as ssim
import numpy as np
import pandas as pd
import os

# Load the YOLO model
model = YOLO('yolov8s.pt')

# Define image path
image_path = 'images/rainfall/2012-12-21_18_30_15_jpg.rf.823ad3e87780ff3fb214468b44e23c8c.jpg'
frame = cv2.imread(image_path)  # Normal parking lot image

# Define the output path
output_path = 'Resault/report/report_real_02.csv'

# Function to adjust brightness and contrast
def adjust_brightness_contrast(image, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        buf = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)
    else:
        buf = image.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

# Function to reduce noise
def reduce_noise(image, noise_value):
    return cv2.fastNlMeansDenoisingColored(image, None, noise_value, noise_value, 7, 21)

# Read "coco.txt" file and split the data into a list of classes
with open("coco.txt", "r") as my_file:
    data = my_file.read()
    class_list = data.split("\n")

# Prepare a list to store results for CSV export
results = []

# Loop through all combinations of brightness, contrast, and noise
for brightness_value in range(0, 101, 5):  # 0 to 100 with step 5
    for contrast_value in range(0, 101, 5):  # 0 to 100 with step 5
        for noise_value in np.arange(0, 21, 0.5):  # 0 to 20 with step 0.5
            # Step 1: Reduce noise first
            frame_preprocessed = reduce_noise(frame, noise_value)

            # Step 2: Adjust brightness and contrast after noise reduction
            frame_preprocessed = adjust_brightness_contrast(frame_preprocessed, brightness_value, contrast_value)

            # Calculate PSNR
            psnr_value = cv2.PSNR(frame, frame_preprocessed)

            # Convert images to grayscale for SSIM calculation
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame_preprocessed = cv2.cvtColor(frame_preprocessed, cv2.COLOR_BGR2GRAY)

            # Calculate SSIM
            ssim_value, _ = ssim(gray_frame, gray_frame_preprocessed, full=True)

            # Detect objects in the preprocessed image
            results_with_preprocess = model(frame_preprocessed)
            car_class_id = class_list.index('car')
            car_boxes_with_preprocess = [det.xyxy.numpy() for result in results_with_preprocess for det in result.boxes if int(det.cls) == car_class_id]
            num_cars_with_preprocess = len(car_boxes_with_preprocess)

            # Prepare the data for CSV export
            data = {
                'Image': os.path.basename(image_path),  # Just the file name, not in a list
                'Cars': num_cars_with_preprocess,        # Not in a list
                'PSNR': psnr_value,                      # Not in a list
                'SSIM': ssim_value,                      # Not in a list
                'Brightness': brightness_value,          # Not in a list
                'Contrast': contrast_value,              # Not in a list
                'Noise': noise_value                     # Not in a list
            }
            results.append(data)

# Create a DataFrame from all results
df = pd.DataFrame(results)

# Create directory if it doesn't exist
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Export the DataFrame to CSV
df.to_csv(output_path, index=False)

print(f"Data successfully saved to {output_path}")
