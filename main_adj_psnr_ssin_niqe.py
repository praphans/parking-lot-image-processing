import cv2
import cvzone
from ultralytics import YOLO
from skimage.metrics import structural_similarity as ssim
import numpy as np
import os

# Load the YOLO model
model = YOLO('yolov8s.pt')

# Define parameters for adjustment
brightness_value = 5
contrast_value = 15
noise_value = 2
is_adj = 1  # 0 = disable adjustment, 1 = enable adjustment

# Define image path
image_path = 'images/rainfall/2012-12-11_14_56_07_jpg.rf.6df322de34acc6e2d02cb1140af3175f.jpg'
frame = cv2.imread(image_path)  # Normal parking lot image

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


# Apply preprocessing if enabled
if is_adj == 1:
    # Step 1: Reduce noise first
    frame_preprocessed = reduce_noise(frame, noise_value)

    # Step 2: Adjust brightness and contrast after noise reduction
    frame_preprocessed = adjust_brightness_contrast(frame_preprocessed, brightness_value, contrast_value)
else:
    # No adjustments, use the original frame
    frame_preprocessed = frame.copy()

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

# Print the number of cars detected
print(f"Car with preprocess: {num_cars_with_preprocess}")
print(f"PSNR: {psnr_value}")
print(f"SSIM: {ssim_value}")

# Display car count, PSNR, and SSIM on the image
cvzone.putTextRect(frame_preprocessed, f'Car with preprocess: {num_cars_with_preprocess}', (50, 60), 2, 2)
cvzone.putTextRect(frame_preprocessed, f'PSNR: {psnr_value:.2f}', (50, 120), 2, 2)
cvzone.putTextRect(frame_preprocessed, f'SSIM: {ssim_value:.4f}', (50, 180), 2, 2)

# Display the processed image
cv2.imshow('Processed Image', frame_preprocessed)
cv2.waitKey(0)
cv2.destroyAllWindows()

