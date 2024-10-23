import cv2
import cvzone
from ultralytics import YOLO
from skimage.metrics import structural_similarity as ssim
import numpy as np

# Load the YOLO model
model = YOLO('yolov8s.pt')

# Read the image
frame = cv2.imread('images/rainfall/2012-12-11_14_56_07_jpg.rf.6df322de34acc6e2d02cb1140af3175f.jpg')  # Normal parking lot image

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

# Detect objects in the original image (without preprocessing)
results_without_preprocess = model(frame)
car_class_id = class_list.index('car')
car_boxes_without_preprocess = [det.xyxy.numpy() for result in results_without_preprocess for det in result.boxes if int(det.cls) == car_class_id]
num_cars_without_preprocess = len(car_boxes_without_preprocess)

# Define ranges for parameters
brightness_range = range(0, 101, 10)
contrast_range = range(0, 101, 10)
noise_range = range(0, 11, 1)

# Variables to store the best parameters
best_brightness = 0
best_contrast = 0
best_noise = 0
best_num_cars = 0
best_psnr = 0
best_ssim = 0

# Loop over parameter ranges to find the best combination
for brightness_value in brightness_range:
    for contrast_value in contrast_range:
        for noise_value in noise_range:
            # Adjust the brightness and contrast of the image
            frame_preprocessed = adjust_brightness_contrast(frame, brightness_value, contrast_value)
            frame_preprocessed = reduce_noise(frame_preprocessed, noise_value)

            # Detect objects in the preprocessed image
            results_with_preprocess = model(frame_preprocessed)
            car_boxes_with_preprocess = [det.xyxy.numpy() for result in results_with_preprocess for det in result.boxes if int(det.cls) == car_class_id]
            num_cars_with_preprocess = len(car_boxes_with_preprocess)

            # Calculate SSIM
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_preprocessed_gray = cv2.cvtColor(frame_preprocessed, cv2.COLOR_BGR2GRAY)
            ssim_index = ssim(frame_gray, frame_preprocessed_gray)

            # Calculate PSNR
            psnr_value = cv2.PSNR(frame, frame_preprocessed)

            # Check if this combination is better
            if (num_cars_with_preprocess > best_num_cars or
                (num_cars_with_preprocess == best_num_cars and ssim_index > best_ssim) or
                (num_cars_with_preprocess == best_num_cars and ssim_index == best_ssim and psnr_value > best_psnr)):
                best_brightness = brightness_value
                best_contrast = contrast_value
                best_noise = noise_value
                best_num_cars = num_cars_with_preprocess
                best_ssim = ssim_index
                best_psnr = psnr_value

# Print the best parameters and their results
print(f"Best brightness: {best_brightness}")
print(f"Best contrast: {best_contrast}")
print(f"Best noise: {best_noise}")
print(f"Best number of cars detected: {best_num_cars}")
print(f"Best PSNR: {best_psnr}")
print(f"Best SSIM: {best_ssim}")

# Adjust the brightness and contrast using the best parameters and display the result
frame_preprocessed = adjust_brightness_contrast(frame, best_brightness, best_contrast)
frame_preprocessed = reduce_noise(frame_preprocessed, best_noise)

# Draw bounding boxes around cars in the image with preprocessing
for box in car_boxes_with_preprocess:
    x1, y1, x2, y2 = map(int, box[0])
    cv2.rectangle(frame_preprocessed, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Display car count, SSIM, and PSNR on the image
cvzone.putTextRect(frame_preprocessed, f'Car without preprocess: {num_cars_without_preprocess}', (50, 60), 2, 2)
cvzone.putTextRect(frame_preprocessed, f'Car with preprocess: {best_num_cars}', (50, 110), 2, 2)
cvzone.putTextRect(frame_preprocessed, f'PSNR: {best_psnr:.2f}', (50, 160), 2, 2)
cvzone.putTextRect(frame_preprocessed, f'SSIM: {best_ssim:.4f}', (50, 210), 2, 2)

# Display the image with bounding boxes and labels
cv2.imshow('Image with cars', frame_preprocessed)
cv2.waitKey(0)
cv2.destroyAllWindows()
