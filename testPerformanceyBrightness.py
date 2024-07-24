import os
import json
import cv2
from ultralytics import YOLO

# Define parameters for adjustment
brightness_values = [10, 20, 30, 35, 40]
contrast_value = 1  
noise_value = 0

# Load the YOLO model
model = YOLO('yolov8s.pt')
# model = YOLO('runs/detect/train/weights/best.pt')

# Function to adjust brightness and contrast
def adjust_brightness_contrast(image, brightness=0, contrast=0):
    # Brightness
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

    # Contrast
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

# Function to extract bounding boxes around objects
def get_boxes(results, class_id):
    boxes = []
    for result in results:
        for det in result.boxes:
            if int(det.cls) == class_id:
                # ย้ายเทนเซอร์ไปยัง CPU ก่อนแปลงเป็น NumPy array
                boxes.append(det.xyxy.cpu().numpy())
    return boxes

# Get the list of images in the folder
image_folder = 'images'
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

# Array to hold the results
results_dict = {}

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    frame = cv2.imread(image_path)  # Normal parking lot image
    
    # Initialize results for this image
    results_dict[image_file] = []

    # Process results for each brightness value
    for brightness_value in brightness_values:
        # Adjust the brightness and contrast of the image
        frame_adjusted = adjust_brightness_contrast(frame, brightness_value, contrast_value)

        # Reduce noise in the image
        frame_adjusted = reduce_noise(frame_adjusted, noise_value)

        # Detect objects in the image
        results = model(frame_adjusted)

        # Extract the class ID for cars
        car_class_id = class_list.index('car')

        # Get bounding boxes for cars
        car_boxes = get_boxes(results, car_class_id)

        # Count the number of cars
        num_cars = len(car_boxes)

        # Append the result to the array for this image
        results_dict[image_file].append({
            'brightness_value': brightness_value,
            'cars': num_cars
        })

# Convert the results dictionary to JSON format
json_result = json.dumps(results_dict, indent=4)

# Print the JSON result
print(json_result)
