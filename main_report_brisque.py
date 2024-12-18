import cv2
import cvzone
from ultralytics import YOLO
import numpy as np
import pandas as pd
import os
from brisque import BRISQUE  # นำเข้า BRISQUE

# Load the YOLO model
model = YOLO('yolov8s.pt')

# List of image paths and their corresponding output filenames
images = [
    ('images/rainfall/2012-12-07_16_42_25_jpg.rf.b3ab0f1190cf376d9f536302e8a4d202.jpg', 'Resault/report_v3/report_real_img01.csv'),
    ('images/rainfall/2012-12-11_14_56_07_jpg.rf.6df322de34acc6e2d02cb1140af3175f.jpg', 'Resault/report_v3/report_real_img02.csv'),
    ('images/rainfall/2013-01-21_08_15_03_jpg.rf.628c99c37665079f97e80afc3a1b4c7d.jpg', 'Resault/report_v3/report_real_img03.csv'),
    ('images/rainfall/2013-01-21_08_40_04_jpg.rf.f8fe78a6337f16cffee20e3fb8e27040.jpg', 'Resault/report_v3/report_real_img04.csv'),
    ('images/rainfall/2013-01-21_08_45_04_jpg.rf.63251624038faf503f6279622979a590.jpg', 'Resault/report_v3/report_real_img05.csv'),
]

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

# Loop through all images
for image_path, output_path in images:
    frame = cv2.imread(image_path)  # Load the current image
    results = []  # Prepare a list to store results for CSV export

    # Create BRISQUE object
    brisque = BRISQUE()

    # Loop through all combinations of brightness, contrast, and noise
    for brightness_value in range(0, 101, 5):  # 0 to 100 with step 5
        for contrast_value in range(0, 101, 5):  # 0 to 100 with step 5
            for noise_value in np.arange(0, 10.5, 0.5):  # 0 to 10.5 with step 0.5
                # Step 1: Reduce noise first
                frame_preprocessed = reduce_noise(frame, noise_value)

                # Step 2: Adjust brightness and contrast after noise reduction
                frame_preprocessed = adjust_brightness_contrast(frame_preprocessed, brightness_value, contrast_value)

                # Calculate BRISQUE score using the BRISQUE object
                brisque_score = brisque.score(frame_preprocessed)

                # Detect objects in the preprocessed image
                results_with_preprocess = model(frame_preprocessed)
                car_class_id = class_list.index('car')
                car_boxes_with_preprocess = [det.xyxy.cpu().numpy() for result in results_with_preprocess for det in result.boxes if int(det.cls) == car_class_id]
                num_cars_with_preprocess = len(car_boxes_with_preprocess)

                # Prepare the data for CSV export
                data = {
                    'Image': os.path.basename(image_path),  # Just the file name, not in a list
                    'Cars': num_cars_with_preprocess,        # Not in a list
                    'BRISQUE': brisque_score,                # บันทึกค่า BRISQUE
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
