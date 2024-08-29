import cv2
import numpy as np
import cvzone
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('yolov8s.pt')

# Read the image
frame = cv2.imread('images/2012-12-21_18_30_15_jpg.rf.823ad3e87780ff3fb214468b44e23c8c.jpg')  # Normal parking lot image

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

# Function to compute SSIM and PSNR
def compute_ssim_psnr(original_image, processed_image):
    original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    ssim_index = ssim(original_gray, processed_gray)
    psnr_value = psnr(original_gray, processed_gray)
    return ssim_index, psnr_value

# Function to find optimal parameters
def find_optimal_params(image, model):
    best_params = {'brightness': 0, 'contrast': 0, 'noise': 0}
    best_num_cars = 0
    best_ssim = 0
    best_psnr = 0

    # Calculate parameter ranges based on image statistics
    mean = np.mean(image)
    std_dev = np.std(image)
    
    brightness_range = np.linspace(-std_dev, std_dev, num=5)
    contrast_range = np.linspace(-std_dev, std_dev, num=5)
    noise_range = np.linspace(1, 3, num=3)

    for brightness in brightness_range:
        for contrast in contrast_range:
            for noise in noise_range:
                adjusted_image = adjust_brightness_contrast(image, brightness, contrast)
                adjusted_image = reduce_noise(adjusted_image, noise)
                ssim_index, psnr_value = compute_ssim_psnr(image, adjusted_image)
                results = model(adjusted_image)
                car_class_id = class_list.index('car')
                car_boxes = [det.xyxy.numpy() for result in results for det in result.boxes if int(det.cls) == car_class_id]
                num_cars = len(car_boxes)

                if num_cars > best_num_cars and ssim_index > best_ssim and psnr_value > best_psnr:
                    best_num_cars = num_cars
                    best_ssim = ssim_index
                    best_psnr = psnr_value
                    best_params = {'brightness': brightness, 'contrast': contrast, 'noise': noise}

    return best_params

# Read "coco.txt" file and split the data into a list of classes
with open("coco.txt", "r") as my_file:
    data = my_file.read()
    class_list = data.split("\n")

# Detect objects in the original image (without preprocessing)
results_without_preprocess = model(frame)
car_class_id = class_list.index('car')
car_boxes_without_preprocess = [det.xyxy.numpy() for result in results_without_preprocess for det in result.boxes if int(det.cls) == car_class_id]
num_cars_without_preprocess = len(car_boxes_without_preprocess)

# Find optimal parameters for preprocessing
optimal_params = find_optimal_params(frame, model)
brightness_value = optimal_params['brightness']
contrast_value = optimal_params['contrast']
noise_value = optimal_params['noise']

# Adjust the brightness and contrast of the image using optimal parameters
frame_preprocessed = adjust_brightness_contrast(frame, brightness_value, contrast_value)
frame_preprocessed = reduce_noise(frame_preprocessed, noise_value)

# Detect objects in the preprocessed image
results_with_preprocess = model(frame_preprocessed)
car_boxes_with_preprocess = [det.xyxy.numpy() for result in results_with_preprocess for det in result.boxes if int(det.cls) == car_class_id]
num_cars_with_preprocess = len(car_boxes_with_preprocess)

# Print the number of cars detected
print(f"Car without preprocess: {num_cars_without_preprocess}")
print(f"Car with preprocess: {num_cars_with_preprocess}")

# Draw bounding boxes around cars in the image with preprocessing
for box in car_boxes_with_preprocess:
    x1, y1, x2, y2 = map(int, box[0])
    cv2.rectangle(frame_preprocessed, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Display car count on the image
cvzone.putTextRect(frame_preprocessed, f'Car without preprocess: {num_cars_without_preprocess}', (50, 60), 2, 2)
cvzone.putTextRect(frame_preprocessed, f'Car with preprocess: {num_cars_with_preprocess}', (50, 110), 2, 2)

# Display the image with bounding boxes and car count
cv2.imshow('Image with cars', frame_preprocessed)
cv2.waitKey(0)
cv2.destroyAllWindows()
