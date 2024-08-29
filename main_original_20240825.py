import cv2
import cvzone
from ultralytics import YOLO

# Define parameters for adjustment
brightness_value = 50
contrast_value = 25
noise_value = 2

# Load the YOLO model
model = YOLO('yolov8s.pt')

# Read the image
frame = cv2.imread('images/rainfall/2013-01-21_08_45_04_jpg.rf.63251624038faf503f6279622979a590.jpg')  # Normal parking lot image

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

# Adjust the brightness and contrast of the image
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
