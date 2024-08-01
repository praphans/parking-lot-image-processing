import cv2
from ultralytics import YOLO

# Define parameters for adjustment
# Brightness default = 0
# Contrast default = 1
# Noise reduction default = 0

brightness_value = 10 
contrast_value = 5  
noise_value = 0.5

# Load the YOLO model
model = YOLO('yolov8s.pt')
# model = YOLO('runs/detect/train/weights/best.pt')

# Read the image
frame = cv2.imread('images/2012-12-11_14_56_07_jpg.rf.6df322de34acc6e2d02cb1140af3175f.jpg')  # Normal parking lot image
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

# Adjust the brightness and contrast of the image
frame = adjust_brightness_contrast(frame, brightness_value, contrast_value)

# Reduce noise in the image
frame = reduce_noise(frame, noise_value)

# Detect objects in the image
results = model(frame)

# Extract the class ID for cars
car_class_id = class_list.index('car')

# Function to extract bounding boxes around objects
def get_boxes(results, class_id):
    boxes = []
    for result in results:
        for det in result.boxes:
            if int(det.cls) == class_id:
                boxes.append(det.xyxy.numpy())
    return boxes

# Get bounding boxes for cars
car_boxes = get_boxes(results, car_class_id)

# Count the number of cars
num_cars = len(car_boxes)

# Print the number of cars detected
print(f"Number of cars detected: {num_cars}")

# Draw bounding boxes around cars in the image
for box in car_boxes:
    x1, y1, x2, y2 = map(int, box[0])
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Display the image with bounding boxes
cv2.imshow('Image with cars', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
