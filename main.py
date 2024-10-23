import cv2
import cvzone
from ultralytics import YOLO
from brisque import BRISQUE
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Load the YOLO model / โหลดโมเดล YOLO
model = YOLO('yolov8s.pt')

# Read the image / อ่านภาพ
frame = cv2.imread('images/rainfall/2012-12-11_14_56_07_jpg.rf.6df322de34acc6e2d02cb1140af3175f.jpg')

# Initialize BRISQUE evaluator / เริ่มต้นตัวประเมินค่า BRISQUE
brisque_evaluator = BRISQUE()

# Car class ID for YOLO / รหัสคลาสรถสำหรับ YOLO
car_class_id = 2  # Use 2 for 'car' in COCO dataset / ใช้ 2 สำหรับ 'car' ในชุดข้อมูล COCO

# Function to adjust brightness and contrast / ฟังก์ชันปรับความสว่างและความคมชัด
def adjust_brightness_contrast(image, brightness=0, contrast=0):
    if brightness != 0:
        shadow = max(0, brightness)
        highlight = min(255, 255 + brightness) if brightness < 0 else 255
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

# Function to reduce noise / ฟังก์ชันลดเสียงรบกวน
def reduce_noise(image, noise_value):
    return cv2.fastNlMeansDenoisingColored(image, None, noise_value, noise_value, 7, 21)

# Detect objects in the original image (without preprocessing) / ตรวจจับวัตถุในภาพต้นฉบับ (ไม่ใช้การประมวลผลล่วงหน้า)
results_without_preprocess = model(frame)
car_boxes_without_preprocess = [det.xyxy.numpy() for result in results_without_preprocess for det in result.boxes if int(det.cls) == car_class_id]
num_cars_without_preprocess = len(car_boxes_without_preprocess)

# Measure initial BRISQUE value for the original image / วัดค่า BRISQUE เริ่มต้นสำหรับภาพต้นฉบับ
default_brisque = brisque_evaluator.score(frame)
default_car = num_cars_without_preprocess

# Define ranges for parameters / กำหนดช่วงของพารามิเตอร์
brightness_range = range(0, 101, 10)
contrast_range = range(0, 101, 10)
noise_range = range(0, 11, 1)

# Function to calculate BRISQUE for each combination of parameters / ฟังก์ชันคำนวณค่า BRISQUE สำหรับแต่ละการรวมกันของพารามิเตอร์
def calculate_brisque_params(brightness_value, contrast_value, noise_value):
    frame_preprocessed = adjust_brightness_contrast(frame, brightness_value, contrast_value)
    frame_preprocessed = reduce_noise(frame_preprocessed, noise_value)
    brisque_value = brisque_evaluator.score(frame_preprocessed)
    return (brightness_value, contrast_value, noise_value, brisque_value)

# Parallel execution to speed up BRISQUE calculations / การประมวลผลแบบคู่ขนานเพื่อเพิ่มความเร็วในการคำนวณ BRISQUE
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(calculate_brisque_params, b, c, n) for b in brightness_range for c in contrast_range for n in noise_range]
    brisque_results = [future.result() for future in futures]

# Sort the results by BRISQUE (lowest to highest) / จัดเรียงผลลัพธ์ตามค่า BRISQUE (จากน้อยไปมาก)
brisque_results.sort(key=lambda x: x[3], reverse=True)

# Apply the best BRISQUE parameters and detect cars / ใช้พารามิเตอร์ BRISQUE ที่ดีที่สุดและตรวจจับรถยนต์
for i in range(len(brisque_results)):
    brightness_value, contrast_value, noise_value, best_brisque = brisque_results[i]
    
    # Adjust the image using the best BRISQUE parameters / ปรับภาพโดยใช้พารามิเตอร์ BRISQUE ที่ดีที่สุด
    frame_preprocessed = adjust_brightness_contrast(frame, brightness_value, contrast_value)
    frame_preprocessed = reduce_noise(frame_preprocessed, noise_value)

    # Detect cars in the adjusted image / ตรวจจับรถยนต์ในภาพที่ปรับแล้ว
    results_with_preprocess = model(frame_preprocessed)
    car_boxes_with_preprocess = [det.xyxy.numpy() for result in results_with_preprocess for det in result.boxes if int(det.cls) == car_class_id]
    num_cars_with_preprocess = len(car_boxes_with_preprocess)

    # Compare the detected cars with default_car / เปรียบเทียบจำนวนรถที่ตรวจจับได้กับ default_car
    if num_cars_with_preprocess > default_car:
        print(f"Detected more cars: {num_cars_with_preprocess} with BRISQUE {best_brisque}")
        break
    elif i == len(brisque_results) - 1:
        print(f"Best cars detected: {num_cars_with_preprocess} with BRISQUE {best_brisque}")

# Draw bounding boxes around cars in the final image / วาดกล่องรอบรถยนต์ในภาพสุดท้าย
for box in car_boxes_with_preprocess:
    x1, y1, x2, y2 = map(int, box[0])
    cv2.rectangle(frame_preprocessed, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Display car count and BRISQUE score on the image / แสดงจำนวนรถยนต์และคะแนน BRISQUE บนภาพ
cvzone.putTextRect(frame_preprocessed, f'Car without preprocess: {num_cars_without_preprocess}', (50, 60), 2, 2)
cvzone.putTextRect(frame_preprocessed, f'Car with preprocess: {num_cars_with_preprocess}', (50, 110), 2, 2)
cvzone.putTextRect(frame_preprocessed, f'BRISQUE: {best_brisque:.2f}', (50, 160), 2, 2)

# Display the image with bounding boxes and labels / แสดงภาพที่มีกล่องรอบรถและป้ายกำกับ
cv2.imshow('Image with cars', frame_preprocessed)
cv2.waitKey(0)
cv2.destroyAllWindows()
