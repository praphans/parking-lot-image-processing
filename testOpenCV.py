import cv2
from ultralytics import YOLO

# กำหนดพารามิเตอร์สำหรับการปรับแต่ง
brightness_value = 50    # ค่า brightness #ค่าปกติ = 0
contrast_value = 25    # ค่า contrast #ค่าปกติ = 1
noise_value = 2        # ค่า noise reduction #ค่าปกติ = 0

# โหลดโมเดล YOLO
model = YOLO('yolov8s.pt')
#model = YOLO('runs/detect/train/weights/best.pt')


# อ่านรูปภาพ
frame = cv2.imread('note/train/2012-12-11_14_56_07_jpg.rf.6df322de34acc6e2d02cb1140af3175f.jpg')  # ภาพลาดจอด pklot ปกติ

# ฟังก์ชันสำหรับการปรับ Brightness และ Contrast
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

# ฟังก์ชันสำหรับการลด Noise
def reduce_noise(image, noise_value):
    return cv2.fastNlMeansDenoisingColored(image, None, noise_value, noise_value, 7, 21)

# อ่านไฟล์ "coco.txt" และแบ่งข้อมูลออกเป็นรายการของคลาส
with open("coco.txt", "r") as my_file:
    data = my_file.read()
    class_list = data.split("\n")


# ปรับ Brightness และ Contrast ของภาพ
frame = adjust_brightness_contrast(frame, brightness_value, contrast_value)

# ลด Noise ของภาพ
frame = reduce_noise(frame, noise_value)

# ตรวจจับวัตถุในภาพ
results = model(frame)

# ดึงข้อมูลการตรวจจับที่เป็นรถยนต์
car_class_id = class_list.index('car')

# ฟังก์ชันสำหรับการดึงกรอบรอบวัตถุ
def get_boxes(results, class_id):
    boxes = []
    for result in results:
        for det in result.boxes:
            if int(det.cls) == class_id:
                boxes.append(det.xyxy.numpy())
    return boxes

# ดึงกรอบรอบรถยนต์
car_boxes = get_boxes(results, car_class_id)

# นับจำนวนรถยนต์
num_cars = len(car_boxes)

# แสดงจำนวนรถยนต์
print(f"Number of cars detected: {num_cars}")

# วาดกรอบรอบๆ รถยนต์ในภาพ
for box in car_boxes:
    x1, y1, x2, y2 = map(int, box[0])
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

# แสดงภาพที่มีการวาดกรอบ
cv2.imshow('Image with cars', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
