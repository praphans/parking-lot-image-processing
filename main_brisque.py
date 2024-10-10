from brisque import BRISQUE

# URL ของภาพ
URL = "https://www.mathworks.com/help/examples/images/win64/CalculateBRISQUEScoreUsingCustomFeatureModelExample_01.png"

# สร้างอ็อบเจกต์ BRISQUE
obj = BRISQUE(url=True)

# คำนวณค่า BRISQUE
brisque_score = obj.score(URL)

# แสดงค่า BRISQUE ที่วัดได้
print(f"BRISQUE Score: {brisque_score:.4f}")
