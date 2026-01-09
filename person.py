import cv2
import time
import torch
import requests
from datetime import datetime

# ================== TELEGRAM CONFIG ==================
TELEGRAM_TOKEN = "8301170878:AAH9OiT7kEiKIrBdvc2qh1QogAXPbI-0Z8c"
CHAT_ID = " 8180688206"

def send_telegram(image_path):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    with open(image_path, "rb") as img:
        requests.post(url, data={"chat_id": CHAT_ID}, files={"photo": img})

# ================== LOAD YOLOv5 (CPU) ==================
model = torch.hub.load(
    'ultralytics/yolov5',
    'yolov5n',      # เบา เหมาะกับ CPU
    device='cpu'
)
model.conf = 0.6   # ความมั่นใจขั้นต่ำ
model.classes = [0]

# ================== CAMERA ==================
cap = cv2.VideoCapture(r"C:\Users\IT-Chuethong\Downloads\person_detection_Edit.mp4")
time.sleep(2)


FRAME_SIZE = (640,480)
ret, frame = cap.read()
frame = cv2.resize(frame, FRAME_SIZE)
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

last_alert_time = 0

person_stable_frames = 0
PERSON_CONFIRM_FRAMES = 10

print("🔍 System started...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, FRAME_SIZE)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if prev_gray.shape != gray.shape:
        prev_gray = gray
        continue

    # ================== MOTION DETECTION ==================
    diff = cv2.absdiff(prev_gray, gray)
    thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    motion_detected = False
    for c in contours:
        if cv2.contourArea(c) < 2000:
            continue
        motion_detected = True
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    confirmed_intruder = False

    # ================== YOLO CONFIRM ==================
    if motion_detected:
        results = model(frame)

        for *box, conf, cls in results.xyxy[0]:
            label = model.names[int(cls)]
            if label == "person":
                confirmed_intruder = True
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # ================== ALERT ==================
    if confirmed_intruder:
        cv2.putText(frame, "INTRUDER CONFIRMED!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # ป้องกันแจ้งเตือนถี่เกิน
        if time.time() - last_alert_time > 15:
            filename = f"intruder_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)
            send_telegram(filename)
            last_alert_time = time.time()
    else:
        cv2.putText(frame, "Monitoring...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Intruder Detection System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    prev_gray = gray

cap.release()
cv2.destroyAllWindows()
