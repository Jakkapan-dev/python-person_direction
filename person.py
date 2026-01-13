# ระบบตรวจจับบุคคลด้วย YOLOv5 และส่งแจ้งเตือนผ่าน Telegram

# TOKEN 8264928996:AAGfNOVkkKTGmPw7R_QHKmWe-vbCyffe9qE
# CHAT ID 8180688206

from datetime import datetime
import cv2
import torch
import time
import requests

# ================== Telegram ==================
TOKEN = "8264928996:AAGfNOVkkKTGmPw7R_QHKmWe-vbCyffe9qE"
CHAT_ID = "8180688206"

def send_telegram(image):
    url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
    
    now = datetime.now()
    time_srt = now.strftime("%Y-%m-%d %H:%M:%S")
      
    caption = (
        "ตรวจพบบุคคล\n\n"
        f" วันที่และเวลา:\n{time_srt}"
    )
    _, img = cv2.imencode(".jpg", image)
    files = {"photo": img.tobytes()}
    data = {"chat_id": CHAT_ID}
    requests.post(url, files=files, data=data)

# ================== Load YOLOv5 ==================
model = torch.hub.load(
    'ultralytics/yolov5',
    'yolov5n',
    pretrained=True
)

model.conf = 0.4        # confidence threshold
model.classes = [0]     # detect only person

# ================== Video Source ==================
# cap = cv2.VideoCapture(0)          # กล้อง
cap = cv2.VideoCapture(0)  # วิดีโอ

last_alert = 0
ALERT_INTERVAL = 30  # วินาที

print("🔍 System started...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results.xyxy[0]

    person_detected = False

    for *box, conf, cls in detections:
        if int(cls) == 0:
            person_detected = True

            x1, y1, x2, y2 = map(int, box)

            # วาดกรอบ
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (0, 255, 0), 2)

            # แสดงข้อความ
            cv2.putText(frame, "person",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

    # ================== Telegram Alert ==================
    if person_detected:
        now = time.time()
        if now - last_alert > ALERT_INTERVAL:
            send_telegram(frame)
            last_alert = now
            print("📨 Telegram alert sent")

    cv2.imshow("YOLOv5 Person Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('c'):
        break

cap.release()
cv2.destroyAllWindows()
