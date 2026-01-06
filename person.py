import cv2
import torch
import time
import os
import requests

# ================= Telegram =================
BOT_TOKEN = "YOUR_BOT_TOKEN"
CHAT_ID = "YOUR_CHAT_ID"

def send_telegram(image_path, message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    with open(image_path, "rb") as img:
        requests.post(
            url,
            data={"chat_id": CHAT_ID, "caption": message},
            files={"photo": img}
        )

# ================= YOLOv5 =================
yolo = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
yolo.conf = 0.5

# ================= HOG =================
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# ================= Camera =================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("เปิดกล้องไม่ได้")
    exit()

# ================= Variables =================
person_timer = {}
SAVE_DIR = "intruder_images"
os.makedirs(SAVE_DIR, exist_ok=True)

print("Hybrid Intrusion Detection Started (HOG + YOLOv5)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (640, 480))

    # ---------- STEP 1: HOG (เร็ว) ----------
    boxes, weights = hog.detectMultiScale(
        frame_resized,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05
    )

    if len(boxes) > 0:
        # ---------- STEP 2: YOLOv5 (ยืนยัน) ----------
        results = yolo(frame_resized)
        detections = results.xyxy[0]

        current_time = time.time()

        for *box, conf, cls in detections:
            if int(cls) == 0:  # person
                x1, y1, x2, y2 = map(int, box)

                person_id = f"{x1}_{y1}"

                if person_id not in person_timer:
                    person_timer[person_id] = current_time

                duration = current_time - person_timer[person_id]

                # ---------- วิเคราะห์พฤติกรรม ----------
                if duration > 10:  # อยู่เกิน 10 วินาที
                    img_path = f"{SAVE_DIR}/intruder_{int(time.time())}.jpg"
                    cv2.imwrite(img_path, frame_resized)

                    send_telegram(
                        img_path,
                        f"⚠️ Intruder Detected\nDuration: {int(duration)} seconds"
                    )

                    person_timer.pop(person_id)
                    break

                # ---------- วาดผล ----------
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    frame_resized,
                    f"Suspicious: {int(duration)}s",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

    else:
        person_timer.clear()

    cv2.imshow("Hybrid Intrusion Detection", frame_resized)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
