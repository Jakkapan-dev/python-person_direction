import cv2

# ---------- 1. เตรียมตัวตรวจจับคน ----------
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# ---------- 2. เปิดกล้อง ----------
cap = cv2.VideoCapture(0)   # ถ้ามีกล้องหลายตัวลองเปลี่ยนเป็น 1, 2, ...

if not cap.isOpened():
    print("เปิดกล้องไม่ได้")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ลดขนาดภาพให้เล็กลงหน่อย เพื่อให้ประมวลผลเร็วขึ้น
    frame_resized = cv2.resize(frame, (640, 480))

    # ---------- 3. ตรวจจับคนในภาพ ----------
    # boxes = กรอบที่เจอคน, weights = ความมั่นใจ
    boxes, weights = hog.detectMultiScale(
        frame_resized,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05
    )

    person_count = 0

    for (x, y, w, h), weight in zip(boxes, weights):
        # ตัด detection ที่ไม่มั่นใจออก (ปรับ 0.6 ตามสภาพแสง/กล้อง)
        if weight < 0.6:
            continue

        person_count += 1
        # วาดกรอบรอบคน
        cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # ---------- 4. แสดงผล + สถานะผู้บุกรุก ----------
    if person_count > 0:
        text = f"INTRUDER DETECTED: {person_count}"
        color = (0, 0, 255)
    else:
        text = "No person detected"
        color = (0, 255, 0)

    cv2.putText(
        frame_resized,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

    cv2.imshow("Person Detection", frame_resized)

    # กด q เพื่อออกจากโปรแกรม
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
