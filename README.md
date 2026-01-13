ไลบรารี่ที่ใช้

Python
Python 3.11

ประมวลผลภาพและวิดีโอ
 OpenCV
 pip install opencv-python
pip install opencv-python-headless

NumPy
pip install numpy

Deep Learning / YOLO PyTorch (CPU)
pip install torch torchvision torchaudio

YOLOv5

การสื่อสารและแจ้งเตือน
 Requests
 pip install requests

import time
import datetime
import os
import sys

ติดตั้งตามลำดับนี้
pip install numpy
pip install opencv-python
pip install torch torchvision torchaudio
pip install requests

git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7
pip install -r requirements.txt

ลำดับ	ไลบรารี่	ใช้ทำอะไร
1	Python	ภาษาหลัก
2	NumPy	จัดการข้อมูลภาพ
3	OpenCV	ประมวลผลภาพและวิดีโอ
4	PyTorch	รันโมเดล Deep Learning
5	YOLOv5 / YOLOv7	ตรวจจับบุคคล
6	Requests	แจ้งเตือนผ่าน Telegram
7	time, datetime	ควบคุมเวลา
8	os, sys	จัดการระบบ
