from ultralytics import YOLO
import cv2
import os
import numpy as np

# Load model
model = YOLO("models/best_yolov8n_seg_cheating.pt")

# Predict
results = model("test.jpg")

# Buat folder output
os.makedirs("result/cheating", exist_ok=True)
os.makedirs("result/normal", exist_ok=True)

for r in results:
    img = r.orig_img.copy()

    for i, cls in enumerate(r.boxes.cls):
        class_name = model.names[int(cls)]

        # Ambil bounding box
        x1, y1, x2, y2 = map(int, r.boxes.xyxy[i])

        # Crop object
        crop = img[y1:y2, x1:x2]

        if class_name == "cheating":
            cv2.imwrite(f"result/cheating/cheating_{i}.jpg", crop)

        elif class_name == "normal":
            cv2.imwrite(f"result/normal/normal_{i}.jpg", crop)

print("Semua hasil dipisahkan")