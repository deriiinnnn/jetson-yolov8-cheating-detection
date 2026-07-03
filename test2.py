from ultralytics import YOLO
import cv2
import os

# Load model
model = YOLO("models/best_yolov8n_seg_cheating.pt")

# Predict
results = model("test.jpg")

# Buat folder hasil
os.makedirs("result", exist_ok=True)

for r in results:
    img = r.orig_img

    # Copy gambar asli
    cheating_img = img.copy()

    for i, cls in enumerate(r.boxes.cls):
        class_name = model.names[int(cls)]
        x1, y1, x2, y2 = map(int, r.boxes.xyxy[i])

        # Hanya tampilkan cheating
        if class_name == "cheating":
            cv2.rectangle(
                cheating_img,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )

            cv2.putText(
                cheating_img,
                "cheating",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

    # Simpan hasil
    cv2.imwrite("result/cheating.jpg", cheating_img)

print("Deteksi cheating selesai")