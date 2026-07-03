from ultralytics import YOLO
import cv2

# Load model
model = YOLO(
    "/home/jetson/cheating/models/best_yolov8n_seg_cheating.pt"
)

# Kamera HP
url = "http://192.168.18.80:8080/video"

cap = cv2.VideoCapture(url)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Gagal membaca frame")
        break

    # Inferensi YOLO
    results = model(frame, verbose=False)

    # Gambar bounding box
    annotated_frame = results[0].plot()

    cv2.imshow("Cheating Detection", annotated_frame)

    # Tekan ESC untuk keluar
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()