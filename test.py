from ultralytics import YOLO

# Load model segmentation
model = YOLO("models/best_yolov8n_seg_cheating.pt")

# Predict image
results = model.predict(
    source="test.jpg",
    imgsz=640,
    conf=0.25,
    save=True,
    project="result",
    name="predict"
)

print("Inference selesai")