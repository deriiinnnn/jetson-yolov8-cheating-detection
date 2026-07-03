import cv2

url = "http://192.168.18.18:8081/video"

cap = cv2.VideoCapture(url)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Gagal membaca frame")
        break

    cv2.imshow("iPhone Camera", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()