import cv2
import numpy as np
from face_detector import detect_faces


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open the camera.")
    exit()

while True:
    success, img = cap.read()

    if not success:
        print("Could not read the camera.")
        break

    img = detect_faces(img)

    cv2.imshow("Webcam", img)

    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
