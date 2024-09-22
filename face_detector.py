import numpy as np
import cv2
from emotion_detector import predict_emotion

face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")


def detect_face_coordinates(img):
    """
    Return the coordinates of the faces in the gray image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # High scale factor for the speed of the program
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    return faces


def detect_faces(img):
    """
    Detect faces in the image with rectangles and predictions.
    """

    # Draw rectangles around the faces
    faces = detect_face_coordinates(img)

    # Draw the predictions
    for x, y, w, h in faces:
        face = img[y : y + h, x : x + w]
        emotion, color, prob = predict_emotion(face)
        cv2.putText(
            img,
            f"{emotion} {prob:.0f}%",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    return img
