import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd

MODEL_PATH = "models/model.h5"

model = load_model(MODEL_PATH)

# colors are in BGR format
emotion_dict = {
    0: ("Angry", (0, 0, 255)),
    1: ("Fear", (255, 255, 255)),
    2: ("Happy", (0, 255, 0)),
    3: ("Neutral", (128, 128, 128)),
    4: ("Sad", (139, 0, 0),),
    5: ("Surprised", (255, 255, 0)),
}
    

def preprocess_image(face):
    """
    Scale the image to 48x48x3 pixels.
    """
    resized = cv2.resize(face, (48, 48))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, 48, 48, 3))
    
    return reshaped
    
    

def predict_emotion(face):
    """
    Predict the emotion of the face with probabilities.
    """
    preprocessed = preprocess_image(face)
    prediction = model.predict(preprocessed)
    emotion, color = emotion_dict[np.argmax(prediction)]
    prob = np.max(prediction) * 100
    
    return emotion, color, prob


if __name__ == "__main__":
    face = cv2.imread("data/train/happy/Training_1206.jpg")
    emotion, color, prob = predict_emotion(face)
    print(f"Emotion: {emotion} - Probability: {prob:.2f}%")
