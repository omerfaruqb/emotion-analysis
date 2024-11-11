import cv2
import numpy as np
import onnx
import onnxruntime as rt

MODEL_PATH = "models/resnet18_v2.onnx"

model = onnx.load(MODEL_PATH)
onnx.checker.check_model(model)
ort_session = rt.InferenceSession(MODEL_PATH)
input_name = ort_session.get_inputs()[0].name


# Emotion dictionary with the corresponding color in BGR
emotion_dict = {
    0: ("Surprised", (255, 255, 0)), 
    1: ("Fear", (255, 255, 255)),
    2: ("Disgust", (50, 50, 50)), 
    3: ("Happy", (0, 255, 0)),
    4: ("Sad", (139, 0, 0),),
    5: ('Angry',  (0, 0, 255)),
    6: ('Neutral', (128, 128, 128))
}
    

def preprocess_image(face):
    """
    Scale the image to 100x100x3 pixels and preprocess.
    """
    resized = cv2.resize(face, (100, 100))
    transformed = ((resized / 255.0) - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    transformed = transformed.astype(np.float32)
    
    reshaped = np.transpose(transformed, (2, 0, 1)) # HWC to CHW
    reshaped = np.expand_dims(reshaped, axis=0) # (3, 100, 100) to (1, 3, 100, 100)
    
    return reshaped


# Define class weights
class_weights = np.array([1.0, 1.2, 1.5, 1.0, 1.3, 1.0, 1.0])

def predict_emotion(face):
    """
    Predict the emotion of the face with probabilities.
    """
    preprocessed = preprocess_image(face)
    ort_inputs = {input_name: preprocessed}
    outputs = ort_session.run(None, ort_inputs)
    probs = np.exp(outputs[0]) / np.sum(np.exp(outputs[0]))
    
    # Adjust probabilities with class weights
    weighted_probs = probs * class_weights
    weighted_probs /= np.sum(weighted_probs)
    
    emotion, color = emotion_dict[np.argmax(weighted_probs)]
    prob = np.max(weighted_probs) * 100
    
    return emotion, color, prob


if __name__ == "__main__":
    face = cv2.imread("data/val/2/train_01157_aligned.jpg")
    emotion, color, prob = predict_emotion(face)
    print(f"Emotion: {emotion} - Probability: {prob:.2f}%")
